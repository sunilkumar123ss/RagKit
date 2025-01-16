"""Persistent local RAG retrieval with explicit optional semantic indexing."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import uuid

from .retrieval import bm25, chunk_text, reciprocal_rank_fusion, tokenize
from .storage import CorpusStore
from .vectors import (CrossEncoderReranker, QdrantVectorStore,
                      SentenceTransformerEmbeddings, validate_vectors)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
MAX_DOCUMENT_CHARACTERS = 1_000_000
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def validate_id(document_id):
    if not isinstance(document_id, str) or not _ID.fullmatch(document_id):
        raise ValueError("document id must be 1..128 letters, digits, dots, underscores, colons or hyphens")
    return document_id


class RagKit:
    """A single engine serializes reads, writes and models across API threads.

    Injected providers are a testing/extension seam, never a production fallback:
    embedding_provider.model_name and .embed(texts); reranker_provider.score(query,
    texts); vector_store_factory(path) with build/search/exists/delete/close methods.
    Use one application process per data directory when using Qdrant local mode.
    """
    def __init__(self, data_dir: str | Path, *, embedding_provider=None,
                 vector_store_factory=None, reranker_provider=None):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self._store = CorpusStore(self.data_dir / "corpus.sqlite3", self._lock)
        self.embedding_model = (getattr(embedding_provider, "model_name", None)
                                or os.getenv("RAGKIT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
        self.reranker_model = os.getenv("RAGKIT_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
        self._embedding = embedding_provider or SentenceTransformerEmbeddings(self.embedding_model)
        self._reranker = reranker_provider or CrossEncoderReranker(self.reranker_model)
        self._vector_store_factory = vector_store_factory or QdrantVectorStore
        self._vectors = None

    def _open(self):
        if self._closed:
            raise RuntimeError("RagKit is closed")

    def _vector_store(self):
        if self._vectors is None:
            self._vectors = self._vector_store_factory(self.data_dir / "qdrant")
        return self._vectors

    def ingest_text(self, text, title, source=None, document_id=None) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonblank string")
        if len(text) > MAX_DOCUMENT_CHARACTERS:
            raise ValueError("document text exceeds one million characters")
        if not isinstance(title, str) or not title.strip() or len(title) > 500:
            raise ValueError("title must be a nonblank string of at most 500 characters")
        if source is not None and (not isinstance(source, str) or not source.strip() or len(source) > 2048):
            raise ValueError("source must be null or a nonblank string of at most 2048 characters")
        title = title.strip()
        source = source.strip() if source is not None else None
        if document_id is None:
            identity = json.dumps([title, source, text], ensure_ascii=False, separators=(",", ":"))
            document_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
        validate_id(document_id)
        chunks = chunk_text(text, document_id)
        document = {"id": document_id, "title": title, "source": source, "text": text,
                    "updated_at": datetime.now(timezone.utc).isoformat()}
        with self._lock:
            self._open()
            saved, unchanged = self._store.replace(document, chunks)
            saved.pop("text")
            saved["unchanged"] = unchanged
            return saved

    def list_documents(self) -> list[dict]:
        with self._lock:
            self._open()
            return self._store.list_documents()

    def get_document(self, document_id) -> dict:
        validate_id(document_id)
        with self._lock:
            self._open()
            return self._store.get_document(document_id)

    def delete_document(self, document_id) -> bool:
        validate_id(document_id)
        with self._lock:
            self._open()
            return self._store.delete(document_id)

    def _scope(self, document_ids):
        if document_ids is None:
            return None
        if not isinstance(document_ids, list) or len(document_ids) > 500:
            raise ValueError("document_ids must be a list of at most 500 ids")
        return sorted(set(validate_id(value) for value in document_ids))

    def _dense_state(self):
        state = self._store.dense_state()
        if not state or state["revision"] != self._store.revision() or state["model"] != self.embedding_model:
            raise RuntimeError("dense index is missing or stale; build the dense index before dense/hybrid search")
        return state

    def search(self, query, top_k=5, mode="lexical", rerank=False, document_ids=None) -> dict:
        if not isinstance(query, str) or not tokenize(query) or len(query) > 2000:
            raise ValueError("query must contain searchable words and be at most 2000 characters")
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer in 1..20")
        if mode not in ("lexical", "dense", "hybrid"):
            raise ValueError("mode must be lexical, dense or hybrid")
        if type(rerank) is not bool:
            raise ValueError("rerank must be boolean")
        scope = self._scope(document_ids)
        with self._lock:
            self._open()
            state = self._dense_state() if mode != "lexical" else None
            chunks = self._store.chunks(scope)
            by_id = {chunk["id"]: chunk for chunk in chunks}
            lexical = bm25(chunks, query) if mode != "dense" else []
            dense = []
            if state and chunks:
                store = self._vector_store()
                if not store.exists(state["collection"]):
                    raise RuntimeError("dense index collection is missing; rebuild the dense index")
                vector = validate_vectors(self._embedding.embed([query]), 1, state["dimensions"])[0]
                hits = store.search(state["collection"], vector, len(chunks), scope)
                seen = set()
                for chunk_id, score in hits:
                    if chunk_id not in by_id or chunk_id in seen or not math.isfinite(score):
                        raise RuntimeError("dense index returned invalid or out-of-scope results; rebuild the index")
                    seen.add(chunk_id)
                    if score > 0:
                        dense.append((chunk_id, float(score)))
                dense.sort(key=lambda item: (-item[1], item[0]))
            if mode == "lexical":
                ranking = lexical
            elif mode == "dense":
                ranking = dense
            else:
                ranking = reciprocal_rank_fusion([lexical, dense])
            lexical_scores, dense_scores = dict(lexical), dict(dense)
            candidates = ranking[:max(20, top_k * 4)] if rerank else ranking[:top_k]
            rerank_scores = {}
            if rerank and candidates:
                try:
                    scores = [float(value) for value in self._reranker.score(
                        query, [by_id[key]["text"] for key, _ in candidates])]
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError("reranker returned invalid scores") from exc
                if len(scores) != len(candidates) or any(not math.isfinite(score) for score in scores):
                    raise RuntimeError("reranker must return one finite score per candidate")
                rerank_scores = dict(zip((key for key, _ in candidates), scores))
                candidates = sorted(candidates, key=lambda item: (-rerank_scores[item[0]], item[0]))[:top_k]
            results = []
            for chunk_id, score in candidates:
                chunk = by_id[chunk_id]
                result = {"chunk_id": chunk_id, "document_id": chunk["document_id"],
                          "title": chunk["title"], "source": chunk["source"], "text": chunk["text"],
                          "score": rerank_scores.get(chunk_id, score),
                          "lexical_score": lexical_scores.get(chunk_id, 0.0),
                          "dense_score": dense_scores.get(chunk_id, 0.0),
                          "citation": {"document_id": chunk["document_id"], "field": "text",
                                       "start": chunk["start"], "end": chunk["end"], "quote": chunk["text"]}}
                if rerank:
                    result["retrieval_score"] = score
                    result["rerank_score"] = rerank_scores[chunk_id]
                results.append(result)
            return {"query": query, "mode": mode, "abstained": not bool(results), "results": results}

    def build_dense_index(self) -> dict:
        with self._lock:
            self._open()
            revision = self._store.revision()
            chunks = self._store.chunks()
            if not chunks:
                raise ValueError("add at least one document before building the dense index")
            # Check the vector-store dependency before potentially downloading a model.
            store = self._vector_store()
            vectors = validate_vectors(self._embedding.embed([chunk["text"] for chunk in chunks]), len(chunks))
            collection = "ragkit_" + uuid.uuid4().hex
            previous = self._store.dense_state()
            try:
                store.build(collection, chunks, vectors)
                state = {"revision": revision, "model": self.embedding_model,
                         "dimensions": len(vectors[0]), "collection": collection, "chunks": len(chunks)}
                self._store.set_dense_state(state, revision)
            except Exception:
                try:
                    store.delete(collection)
                except Exception:
                    pass
                raise
            if previous:
                try:
                    store.delete(previous["collection"])
                except Exception:
                    pass  # A leftover collection does not invalidate the complete new generation.
            return {"chunks": len(chunks), "model": self.embedding_model, "revision": revision}

    def stats(self) -> dict:
        with self._lock:
            self._open()
            documents = self._store.list_documents()
            revision = self._store.revision()
            state = self._store.dense_state()
            ready = bool(state and state["revision"] == revision and state["model"] == self.embedding_model)
            if ready:
                try:
                    ready = self._vector_store().exists(state["collection"])
                except Exception:
                    # Status remains usable without optional SDKs or if the local
                    # index is locked/corrupt. Search/build surfaces the precise error.
                    ready = False
            return {"documents": len(documents), "chunks": sum(doc["chunks"] for doc in documents),
                    "dense_ready": ready, "revision": revision, "embedding_model": self.embedding_model}

    def close(self):
        with self._lock:
            if not self._closed:
                try:
                    if self._vectors is not None:
                        self._vectors.close()
                finally:
                    self._store.close()
                    self._closed = True

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()
