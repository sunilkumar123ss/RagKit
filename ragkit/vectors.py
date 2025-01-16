"""Lazy, real optional model and Qdrant adapters; no synthetic semantic fallback."""
from __future__ import annotations

import math
from pathlib import Path
import uuid


def validate_vectors(vectors, count: int, dimension: int | None = None) -> list[list[float]]:
    try:
        output = [[float(value) for value in row] for row in vectors]
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("embedding provider returned invalid vectors") from exc
    if len(output) != count:
        raise RuntimeError("embedding provider returned the wrong number of vectors")
    expected = dimension if dimension is not None else (len(output[0]) if output else 0)
    for row in output:
        if not expected or len(row) != expected or any(not math.isfinite(value) for value in row):
            raise RuntimeError("embedding vectors must have consistent dimensions and finite values")
        if not any(value != 0 for value in row):
            raise RuntimeError("embedding provider returned a zero vector")
    return output


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, trust_remote_code=False)
            result = self._model.encode(texts, batch_size=32, show_progress_bar=False,
                                        convert_to_numpy=True, normalize_embeddings=True)
            return result.tolist()
        except ImportError as exc:
            raise RuntimeError("dense retrieval needs the optional 'dense' dependencies: pip install '.[dense]'") from exc
        except Exception as exc:
            raise RuntimeError("embedding model could not run; check the model name, model download and local resources") from exc


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, trust_remote_code=False)
            return self._model.predict([(query, text) for text in texts],
                                       show_progress_bar=False, convert_to_numpy=True).tolist()
        except ImportError as exc:
            raise RuntimeError("cross-encoder reranking needs the optional 'dense' dependencies: pip install '.[dense]'") from exc
        except Exception as exc:
            raise RuntimeError("cross-encoder model could not run; check the model name, model download and local resources") from exc


class QdrantVectorStore:
    """One local Qdrant client per engine; callers serialize operations."""
    def __init__(self, path: Path):
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RuntimeError("dense retrieval needs the optional 'dense' dependencies: pip install '.[dense]'") from exc
        self.models = models
        try:
            self.client = QdrantClient(path=str(path), force_disable_check_same_thread=True)
        except Exception as exc:
            raise RuntimeError("cannot open the local dense index; use one server process per data directory") from exc

    def build(self, collection: str, chunks: list[dict], vectors: list[list[float]]):
        vectors = validate_vectors(vectors, len(chunks))
        if not vectors:
            raise ValueError("cannot build a dense index without chunks")
        self.client.create_collection(collection_name=collection, vectors_config=self.models.VectorParams(
            size=len(vectors[0]), distance=self.models.Distance.COSINE))
        for offset in range(0, len(chunks), 64):
            points = [self.models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, "ragkit:" + chunk["id"])),
                vector=vector,
                payload={"chunk_id": chunk["id"], "document_id": chunk["document_id"]})
                for chunk, vector in zip(chunks[offset:offset + 64], vectors[offset:offset + 64])]
            self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, collection: str, vector: list[float], limit: int,
               document_ids=None) -> list[tuple[str, float]]:
        query_filter = None
        if document_ids is not None:
            if not document_ids:
                return []
            query_filter = self.models.Filter(must=[self.models.FieldCondition(
                key="document_id", match=self.models.MatchAny(any=document_ids))])
        hits = self.client.query_points(collection_name=collection, query=vector,
                                        query_filter=query_filter, limit=limit,
                                        with_payload=True, with_vectors=False).points
        return [(hit.payload["chunk_id"], float(hit.score)) for hit in hits]

    def exists(self, collection: str) -> bool:
        return self.client.collection_exists(collection_name=collection)

    def delete(self, collection: str):
        if self.exists(collection):
            self.client.delete_collection(collection_name=collection)

    def close(self):
        self.client.close()
