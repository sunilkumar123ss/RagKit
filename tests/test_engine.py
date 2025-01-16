"""Core regressions use explicit test doubles; Qdrant tests use the real local SDK."""
from concurrent.futures import ThreadPoolExecutor
import builtins
import importlib.util
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from ragkit.engine import RagKit
from ragkit.retrieval import bm25, chunk_text, reciprocal_rank_fusion
from ragkit.vectors import QdrantVectorStore, validate_vectors


class TestEmbeddings:
    """A deterministic contract test double, not a semantic embedding model."""
    model_name = "test-double-not-a-semantic-model"

    def embed(self, texts):
        return [[float("cat" in text.lower() or "feline" in text.lower()),
                 float("dog" in text.lower() or "canine" in text.lower()), 0.1]
                for text in texts]


class MemoryVectors:
    def __init__(self):
        self.collections = {}
        self.fail_build = False

    def build(self, collection, chunks, vectors):
        self.collections[collection] = list(zip(chunks, vectors))
        if self.fail_build:
            raise RuntimeError("simulated failed index write")

    def search(self, collection, vector, limit, document_ids=None):
        results = []
        for chunk, candidate in self.collections[collection]:
            if document_ids is not None and chunk["document_id"] not in document_ids:
                continue
            score = sum(a * b for a, b in zip(vector, candidate))
            score /= math.sqrt(sum(a * a for a in vector) * sum(b * b for b in candidate))
            results.append((chunk["id"], score))
        return sorted(results, key=lambda item: (-item[1], item[0]))[:limit]

    def exists(self, collection):
        return collection in self.collections

    def delete(self, collection):
        self.collections.pop(collection, None)

    def close(self):
        pass


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = RagKit(self.temp.name)

    def tearDown(self):
        self.engine.close()
        self.temp.cleanup()

    def test_stable_id_idempotency_and_persistence(self):
        first = self.engine.ingest_text("Exact café 🦊 text\r\n", "Example", source="notes.md")
        repeated = self.engine.ingest_text("Exact café 🦊 text\r\n", "Example", source="notes.md")
        self.assertEqual(first["id"], repeated["id"])
        self.assertFalse(first["unchanged"])
        self.assertTrue(repeated["unchanged"])
        self.assertEqual(first["updated_at"], repeated["updated_at"])
        self.assertEqual(self.engine.stats()["revision"], 1)
        self.engine.close()
        self.engine = RagKit(self.temp.name)
        self.assertEqual(self.engine.get_document(first["id"])["text"], "Exact café 🦊 text\r\n")
        self.assertEqual(self.engine.search("CAFÉ")["results"][0]["document_id"], first["id"])

    def test_replacement_removes_old_chunks_and_advances_revision(self):
        self.engine.ingest_text("cat " * 900, "Original", document_id="same")
        old_chunks = self.engine.stats()["chunks"]
        self.engine.ingest_text("A dog arrives.", "Replacement", document_id="same")
        self.assertGreater(old_chunks, self.engine.stats()["chunks"])
        self.assertTrue(self.engine.search("cat")["abstained"])
        self.assertEqual(self.engine.search("dog")["results"][0]["title"], "Replacement")
        self.assertEqual(self.engine.stats()["revision"], 2)

    def test_transaction_failure_keeps_previous_document_and_revision(self):
        self.engine.ingest_text("Original cat.", "Original", document_id="same")
        document = self.engine.get_document("same")
        document.update(text="Replacement dog.", title="Replacement")
        chunks = chunk_text(document["text"], "same")
        with self.assertRaises(sqlite3.IntegrityError):
            self.engine._store.replace(document, chunks + chunks)
        self.assertEqual(self.engine.get_document("same")["text"], "Original cat.")
        self.assertEqual(self.engine.stats()["revision"], 1)

    def test_exact_unicode_offsets_and_embedded_null_character_count(self):
        original = ("Café 🦊 evidence\r\nwith more exact text. " * 100) + "\x00tail"
        result = self.engine.ingest_text(original, "Unicode", document_id="unicode")
        self.assertGreater(result["chunks"], 1)
        self.assertEqual(self.engine.list_documents()[0]["characters"], len(original))
        for item in self.engine.search("evidence", top_k=20)["results"]:
            citation = item["citation"]
            self.assertEqual(original[citation["start"]:citation["end"]], citation["quote"])
            self.assertEqual(item["text"], citation["quote"])

    def test_delete_is_idempotent_and_cascades(self):
        self.engine.ingest_text("cat " * 900, "Many chunks", document_id="delete-me")
        self.assertTrue(self.engine.delete_document("delete-me"))
        self.assertFalse(self.engine.delete_document("delete-me"))
        self.assertEqual(self.engine.stats()["chunks"], 0)
        self.assertEqual(self.engine.stats()["revision"], 2)
        with self.assertRaises(KeyError):
            self.engine.get_document("delete-me")

    def test_scope_never_returns_other_documents(self):
        self.engine.ingest_text("cat cat cat", "First", document_id="first")
        self.engine.ingest_text("cat dog", "Second", document_id="second")
        results = self.engine.search("cat", document_ids=["second"])["results"]
        self.assertEqual({result["document_id"] for result in results}, {"second"})
        self.assertTrue(self.engine.search("cat", document_ids=[])["abstained"])
        self.assertTrue(self.engine.search("cat", document_ids=["missing"])["abstained"])

    def test_abstention_empty_and_no_terms(self):
        self.assertTrue(self.engine.search("query")["abstained"])
        self.engine.ingest_text("ordinary text", "Document")
        self.assertEqual(self.engine.search("zebracorn")["results"], [])
        with self.assertRaises(ValueError):
            self.engine.search("?!")

    def test_input_validation(self):
        for text in (None, "", " \n ", 42, "x" * 1_000_001):
            with self.subTest(text_type=type(text).__name__), self.assertRaises(ValueError):
                self.engine.ingest_text(text, "Title")
        for title in (None, "", "  ", "a" * 501):
            with self.assertRaises(ValueError):
                self.engine.ingest_text("text", title)
        for identifier in ("../escape", "", "bad id", "a" * 129, True):
            with self.assertRaises(ValueError):
                self.engine.ingest_text("text", "title", document_id=identifier)
        for options in ({"top_k": True}, {"top_k": 0}, {"top_k": 21}, {"mode": "fake"},
                        {"rerank": 1}, {"document_ids": "a"}, {"document_ids": ["../escape"]}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.engine.search("text", **options)
        with self.assertRaises(ValueError):
            self.engine.search("x" * 2001)
        with self.assertRaises(ValueError):
            self.engine.ingest_text("text", "title", source="")

    def test_threaded_ingestion_and_reads_share_safe_connection(self):
        def task(index):
            self.engine.ingest_text("Concurrent evidence %d" % index, "Thread", document_id=str(index))
            self.engine.search("evidence")
        with ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(task, range(18)))
        self.assertEqual(self.engine.stats()["documents"], 18)
        self.assertEqual(self.engine.stats()["revision"], 18)

    def test_close_is_idempotent_and_disallows_use(self):
        self.engine.close()
        self.engine.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.engine.search("anything")

    def test_lexical_mode_never_instantiates_optional_clients(self):
        self.engine.close()
        def unavailable(_):
            raise AssertionError("optional client should not be opened")
        self.engine = RagKit(self.temp.name, vector_store_factory=unavailable)
        self.engine.ingest_text("Offline evidence", "Example")
        self.assertFalse(self.engine.search("evidence")["abstained"])
        self.assertFalse(self.engine.stats()["dense_ready"])

    def test_missing_optional_dependencies_produce_install_instructions(self):
        self.engine.ingest_text("Offline cat evidence", "Example")
        real_import = builtins.__import__
        def without_optional(name, *args, **kwargs):
            if name in ("qdrant_client", "sentence_transformers"):
                raise ImportError("simulated optional dependency absence")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=without_optional):
            with self.assertRaisesRegex(RuntimeError, "optional 'dense' dependencies"):
                self.engine.build_dense_index()
            with self.assertRaisesRegex(RuntimeError, "optional 'dense' dependencies"):
                self.engine.search("cat", rerank=True)


class RetrievalTests(unittest.TestCase):
    def test_chunking_preserves_every_nonwhitespace_character(self):
        text = "  " + ("one café 🦊\r\n" * 80) + "no-space-tail" * 30
        chunks = chunk_text(text, "doc", size=80, overlap=15)
        covered = set()
        for chunk in chunks:
            self.assertLessEqual(len(chunk["text"]), 80)
            self.assertEqual(text[chunk["start"]:chunk["end"]], chunk["text"])
            covered.update(range(chunk["start"], chunk["end"]))
        self.assertTrue(all(i in covered for i, char in enumerate(text) if not char.isspace()))
        self.assertEqual(chunks, chunk_text(text, "doc", size=80, overlap=15))

    def test_chunk_parameters_cannot_create_nonprogress_loop(self):
        for options in ({"size": 0}, {"size": True}, {"size": 32, "overlap": 16}, {"overlap": -1}):
            with self.assertRaises(ValueError):
                chunk_text("some text", "doc", **options)

    def test_bm25_expected_single_document_formula_and_tie_order(self):
        score = bm25([{"id": "a", "text": "cat cat dog"}], "cat")[0][1]
        self.assertAlmostEqual(score, math.log(1 + 0.5 / 1.5) * 2 * 2.5 / (2 + 1.5))
        rows = [{"id": "z", "text": "cat"}, {"id": "a", "text": "cat"}]
        self.assertEqual([key for key, _ in bm25(rows, "cat")], ["a", "z"])
        self.assertEqual(bm25(rows, "cat"), bm25(list(reversed(rows)), "cat cat"))
        self.assertEqual(bm25([{"id": "a", "text": "!!!"}], "cat"), [])

    def test_fusion_combines_overlap_and_rejects_duplicate_candidates(self):
        ranking = reciprocal_rank_fusion([[('a', 1), ('b', .5)], [('c', 1), ('b', .2)]])
        self.assertEqual(ranking[0][0], "b")
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[('a', 1), ('a', 1)]])


class DenseContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.vectors = MemoryVectors()
        self.engine = RagKit(self.temp.name, embedding_provider=TestEmbeddings(),
                             vector_store_factory=lambda _: self.vectors)
        self.engine.ingest_text("The cat likes naps.", "Cats", document_id="cats")
        self.engine.ingest_text("The dog likes walks.", "Dogs", document_id="dogs")

    def tearDown(self):
        self.engine.close()
        self.temp.cleanup()

    def test_dense_requires_explicit_build_and_preserves_scope(self):
        with self.assertRaisesRegex(RuntimeError, "missing or stale"):
            self.engine.search("feline", mode="dense")
        built = self.engine.build_dense_index()
        self.assertEqual(built, {"chunks": 2, "revision": 2, "model": TestEmbeddings.model_name})
        self.assertTrue(self.engine.stats()["dense_ready"])
        self.assertEqual(self.engine.search("feline", mode="dense")["results"][0]["document_id"], "cats")
        scoped = self.engine.search("feline", mode="dense", document_ids=["dogs"])
        self.assertEqual({item["document_id"] for item in scoped["results"]}, {"dogs"})

    def test_hybrid_contains_component_scores_and_exact_citations(self):
        self.engine.build_dense_index()
        results = self.engine.search("cat canine", mode="hybrid")["results"]
        self.assertEqual(len(results), 2)
        self.assertTrue(any(item["lexical_score"] > 0 for item in results))
        self.assertTrue(all(item["dense_score"] > 0 for item in results))
        for item in results:
            document = self.engine.get_document(item["document_id"])
            cite = item["citation"]
            self.assertEqual(document["text"][cite["start"]:cite["end"]], cite["quote"])

    def test_identical_ingest_retains_index_but_update_and_delete_make_it_stale(self):
        self.engine.build_dense_index()
        self.engine.ingest_text("The cat likes naps.", "Cats", document_id="cats")
        self.assertTrue(self.engine.stats()["dense_ready"])
        self.engine.ingest_text("The cat likes boxes.", "Cats", document_id="cats")
        self.assertFalse(self.engine.stats()["dense_ready"])
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.engine.search("cat", mode="hybrid")
        self.engine.build_dense_index()
        self.assertEqual(len(self.vectors.collections), 1)
        self.engine.delete_document("dogs")
        self.assertFalse(self.engine.stats()["dense_ready"])

    def test_model_change_invalidates_persisted_index(self):
        self.engine.build_dense_index()
        self.engine.close()
        other_model = TestEmbeddings()
        other_model.model_name = "different-model"
        self.engine = RagKit(self.temp.name, embedding_provider=other_model,
                             vector_store_factory=lambda _: self.vectors)
        self.assertFalse(self.engine.stats()["dense_ready"])
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.engine.search("cat", mode="dense")

    def test_failed_rebuild_keeps_previous_complete_index(self):
        self.engine.build_dense_index()
        before = self.engine._store.dense_state()
        self.vectors.fail_build = True
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            self.engine.build_dense_index()
        self.assertEqual(before, self.engine._store.dense_state())
        self.assertEqual(len(self.vectors.collections), 1)
        self.assertFalse(self.engine.search("cat", mode="dense")["abstained"])

    def test_build_detects_cross_connection_corpus_change(self):
        original_build = self.vectors.build
        def concurrent_write(*args):
            original_build(*args)
            with RagKit(self.temp.name) as other:
                other.ingest_text("New cat evidence", "New", document_id="new")
        with patch.object(self.vectors, "build", side_effect=concurrent_write):
            with self.assertRaisesRegex(RuntimeError, "corpus changed"):
                self.engine.build_dense_index()
        self.assertFalse(self.engine.stats()["dense_ready"])
        self.assertEqual(len(self.vectors.collections), 0)

    def test_dimension_mismatch_nonfinite_and_zero_vectors_fail_clearly(self):
        for vectors, count, dimension in (([[1, 2], [1]], 2, None), ([[float('nan')]], 1, None),
                                           ([[0, 0]], 1, None), ([[1, 2]], 1, 3), ([], 1, None)):
            with self.subTest(vectors=vectors), self.assertRaises(RuntimeError):
                validate_vectors(vectors, count, dimension)
        self.engine.build_dense_index()
        with patch.object(self.engine._embedding, "embed", return_value=[[1, 2]]):
            with self.assertRaisesRegex(RuntimeError, "dimensions"):
                self.engine.search("cat", mode="dense")

    def test_missing_collection_or_out_of_scope_result_is_not_silent_fallback(self):
        self.engine.build_dense_index()
        with patch.object(self.vectors, "search", return_value=[("unknown-chunk", .9)]):
            with self.assertRaisesRegex(RuntimeError, "out-of-scope"):
                self.engine.search("cat", mode="hybrid")
        self.vectors.collections.clear()
        self.assertFalse(self.engine.stats()["dense_ready"])
        with self.assertRaisesRegex(RuntimeError, "collection is missing"):
            self.engine.search("cat", mode="dense")

    def test_reranker_changes_order_and_records_both_scores(self):
        self.engine.build_dense_index()
        class Reranker:
            def score(self, query, texts):
                return [10.0 if "dog" in text else -1.0 for text in texts]
        self.engine._reranker = Reranker()
        items = self.engine.search("feline", mode="dense", rerank=True)["results"]
        self.assertEqual(items[0]["document_id"], "dogs")
        self.assertEqual(items[0]["score"], 10)
        self.assertGreater(items[0]["retrieval_score"], 0)
        self.assertEqual(items[0]["rerank_score"], 10)

    def test_reranker_invalid_output_and_no_evidence_behavior(self):
        class BadReranker:
            def score(self, query, texts):
                return [float('nan')]
        self.engine._reranker = BadReranker()
        self.assertTrue(self.engine.search("zebracorn", rerank=True)["abstained"])
        with self.assertRaisesRegex(RuntimeError, "finite score"):
            self.engine.search("cat", rerank=True)


@unittest.skipUnless(importlib.util.find_spec("qdrant_client"), "optional qdrant-client is not installed")
class RealQdrantTests(unittest.TestCase):
    def test_real_local_index_persists_and_filters_with_injected_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            with RagKit(directory, embedding_provider=TestEmbeddings()) as engine:
                engine.ingest_text("cat cat", "Cats", document_id="cats")
                engine.ingest_text("dog dog", "Dogs", document_id="dogs")
                engine.build_dense_index()
                self.assertEqual(engine.search("feline", mode="dense")["results"][0]["document_id"], "cats")
                filtered = engine.search("feline", mode="hybrid", document_ids=["dogs"])
                self.assertEqual([item["document_id"] for item in filtered["results"]], ["dogs"])
            with RagKit(directory, embedding_provider=TestEmbeddings()) as reopened:
                self.assertTrue(reopened.stats()["dense_ready"])
                self.assertEqual(reopened.search("canine", mode="dense")["results"][0]["document_id"], "dogs")
                reopened.ingest_text("cat dog", "Both", document_id="both")
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    reopened.search("canine", mode="dense")


if __name__ == "__main__":
    unittest.main()
