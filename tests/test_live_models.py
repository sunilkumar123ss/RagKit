"""Opt-in tests using actual downloaded models and the real persistent Qdrant SDK."""
import os
import tempfile
import unittest

import pytest

from ragkit.engine import RagKit


@pytest.mark.live_models
@unittest.skipUnless(os.environ.get("RAGKIT_RUN_LIVE_MODELS") == "1", "set RAGKIT_RUN_LIVE_MODELS=1 to download/run actual models")
class ActualModelTests(unittest.TestCase):
    def test_hybrid_reranking_persistence_and_exact_citations(self):
        with tempfile.TemporaryDirectory() as directory:
            with RagKit(directory) as engine:
                engine.ingest_text("Approved business expenses are reimbursed within ten business days.", "Finance", "Synthetic", "finance")
                engine.ingest_text("Teammates may take 24 days of vacation each year.", "Time off", "Synthetic", "leave")
                engine.ingest_text("The kitchen closes at six in the evening.", "Office", "Synthetic", "office")
                built = engine.build_dense_index()
                self.assertEqual(built["chunks"], 3)
                result = engine.search("How long must I wait to get my travel costs paid back?", mode="hybrid", rerank=True, top_k=2)
                self.assertEqual(result["results"][0]["document_id"], "finance")
                self.assertTrue(engine.stats()["dense_ready"])
                for row in result["results"]:
                    cite = row["citation"]
                    doc = engine.get_document(cite["document_id"])
                    self.assertEqual(doc["text"][cite["start"]:cite["end"]], cite["quote"])
            with RagKit(directory) as reopened:
                self.assertTrue(reopened.stats()["dense_ready"])
                result = reopened.search("time away from work", mode="dense", document_ids=["leave"], top_k=1)
                self.assertEqual(result["results"][0]["document_id"], "leave")
                reopened.delete_document("leave")
                self.assertFalse(reopened.stats()["dense_ready"])
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    reopened.search("holiday", mode="hybrid")
