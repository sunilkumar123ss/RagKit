import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ragkit.evaluation import ranking_metrics


class MetricTests(unittest.TestCase):
    def test_document_deduplication_and_multiple_relevant_sources(self):
        result = ranking_metrics(["wrong", "a", "a", "b"], ["a", "b"], 3)
        self.assertEqual(result["recall"], 1)
        self.assertEqual(result["reciprocal_rank"], 0.5)
        self.assertAlmostEqual(result["ndcg"], (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3)))

    def test_missing_hits_and_truncation(self):
        self.assertEqual(ranking_metrics(["x", "a"], ["a"], 1), {"recall": 0, "reciprocal_rank": 0, "ndcg": 0})
        with self.assertRaises(ValueError):
            ranking_metrics([], [], 3)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "corpus"

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args, code=0):
        result = subprocess.run([sys.executable, "-m", "ragkit", "--data-dir", str(self.database), *args],
                                cwd=self.root, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, code, result.stderr + result.stdout)
        return json.loads(result.stdout if code == 0 else result.stderr)

    def test_sample_query_reload_delete_from_another_directory(self):
        docs = self.run_cli("sample")["documents"]
        self.assertEqual(len(docs), 12)
        self.assertEqual(len(self.run_cli("documents")["documents"]), 12)
        found = self.run_cli("search", "expense reimbursement", "--top-k", "2")
        self.assertEqual(found["results"][0]["document_id"], "sample-expenses")
        response = self.run_cli("ask", "expense reimbursement")
        self.assertFalse(response["abstained"])
        self.assertTrue(response["citations"])
        self.assertTrue(self.run_cli("delete", "sample-expenses")["deleted"])
        self.assertEqual(len(self.run_cli("documents")["documents"]), 11)
        self.assertIn("Document not found", self.run_cli("show", "sample-expenses", code=2)["error"])

    def test_ingest_file_and_invalid_query(self):
        file = self.root / "notes.md"
        file.write_text("The launch codeword is marigold.", encoding="utf-8")
        row = self.run_cli("ingest", str(file))["documents"][0]
        self.assertIn("marigold", self.run_cli("show", row["id"])["text"])
        self.run_cli("search", "!!!", code=2)

    def test_evaluation_is_a_measured_report(self):
        output = self.root / "report.json"
        report = self.run_cli("evaluate", "--output", str(output))
        self.assertEqual(report["queries"], 24)
        self.assertEqual(report["documents"], 12)
        self.assertEqual(json.loads(output.read_text()), report)
        for key in ("recall_at_k", "mrr_at_k", "ndcg_at_k"):
            self.assertGreaterEqual(report[key], 0)
            self.assertLessEqual(report[key], 1)
        self.assertEqual(len(report["per_query"]), 24)


if __name__ == "__main__":
    unittest.main()
