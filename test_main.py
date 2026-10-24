import json
from pathlib import Path
import unittest
from main import BM25, Retriever, fuse, tokenize


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = json.loads(Path(__file__).with_name("example.json").read_text())
        self.retriever = Retriever(self.docs)

    def test_relevant_document_leads(self):
        self.assertEqual(self.retriever.search("citations")["results"][0]["id"], "retrieval-guide")

    def test_unknown_query_abstains(self):
        self.assertEqual(self.retriever.search("zebracorn")["results"], [])
        self.assertTrue(self.retriever.search("zebracorn")["abstained"])

    def test_citations_round_trip_exact_offsets(self):
        by_id = {d["id"]:d for d in self.docs}
        for item in self.retriever.search("retrieval citations")["results"]:
            cite = item["citation"]
            self.assertEqual(cite["quote"], by_id[cite["document_id"]][cite["field"]][cite["start"]:cite["end"]])
            self.assertTrue(cite["quote"])

    def test_title_only_evidence_is_labeled(self):
        r = Retriever([{"id":"a","title":"Unicorn","text":"Ordinary prose."}])
        self.assertEqual(r.search("unicorn")["results"][0]["citation"]["field"], "title")

    def test_rank_fusion_consensus(self):
        self.assertEqual(fuse([["a","b"],["c","b"]])[0][0], "b")
        with self.assertRaises(ValueError): fuse([["a","a"]])

    def test_invalid_documents_queries_and_limits(self):
        with self.assertRaises(ValueError): Retriever(self.docs + [self.docs[0]])
        with self.assertRaises(ValueError): self.retriever.search("!!!")
        with self.assertRaises(ValueError): self.retriever.search("voice", True)

    def test_unicode_and_empty_index(self):
        self.assertEqual(tokenize("CAFÉ café"), ["café","café"])
        self.assertEqual(BM25([]).rank("anything"), [])

    def test_input_order_does_not_change_results(self):
        self.assertEqual(self.retriever.search("retrieval"), Retriever(list(reversed(self.docs))).search("retrieval"))


if __name__ == "__main__": unittest.main()
