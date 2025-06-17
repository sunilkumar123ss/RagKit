"""Reproducible document-level retrieval metrics on a tiny labeled synthetic set."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
from statistics import mean, median
from tempfile import TemporaryDirectory
from time import perf_counter


def ranking_metrics(ranked_ids, relevant_ids, top_k):
    """Deduplicate chunk hits into document ranks before scoring."""
    ranked = list(dict.fromkeys(ranked_ids))[:top_k]
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Each evaluation query needs at least one relevant document")
    hits = [index for index, value in enumerate(ranked, 1) if value in relevant]
    recall = len(hits) / len(relevant)
    reciprocal_rank = 1 / hits[0] if hits else 0.0
    dcg = sum(1 / math.log2(position + 1) for position in hits)
    ideal = sum(1 / math.log2(position + 1) for position in range(1, min(top_k, len(relevant)) + 1))
    return {"recall": recall, "reciprocal_rank": reciprocal_rank, "ndcg": dcg / ideal}


def evaluate(mode="lexical", rerank=False, top_k=3):
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("top_k must be an integer from 1 to 20")
    from .engine import RagKit
    folder = Path(__file__).with_name("data")
    corpus = json.loads((folder / "sample.json").read_text(encoding="utf-8"))
    dataset = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    rows = []
    with TemporaryDirectory(prefix="ragkit-evaluation-") as directory:
        engine = RagKit(directory)
        try:
            for doc in corpus:
                engine.ingest_text(doc["text"], doc["title"], doc["source"], doc["id"])
            if mode != "lexical":
                engine.build_dense_index()
            for case in dataset["queries"]:
                start = perf_counter()
                result = engine.search(case["query"], top_k=top_k, mode=mode, rerank=rerank)
                latency = (perf_counter() - start) * 1000
                ranked = list(dict.fromkeys(item["document_id"] for item in result["results"]))
                rows.append({"query": case["query"], "relevant": case["relevant"], "ranked": ranked,
                             **ranking_metrics(ranked, case["relevant"], top_k), "latency_ms": round(latency, 3)})
        finally:
            engine.close()
    return {"dataset": "ragkit-synthetic-handbook-v1", "disclosure": dataset["description"],
            "measured_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "mode": mode, "rerank": rerank, "top_k": top_k, "documents": len(corpus), "queries": len(rows),
            "recall_at_k": mean(row["recall"] for row in rows),
            "mrr_at_k": mean(row["reciprocal_rank"] for row in rows),
            "ndcg_at_k": mean(row["ndcg"] for row in rows),
            "median_query_latency_ms": median(row["latency_ms"] for row in rows),
            "timing_note": "Single local run; query timing excludes index construction and includes lazy reranker startup when enabled.",
            "per_query": rows}
