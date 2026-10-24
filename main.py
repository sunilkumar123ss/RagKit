#!/usr/bin/env python3
"""Offline BM25 retrieval over body and title indexes, rank fusion and exact citations."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re


def tokenize(text):
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("invalid BM25 parameters")
        self.k1, self.b = k1, b
        self.rows = [(key, Counter(tokenize(text))) for key, text in documents]
        self.df = Counter(term for _, terms in self.rows for term in terms)
        self.average = sum(sum(terms.values()) for _, terms in self.rows) / max(1, len(self.rows))

    def rank(self, query):
        output = []
        for key, terms in self.rows:
            length = sum(terms.values())
            score = 0.0
            for term in sorted(set(tokenize(query))):
                frequency = terms[term]
                if not frequency: continue
                df = self.df[term]
                idf = math.log(1 + (len(self.rows) - df + 0.5) / (df + 0.5))
                norm = self.k1 * (1 - self.b + self.b * length / self.average)
                score += idf * frequency * (self.k1 + 1) / (frequency + norm)
            if score > 0: output.append((key, score))
        return sorted(output, key=lambda pair: (-pair[1], pair[0]))


def fuse(rankings, constant=60):
    if type(constant) is not int or constant < 1: raise ValueError("constant must be positive")
    scores = Counter()
    for ranking in rankings:
        seen = set()
        for rank, key in enumerate(ranking, 1):
            if key in seen: raise ValueError("a ranking cannot repeat a document")
            seen.add(key)
            scores[key] += 1 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def citation(document, query, field):
    text = document[field]
    terms = set(tokenize(query))
    spans = list(re.finditer(r"[^.!?\n]+[.!?]?", text))
    if not spans: return {"document_id": document["id"], "field": field, "start": 0, "end": len(text), "quote": text}
    best = max(spans, key=lambda match: (len(terms & set(tokenize(match.group()))), -match.start()))
    start, end = best.span()
    while start < end and text[start].isspace(): start += 1
    return {"document_id": document["id"], "field": field, "start": start, "end": end, "quote": text[start:end]}


class Retriever:
    def __init__(self, documents):
        if not isinstance(documents, list): raise ValueError("documents must be a list")
        self.documents = {}
        for document in documents:
            if not isinstance(document, dict) or any(not isinstance(document.get(key), str) or not document[key].strip()
                                                    for key in ("id", "title", "text")):
                raise ValueError("documents require non-empty id, title and text")
            if document["id"] in self.documents: raise ValueError("duplicate document id")
            self.documents[document["id"]] = document.copy()
        self.body = BM25([(key, doc["text"]) for key, doc in self.documents.items()])
        self.title = BM25([(key, doc["title"]) for key, doc in self.documents.items()])

    def search(self, query, top_k=3):
        if not isinstance(query, str) or not tokenize(query): raise ValueError("query needs searchable words")
        if type(top_k) is not int or not 1 <= top_k <= 10: raise ValueError("top_k must be 1..10")
        body, title = self.body.rank(query), self.title.rank(query)
        fused = fuse([[key for key, _ in body], [key for key, _ in title]])
        body_scores, title_scores = dict(body), dict(title)
        results = []
        for key, score in fused[:top_k]:
            field = "text" if key in body_scores else "title"
            results.append({"id": key, "title": self.documents[key]["title"], "fusion_score": score,
                            "body_score": body_scores.get(key, 0), "title_score": title_scores.get(key, 0),
                            "citation": citation(self.documents[key], query, field)})
        return {"query": query, "abstained": not bool(results), "results": results,
                "method": "lexical BM25 body + title indexes with reciprocal rank fusion"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="retrieval citations")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("example.json"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    try: print(json.dumps(Retriever(json.loads(args.input.read_text())).search(args.query, args.top_k), indent=2))
    except (ValueError, OSError, TypeError) as exc: parser.error(str(exc))


if __name__ == "__main__": main()
