"""Deterministic lexical retrieval and offset-preserving text chunking."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import re


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def chunk_text(text: str, document_id: str, size: int = 900,
               overlap: int = 120) -> list[dict]:
    """Keep exact Python character offsets; never normalize source whitespace."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be nonblank")
    if type(size) is not int or size < 32:
        raise ValueError("chunk size must be an integer of at least 32")
    if type(overlap) is not int or not 0 <= overlap < size // 2:
        raise ValueError("chunk overlap must be nonnegative and less than half the size")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundaries = list(re.finditer(r"\s+", text[start + size // 2:end]))
            if boundaries:
                end = start + size // 2 + boundaries[-1].end()
        excerpt = text[start:end]
        if excerpt.strip():
            identity = f"{document_id}\0{start}\0{end}\0{excerpt}"
            chunks.append({"id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                           "document_id": document_id, "start": start,
                           "end": end, "text": excerpt})
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def bm25(chunks: list[dict], query: str, k1: float = 1.5,
         b: float = 0.75) -> list[tuple[str, float]]:
    """Score text and title terms with BM25; title occurrences have weight two."""
    if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
        raise ValueError("invalid BM25 parameters")
    rows = [(row["id"], Counter(tokenize(row["text"]))
             + Counter(tokenize(row.get("title", "")) * 2)) for row in chunks]
    if not rows:
        return []
    frequencies = Counter(term for _, terms in rows for term in terms)
    average_length = sum(sum(terms.values()) for _, terms in rows) / len(rows)
    if not average_length:
        return []
    query_terms = sorted(set(tokenize(query)))
    scores = []
    for chunk_id, terms in rows:
        length = sum(terms.values())
        score = 0.0
        for term in query_terms:
            frequency = terms[term]
            if frequency:
                inverse_frequency = math.log(1 + (len(rows) - frequencies[term] + 0.5)
                                             / (frequencies[term] + 0.5))
                denominator = frequency + k1 * (1 - b + b * length / average_length)
                score += inverse_frequency * frequency * (k1 + 1) / denominator
        if score > 0:
            scores.append((chunk_id, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def reciprocal_rank_fusion(rankings: list[list[tuple[str, float]]],
                           constant: int = 60) -> list[tuple[str, float]]:
    if type(constant) is not int or constant < 1:
        raise ValueError("fusion constant must be a positive integer")
    scores = Counter()
    for ranking in rankings:
        seen = set()
        for rank, (chunk_id, _) in enumerate(ranking, 1):
            if chunk_id in seen:
                raise ValueError("a ranking cannot repeat a chunk")
            seen.add(chunk_id)
            scores[chunk_id] += 1 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
