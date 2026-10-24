# RagKit

**Small retrieval primitives with scores you can inspect and citations you can verify.**

RagKit indexes document bodies and titles separately with BM25, combines both rankings through reciprocal rank fusion, and returns an exact source excerpt with character offsets. All computation is local and deterministic.

## Run

Python 3.9+ · standard library only.

```sh
python3 main.py
python3 main.py "voice playback" --top-k 2
python3 main.py --input example.json "retrieval citations"
python3 main.py --help
python3 -m unittest discover -s . -p 'test_main.py' -v
```

The default fixture is resolved relative to the script. Default demo and help work from any working directory; explicit `--input` paths follow the caller.

## Retrieval path

1. Unicode-aware tokenization case-folds document text and queries.
2. Body and title BM25 indexes score matching query terms using document frequency and length normalization.
3. Reciprocal rank fusion combines their ranks with `1 / (60 + rank)`.
4. A matching source sentence is selected by query-term overlap. The result includes the document ID, source field, exact quote and `[start, end)` character offsets.

Both component scores remain visible. A title-only match cites the title and labels that field explicitly. Queries containing no searchable words are rejected; queries with no matching terms return `abstained: true` and an empty result list.

## Verify the evidence

For every result, the invariant is:

```python
document[citation["field"]][citation["start"]:citation["end"]] == citation["quote"]
```

Eight tests check relevance on the synthetic corpus, abstention, exact citation offsets, title-only evidence, fusion consensus, invalid inputs, Unicode tokenization and stable ordering.

## Scope and provenance

Created in **October 2026 with AI coding assistance**, using the project name from Sunil Kumar's CV. Original project source was not recovered. This is a new independent reference; it does not claim the original implementation, adoption, stars, benchmark results or historical development dates.

Both indexes are lexical. There are no embeddings, dense retrieval, cross-encoder reranking, LangChain, Qdrant, MCP or LLM generation adapters. A source quote proves textual provenance, not that a generated claim is supported or true. The tool returns evidence rather than generating an answer. Source documents are treated as data and are never executed.

## Repository checks

```sh
python3 verify.py
```

This runs the isolated test suite, synthetic example, and CLI help. The `verify.yml.template` file contains an optional GitHub Actions workflow. Place it at `.github/workflows/verify.yml` to enable hosted checks; the current repository is verified locally.
