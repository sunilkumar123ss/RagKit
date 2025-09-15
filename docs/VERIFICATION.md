# Verification evidence

Verified locally on 24 October 2026 with Python 3.12.14 on macOS arm64. This is an engineering test record, not a production performance claim.

The final non-neural suite passed **108 tests and 49 subtests**; the separate actual embedding/reranker test passed. An independent install check passed 24 checks against a wheel installed in a fresh environment outside the checkout, including packaged assets, persistent CLI operations and installed-wheel API requests. Optional web dependencies were reused from the verified development environment for that API check.

## Coverage

- Original eight retrieval-reference tests retained.
- Persistent SQLite ingestion, replacement/deletion, exact Unicode chunk offsets, deterministic BM25/fusion, scope filters, stale-index handling and invalid-vector checks.
- Actual Qdrant 1.20.0 local persistence/reopen/filtering with deterministic test embeddings.
- Actual sentence-transformers/all-MiniLM-L6-v2 and cross-encoder/ms-marco-MiniLM-L6-v2 execution, hybrid/reranked relevance, citation round-trips, persistence and stale-index rejection.
- FastAPI validation, multipart upload, text/Markdown/JSON/PDF extraction, source retrieval, persistence, Host/Origin boundaries and bounded requests.
- Actual MCP 1.30.0 stdio client/server initialization, tool discovery, search/answer and shutdown; actual LangChain 1.7.0 invoke/ainvoke.
- OpenAI 2.54.0 and Anthropic 0.125.0 SDK serialization used mock HTTP transports. Ollama used a local test HTTP server. External answer services were not called; credentials/model availability must be supplied by the operator.
- Chrome desktop/mobile tests cover sample/pasted document ingestion, scoped questions, source inspection, Unicode highlighting, literal HTML-shaped source text, deletion, errors and responsive layouts at 1280, 375 and 320 pixels. No horizontal overflow or final console errors. File-picker automation was unavailable because of extension permissions; multipart API upload tests passed.

Run `python verify.py` in the documented development environment for application tests, installed CLI help and the preserved reference demo. The real neural-model test is opt-in: `RAGKIT_RUN_LIVE_MODELS=1 python -m pytest -q -m live_models`. First execution downloads model weights; it does not send the corpus to an inference service.

## Measured synthetic retrieval results

The same 12 synthetic documents and 24 labeled questions are used in both runs, with k=3. Multiple chunks are deduplicated to document IDs before computing metrics. This set was authored for demonstrations and is too small to establish general performance.

| Mode | Recall@3 | MRR@3 | nDCG@3 | Median query time |
| --- | ---: | ---: | ---: | ---: |
| BM25 text + weighted title | 1.0000 | 0.9583 | 0.9692 | 0.153 ms |
| Hybrid + real cross-encoder | 1.0000 | 0.9792 | 0.9846 | 18.728 ms |

Raw reports: [lexical](benchmarks/lexical.json), [hybrid + reranking](benchmarks/hybrid-reranked.json). Each includes every query, ranking, timing and actual measurement timestamp. Timing excludes index construction and includes lazy reranker startup; it is one local run, not a controlled throughput benchmark.

## Reproducibility and limits

`requirements-tested.txt` records the local tested dependency versions; it is an environment snapshot, not a portable lock file. `pyproject.toml` defines supported dependency ranges and optional extras. GitHub Actions tests the non-neural suite across Python 3.10, 3.12 and 3.13; consult the workflow results for hosted outcomes rather than assuming a local run proves them.

The Docker/Compose files were statically checked; container execution was unavailable because the local Docker-compatible Podman daemon was not running. Native installation and application execution were verified independently. This application uses one process per embedded Qdrant directory; it is not a multi-tenant hosted service. Citation existence checks do not prove that generated claims follow from the evidence.
