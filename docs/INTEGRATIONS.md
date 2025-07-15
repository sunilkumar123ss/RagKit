# Integration examples

## Python and LangChain

```python
from ragkit.engine import RagKit
from ragkit.answers import answer
from ragkit.langchain_adapter import RagKitRetriever

with RagKit(".ragkit") as engine:
    engine.ingest_text(
        "Approved expenses are reimbursed within ten business days.",
        title="Finance guide", source="Synthetic example", document_id="finance"
    )
    response = answer(engine, "When are expenses reimbursed?")
    print(response["answer"])
    retriever = RagKitRetriever(engine=engine, top_k=3, mode="lexical")
    documents = retriever.invoke("expense reimbursement")
    print(documents[0].metadata["citation"])
```

Install the `langchain` extra for the adapter. `ainvoke` uses LangChain's asynchronous retriever interface. The caller owns the engine lifecycle. To use dense/hybrid retrieval, install the `dense` extra, build the index, and select the appropriate `mode`; no adapter silently creates or substitutes an index.

## MCP client configuration

Install the `mcp` extra and ingest a corpus first. Substitute the absolute paths below for your checkout and existing data directory:

```json
{
  "mcpServers": {
    "ragkit": {
      "command": "/absolute/path/to/RagKit/.venv/bin/ragkit",
      "args": ["--data-dir", "/absolute/path/to/library", "mcp"]
    }
  }
}
```

The `search` and `ask` tools accept query, top_k, mode, rerank and optional document_ids. `ask` also accepts provider, defaulting to offline extractive output. Selecting a configured external provider shares the question and bounded evidence excerpts with that service. The tools expose no arbitrary paths, document ingestion, deletion, or shell execution.

The server is tested with MCP Python SDK 1.30.0 and intentionally pins `mcp<2` to keep Python 3.10 compatibility. Its actual stdio initialization, tool discovery, search, answer and shutdown are exercised in the test suite. MCP protocol output stays on stdout; diagnostics belong on stderr.

## API request examples

```json
{
  "query": "When are expenses reimbursed?",
  "top_k": 3,
  "mode": "hybrid",
  "rerank": true,
  "document_ids": ["finance"],
  "provider": "extractive"
}
```

Send this JSON to `/api/answer`; omit provider for `/api/search`. `GET /api/status` returns document/chunk counts, revision, dense-index readiness, embedding model and safe provider configuration flags. Provider configuration flags do not assert network connectivity or that a configured model exists remotely.

The source inspector uses offsets into the stored Python Unicode string. It converts those positions appropriately when presenting text in the browser; metadata offsets are not UTF-8 byte offsets. PDFs are represented by their extracted text.
