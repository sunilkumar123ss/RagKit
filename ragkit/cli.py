"""Command line for the same persistent engine used by the web app and MCP tools."""
import argparse
import json
import os
from pathlib import Path
import sys


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def parser():
    app = argparse.ArgumentParser(prog="ragkit", description="Index local documents and ask questions with inspectable sources.")
    app.add_argument("--data-dir", type=Path, default=Path(os.environ.get("RAGKIT_DATA_DIR", ".ragkit")))
    commands = app.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Import TXT, Markdown, JSON, or text PDF files")
    ingest.add_argument("files", nargs="+", type=Path)
    commands.add_parser("sample", help="Load the synthetic demo handbook")
    commands.add_parser("documents", help="List the persistent document library")
    show = commands.add_parser("show", help="Show a document and its original text")
    show.add_argument("document_id")
    delete = commands.add_parser("delete", help="Delete a document and invalidate its semantic index")
    delete.add_argument("document_id")
    commands.add_parser("status", help="Show corpus and semantic index status")
    commands.add_parser("index", help="Build the optional Sentence-Transformers / Qdrant semantic index")
    for name in ("search", "ask"):
        command = commands.add_parser(name, help="Retrieve sources" if name == "search" else "Answer with source references")
        command.add_argument("query")
        command.add_argument("--top-k", type=int, default=5)
        command.add_argument("--mode", choices=("lexical", "dense", "hybrid"), default="lexical")
        command.add_argument("--rerank", action="store_true")
        command.add_argument("--document-id", action="append", dest="document_ids")
        if name == "ask":
            command.add_argument("--provider", choices=("extractive", "ollama", "openai", "anthropic"), default="extractive")
    serve = commands.add_parser("serve", help="Start the local web workspace and API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    commands.add_parser("mcp", help="Run the optional MCP tool server over stdio")
    evaluate = commands.add_parser("evaluate", help="Measure retrieval on the bundled labeled synthetic corpus")
    evaluate.add_argument("--mode", choices=("lexical", "dense", "hybrid"), default="lexical")
    evaluate.add_argument("--rerank", action="store_true")
    evaluate.add_argument("--top-k", type=int, default=3)
    evaluate.add_argument("--output", type=Path)
    return app


def main(argv=None):
    args = parser().parse_args(argv)
    engine = None
    try:
        if args.command == "serve":
            try:
                import uvicorn
                from .api import create_app
            except ImportError as exc:
                raise RuntimeError("Install the web extra: python -m pip install -e '.[web]'") from exc
            uvicorn.run(create_app(args.data_dir), host=args.host, port=args.port)
            return 0
        if args.command == "mcp":
            from .mcp_server import serve
            serve(args.data_dir)
            return 0
        if args.command == "evaluate":
            from .evaluation import evaluate
            result = evaluate(args.mode, args.rerank, args.top_k)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            emit(result)
            return 0
        from .engine import RagKit
        engine = RagKit(args.data_dir)
        if args.command == "ingest":
            from .loaders import MAX_UPLOAD_BYTES, load_documents
            loaded = []
            for file in args.files:
                if file.stat().st_size > MAX_UPLOAD_BYTES:
                    raise ValueError("Upload size exceeds the 8 MB limit")
                for row in load_documents(file.name, file.read_bytes()):
                    loaded.append(engine.ingest_text(row["text"], row["title"], row.get("source", file.name), row.get("id")))
            emit({"documents": loaded})
        elif args.command == "sample":
            rows = json.loads(Path(__file__).with_name("data").joinpath("sample.json").read_text(encoding="utf-8"))
            emit({"documents": [engine.ingest_text(row["text"], row["title"], row["source"], row["id"]) for row in rows]})
        elif args.command == "documents":
            emit({"documents": engine.list_documents()})
        elif args.command == "show":
            emit(engine.get_document(args.document_id))
        elif args.command == "delete":
            deleted = engine.delete_document(args.document_id)
            if not deleted:
                raise KeyError(args.document_id)
            emit({"deleted": True, "document_id": args.document_id})
        elif args.command == "status":
            emit(engine.stats())
        elif args.command == "index":
            emit(engine.build_dense_index())
        elif args.command in ("search", "ask"):
            options = {"top_k": args.top_k, "mode": args.mode, "rerank": args.rerank, "document_ids": args.document_ids}
            if args.command == "search":
                emit(engine.search(args.query, **options))
            else:
                from .answers import answer
                emit(answer(engine, args.query, provider=args.provider, **options))
        return 0
    except KeyError as exc:
        print(json.dumps({"error": f"Document not found: {exc.args[0]}"}), file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
