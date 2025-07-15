"""Read/search and answer tools over stdio using the optional MCP Python SDK 1.x.

Install RagKit's mcp extra (mcp<2). The server owns one engine per lifespan;
its tools never accept filesystem paths or expose arbitrary file operations.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any, Literal

from .answers import answer


def create_server(data_dir: str | Path | None = None, *, engine: Any = None):
    """Build a FastMCP 1.x server; a supplied engine remains caller-owned."""
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.server.fastmcp.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        raise RuntimeError('MCP integration requires the optional MCP SDK 1.x; install RagKit with its mcp extra (mcp<2).') from exc

    selected_dir = data_dir if data_dir is not None else os.environ.get('RAGKIT_DATA_DIR', '.ragkit')
    active = {'engine': engine}

    @asynccontextmanager
    async def lifespan(_server):
        owned = None
        if engine is None:
            from .engine import RagKit
            owned = await asyncio.to_thread(RagKit, selected_dir)
            active['engine'] = owned
        try:
            yield {'engine': active['engine']}
        finally:
            if owned is not None:
                await asyncio.to_thread(owned.close)
                active['engine'] = None

    server = FastMCP('RagKit', lifespan=lifespan,
                     instructions='Search the configured local RagKit corpus and return inspectable source citations. External answer providers receive the question and bounded source quotations only when explicitly selected.')

    def current_engine():
        if active['engine'] is None:
            raise RuntimeError('The RagKit MCP server lifespan has not started.')
        return active['engine']

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
    async def search(query: str, top_k: int = 5, mode: Literal['lexical', 'dense', 'hybrid'] = 'lexical',
                     rerank: bool = False, document_ids: list[str] | None = None) -> dict[str, Any]:
        """Search the configured corpus and return exact source spans. Dense retrieval requires a current index."""
        try:
            return await asyncio.to_thread(current_engine().search, query, top_k=top_k,
                                           mode=mode, rerank=rerank, document_ids=document_ids)
        except (ValueError, RuntimeError) as exc:
            raise ToolError(str(exc)) from None

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True))
    async def ask(query: str, top_k: int = 5, mode: Literal['lexical', 'dense', 'hybrid'] = 'lexical',
                  rerank: bool = False, document_ids: list[str] | None = None,
                  provider: Literal['extractive', 'openai', 'anthropic', 'ollama'] = 'extractive') -> dict[str, Any]:
        """Answer from retrieved evidence. Extractive is offline; selecting a model provider shares the question and bounded quotations with that configured service."""
        try:
            return await asyncio.to_thread(answer, current_engine(), query, top_k=top_k,
                                           mode=mode, rerank=rerank, document_ids=document_ids, provider=provider)
        except (ValueError, RuntimeError) as exc:
            raise ToolError(str(exc)) from None

    return server


def serve(data_dir: str | Path | None = None) -> None:
    """Run the configured corpus as an MCP stdio server, without opening an HTTP port."""
    create_server(data_dir).run(transport='stdio')
