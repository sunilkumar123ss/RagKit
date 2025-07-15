"""Optional SDK integration tests: local MCP stdio and real LangChain invocation."""
import asyncio
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from ragkit.engine import RagKit


@unittest.skipUnless(importlib.util.find_spec('langchain_core'), 'optional LangChain SDK not installed')
class LangChainAdapterTests(unittest.TestCase):
    def setUp(self):
        tracing = patch.dict('os.environ', {'LANGSMITH_TRACING': 'false',
                                           'LANGCHAIN_TRACING': 'false',
                                           'LANGCHAIN_TRACING_V2': 'false'})
        tracing.start()
        self.addCleanup(tracing.stop)

    def test_real_base_retriever_invoke_preserves_citation_metadata_and_scope(self):
        from langchain_core.retrievers import BaseRetriever
        from ragkit.langchain_adapter import RagKitRetriever
        with tempfile.TemporaryDirectory() as directory, RagKit(directory) as engine:
            engine.ingest_text('Café retrieval evidence.', 'One', document_id='one')
            engine.ingest_text('More retrieval evidence.', 'Two', document_id='two')
            retriever = RagKitRetriever(engine=engine, top_k=2, document_ids=['one'])
            self.assertIsInstance(retriever, BaseRetriever)
            documents = retriever.invoke('retrieval')
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].metadata['document_id'], 'one')
            self.assertEqual(documents[0].page_content, documents[0].metadata['citation']['quote'])
            self.assertEqual(documents[0].id, documents[0].metadata['chunk_id'])
            self.assertNotIn('engine', retriever.model_dump())
            self.assertEqual(engine.stats()['documents'], 2)

    def test_async_invoke_empty_result_and_explicit_stale_dense_failure(self):
        from ragkit.langchain_adapter import RagKitRetriever
        with tempfile.TemporaryDirectory() as directory, RagKit(directory) as engine:
            engine.ingest_text('Local retrieval evidence.', 'One')
            retriever = RagKitRetriever(engine=engine)
            self.assertEqual(asyncio.run(retriever.ainvoke('unmatchedword')), [])
            with self.assertRaisesRegex(RuntimeError, 'stale'):
                RagKitRetriever(engine=engine, mode='dense').invoke('retrieval')

    def test_settings_validation(self):
        from ragkit.langchain_adapter import RagKitRetriever
        for top_k in [0, 21, True]:
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                RagKitRetriever(engine=object(), top_k=top_k)


@unittest.skipUnless(importlib.util.find_spec('mcp'), 'optional MCP SDK not installed')
class MCPIntegrationTests(unittest.TestCase):
    def test_sdk_registers_only_search_and_ask(self):
        from ragkit.mcp_server import create_server
        with tempfile.TemporaryDirectory() as directory, RagKit(directory) as engine:
            server = create_server(engine=engine)
            tools = asyncio.run(server.list_tools())
            self.assertEqual({tool.name for tool in tools}, {'search', 'ask'})
            for tool in tools:
                self.assertNotIn('path', tool.inputSchema['properties'])
                self.assertNotIn('data_dir', tool.inputSchema['properties'])
                self.assertTrue(tool.annotations.readOnlyHint)
            self.assertEqual(engine.stats()['documents'], 0)

    def test_real_stdio_session_search_ask_and_input_error(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        with tempfile.TemporaryDirectory() as directory:
            with RagKit(directory) as engine:
                engine.ingest_text('Café evidence is stored locally.', 'A note', document_id='note')
            repository = Path(__file__).resolve().parents[1]
            parameters = StdioServerParameters(
                command=sys.executable,
                args=['-c', 'import sys; from ragkit.mcp_server import serve; serve(sys.argv[1])', directory],
                cwd=str(repository), env={'PYTHONDONTWRITEBYTECODE': '1'},
            )
            async def check():
                async with stdio_client(parameters) as (read, write):
                    async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=20)) as session:
                        initialized = await session.initialize()
                        self.assertEqual(initialized.serverInfo.name, 'RagKit')
                        tools = await session.list_tools()
                        self.assertEqual({t.name for t in tools.tools}, {'search', 'ask'})
                        found = await session.call_tool('search', {'query': 'Café', 'document_ids': ['note']})
                        self.assertFalse(found.isError)
                        result = found.structuredContent or json.loads(found.content[0].text)
                        self.assertEqual(result['results'][0]['document_id'], 'note')
                        answered = await session.call_tool('ask', {'query': 'Café'})
                        self.assertFalse(answered.isError)
                        answer = answered.structuredContent or json.loads(answered.content[0].text)
                        self.assertEqual(answer['provider'], 'extractive')
                        self.assertIn('[S1]', answer['answer'])
                        missing = await session.call_tool('ask', {'query': 'unmatchedword', 'provider': 'openai'})
                        self.assertFalse(missing.isError)
                        empty = missing.structuredContent or json.loads(missing.content[0].text)
                        self.assertTrue(empty['abstained'])
                        invalid = await session.call_tool('search', {'query': 'Café', 'top_k': 0})
                        self.assertTrue(invalid.isError)
            asyncio.run(check())
            # The owned engine was closed when the stdio server exited.
            with RagKit(directory) as reopened:
                self.assertEqual(reopened.stats()['documents'], 1)


if __name__ == '__main__':
    unittest.main()
