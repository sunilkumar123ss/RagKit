"""Real SDK request serialization against stub HTTP transports; no paid requests."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import threading
import unittest
from unittest.mock import patch

from ragkit import providers

ANSWER = json.dumps({'answer': 'Evidence [S1].', 'citations': ['S1'], 'abstained': False})


@contextmanager
def ollama_stub(payload, status=200):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


class ProviderConfigurationTests(unittest.TestCase):
    def test_status_is_local_and_never_returns_keys_or_host_credentials(self):
        env = {'OPENAI_API_KEY': 'test-secret-not-a-real-key', 'RAGKIT_OPENAI_MODEL': 'configured-model',
               'ANTHROPIC_API_KEY': 'another-test-secret', 'RAGKIT_ANTHROPIC_MODEL': 'sk-secret-wrong-field',
               'RAGKIT_OLLAMA_MODEL': 'local-model:small', 'OLLAMA_HOST': 'http://user:password@localhost:11434'}
        with patch.dict('os.environ', env, clear=True), patch('ragkit.providers.urlopen') as request:
            status = providers.provider_status()
        self.assertTrue(status['openai']['configured'])
        self.assertFalse(status['anthropic']['configured'])
        self.assertIsNone(status['anthropic']['model'])
        self.assertFalse(status['ollama']['configured'])
        serialized = json.dumps(status)
        for secret in ['test-secret', 'another-test-secret', 'sk-secret', 'password']:
            self.assertNotIn(secret, serialized)
        request.assert_not_called()

    def test_unconfigured_models_and_keys_fail_clearly(self):
        with patch.dict('os.environ', {}, clear=True):
            for name in ['openai', 'anthropic', 'ollama']:
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'RAGKIT_'):
                    providers.generate(name, 'system', 'question')
        with patch.dict('os.environ', {'RAGKIT_OPENAI_MODEL': 'configured-model'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'OPENAI_API_KEY'):
                providers.generate('openai', 'system', 'question')

    def test_host_validation(self):
        for host in ['file:///tmp/data', 'https://user:pass@example.com', 'http://localhost:abc',
                     'http://localhost:0', 'http://localhost/path', 'http://localhost?key=secret']:
            with self.subTest(host=host), patch.dict('os.environ', {'OLLAMA_HOST': host}):
                with self.assertRaises(RuntimeError):
                    providers._ollama_url()
        with patch.dict('os.environ', {'OLLAMA_HOST': 'localhost:11434'}):
            self.assertEqual(providers._ollama_url(), 'http://localhost:11434/api/chat')

    def test_ollama_posts_bounded_nonstreaming_chat_to_local_stub(self):
        with ollama_stub({'done': True, 'done_reason': 'stop', 'message': {'content': ANSWER}}) as (host, calls):
            with patch.dict('os.environ', {'RAGKIT_OLLAMA_MODEL': 'configured-local', 'OLLAMA_HOST': host}, clear=True):
                with patch('ragkit.providers.urlopen', wraps=providers.urlopen) as request:
                    self.assertEqual(providers.generate('ollama', 'system', 'question'), ANSWER)
                self.assertEqual(request.call_args.kwargs['timeout'], 30.0)
        self.assertEqual(calls[0][0], '/api/chat')
        payload = calls[0][1]
        self.assertEqual(payload['model'], 'configured-local')
        self.assertFalse(payload['stream'])
        self.assertEqual(payload['format'], 'json')
        self.assertEqual(payload['options']['num_predict'], providers.MAX_OUTPUT_TOKENS)
        self.assertEqual(payload['messages'][1], {'role': 'user', 'content': 'question'})

    def test_ollama_rejects_incomplete_invalid_and_oversized_responses(self):
        for payload in [{'done': False, 'message': {'content': ANSWER}},
                        {'done': True, 'done_reason': 'length', 'message': {'content': ANSWER}},
                        b'not json', b'x' * (providers.MAX_RESPONSE_BYTES + 1)]:
            with self.subTest(payload=str(payload)[:40]), ollama_stub(payload) as (host, _):
                with patch.dict('os.environ', {'RAGKIT_OLLAMA_MODEL': 'local-model', 'OLLAMA_HOST': host}, clear=True):
                    with self.assertRaises(RuntimeError):
                        providers.generate('ollama', 'system', 'question')


@unittest.skipUnless(importlib.util.find_spec('openai') and importlib.util.find_spec('httpx'), 'optional OpenAI SDK not installed')
class OpenAIContractTests(unittest.TestCase):
    def test_responses_api_request_and_text_extraction_with_real_sdk(self):
        import httpx
        import openai
        original = openai.OpenAI
        seen = []
        kwargs_seen = []
        def handle(request):
            seen.append((str(request.url), json.loads(request.content)))
            return httpx.Response(200, json={'id': 'resp_test', 'object': 'response', 'created_at': 0,
                'status': 'completed', 'model': 'configured-openai', 'output': [{'id': 'msg_test',
                'type': 'message', 'role': 'assistant', 'status': 'completed',
                'content': [{'type': 'output_text', 'text': ANSWER, 'annotations': []}]}]})
        def client(**kwargs):
            kwargs_seen.append(kwargs)
            return original(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key', 'RAGKIT_OPENAI_MODEL': 'configured-openai'}, clear=True), patch('openai.OpenAI', side_effect=client):
            self.assertEqual(providers.generate('openai', 'system JSON instruction', 'question'), ANSWER)
        url, payload = seen[0]
        self.assertEqual(url, 'https://api.openai.com/v1/responses')
        self.assertEqual(payload['model'], 'configured-openai')
        self.assertFalse(payload['store'])
        self.assertEqual(payload['max_output_tokens'], providers.MAX_OUTPUT_TOKENS)
        self.assertEqual(payload['text'], {'format': {'type': 'json_object'}})
        self.assertEqual(kwargs_seen[0]['timeout'], 30.0)
        self.assertEqual(kwargs_seen[0]['max_retries'], 0)

    def test_timeout_is_not_retried_and_error_does_not_leak_content(self):
        import httpx
        import openai
        original = openai.OpenAI
        calls = []
        def handle(request):
            calls.append(request)
            raise httpx.ReadTimeout('secret-data-from-provider', request=request)
        def client(**kwargs):
            return original(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key', 'RAGKIT_OPENAI_MODEL': 'configured-openai'}, clear=True), patch('openai.OpenAI', side_effect=client):
            with self.assertRaisesRegex(RuntimeError, 'timed out') as raised:
                providers.generate('openai', 'system', 'question')
        self.assertEqual(len(calls), 1)
        self.assertNotIn('secret-data', str(raised.exception))


@unittest.skipUnless(importlib.util.find_spec('anthropic') and importlib.util.find_spec('httpx'), 'optional Anthropic SDK not installed')
class AnthropicContractTests(unittest.TestCase):
    def test_messages_request_and_text_extraction_with_real_sdk(self):
        import anthropic
        import httpx
        original = anthropic.Anthropic
        seen = []
        kwargs_seen = []
        def handle(request):
            seen.append((str(request.url), json.loads(request.content)))
            return httpx.Response(200, json={'id': 'msg_test', 'type': 'message', 'role': 'assistant',
                'model': 'configured-claude', 'content': [{'type': 'text', 'text': ANSWER}],
                'stop_reason': 'end_turn', 'stop_sequence': None,
                'usage': {'input_tokens': 10, 'output_tokens': 10}})
        def client(**kwargs):
            kwargs_seen.append(kwargs)
            return original(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key', 'RAGKIT_ANTHROPIC_MODEL': 'configured-claude'}, clear=True), patch('anthropic.Anthropic', side_effect=client):
            self.assertEqual(providers.generate('anthropic', 'system', 'question'), ANSWER)
        self.assertEqual(seen[0][0], 'https://api.anthropic.com/v1/messages')
        self.assertEqual(seen[0][1]['model'], 'configured-claude')
        self.assertEqual(seen[0][1]['max_tokens'], providers.MAX_OUTPUT_TOKENS)
        self.assertEqual(kwargs_seen[0]['timeout'], 30.0)
        self.assertEqual(kwargs_seen[0]['max_retries'], 0)


if __name__ == '__main__':
    unittest.main()
