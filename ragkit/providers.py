"""Optional model transports; credentials are read only when a provider is used."""
from __future__ import annotations

import importlib.util
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

REQUEST_TIMEOUT_SECONDS = 30.0
MAX_OUTPUT_TOKENS = 2048
MAX_RESPONSE_BYTES = 262_144
MODEL_ENVS = {
    'openai': 'RAGKIT_OPENAI_MODEL',
    'anthropic': 'RAGKIT_ANTHROPIC_MODEL',
    'ollama': 'RAGKIT_OLLAMA_MODEL',
}
_MODEL_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:/-]{0,119}\Z')


def _model(provider: str, *, required: bool = True) -> str | None:
    name = os.environ.get(MODEL_ENVS[provider], '').strip()
    valid = bool(_MODEL_PATTERN.fullmatch(name)) and not name.lower().startswith(('sk-', 'sk_', 'bearer'))
    if not valid:
        if required:
            raise RuntimeError(f'Set {MODEL_ENVS[provider]} to a valid model name before using {provider}.')
        return None
    return name


def _key(variable: str) -> str:
    value = os.environ.get(variable, '').strip()
    if not value:
        raise RuntimeError(f'Set {variable} on the server before selecting this provider.')
    return value


def _ollama_url() -> str:
    host = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434').strip()
    if '://' not in host:
        host = 'http://' + host
    try:
        parsed = urlsplit(host)
        valid = (parsed.scheme in ('http', 'https') and parsed.hostname and
                 not parsed.username and not parsed.password and not parsed.query and
                 not parsed.fragment and parsed.path in ('', '/') and
                 parsed.port != 0)
    except ValueError:
        valid = False
    if not valid:
        raise RuntimeError('OLLAMA_HOST must be an HTTP(S) host URL without credentials, path, query, or fragment.')
    return host.rstrip('/') + '/api/chat'


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def provider_status() -> dict:
    """Report dependency/configuration state without contacting services or exposing keys.

    ``available`` means the local client dependency is present, not that a model
    endpoint is reachable. ``configured`` does not validate the credentials.
    """
    status = {'extractive': {'available': True, 'configured': True, 'model': None}}
    for name, key in [('openai', 'OPENAI_API_KEY'), ('anthropic', 'ANTHROPIC_API_KEY')]:
        model = _model(name, required=False)
        status[name] = {'available': _installed(name),
                        'configured': bool(model and os.environ.get(key, '').strip()), 'model': model}
    model = _model('ollama', required=False)
    try:
        _ollama_url()
        valid_host = True
    except RuntimeError:
        valid_host = False
    status['ollama'] = {'available': True, 'configured': bool(model and valid_host), 'model': model}
    return status


def _failure(provider: str, exc: Exception) -> RuntimeError:
    # Provider exceptions may embed keys, request bodies, or endpoint URLs.
    # Expose only a small, fixed classification rather than str(exc).
    name = type(exc).__name__.lower()
    if 'timeout' in name:
        detail = 'the request timed out'
    elif 'authentication' in name or getattr(exc, 'status_code', None) in (401, 403):
        detail = 'authentication or access was rejected'
    elif 'ratelimit' in name or getattr(exc, 'status_code', None) == 429:
        detail = 'the service rate limit was reached'
    else:
        detail = 'the service request failed'
    return RuntimeError(f'{provider} generation failed: {detail}. Check the configured model, credentials and service availability.')


def _openai(system: str, user: str) -> str:
    model, key = _model('openai'), _key('OPENAI_API_KEY')
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('OpenAI generation requires the optional openai package; install RagKit with its providers extra.') from exc
    try:
        with OpenAI(api_key=key, base_url='https://api.openai.com/v1',
                    timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0) as client:
            response = client.responses.create(
                model=model, instructions=system, input=user, store=False,
                max_output_tokens=MAX_OUTPUT_TOKENS, text={'format': {'type': 'json_object'}},
            )
    except Exception as exc:
        raise _failure('OpenAI', exc) from None
    if response.status != 'completed':
        raise RuntimeError('OpenAI returned an incomplete response; no answer was accepted.')
    try:
        return response.output_text
    except (AttributeError, TypeError):
        raise RuntimeError('OpenAI returned invalid text content; no answer was accepted.') from None


def _anthropic(system: str, user: str) -> str:
    model, key = _model('anthropic'), _key('ANTHROPIC_API_KEY')
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError('Anthropic generation requires the optional anthropic package; install RagKit with its providers extra.') from exc
    try:
        with Anthropic(api_key=key, base_url='https://api.anthropic.com',
                       timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0) as client:
            response = client.messages.create(
                model=model, max_tokens=MAX_OUTPUT_TOKENS, system=system,
                messages=[{'role': 'user', 'content': user}],
            )
    except Exception as exc:
        raise _failure('Anthropic', exc) from None
    if response.stop_reason != 'end_turn':
        raise RuntimeError('Anthropic did not complete an answer normally; no answer was accepted.')
    try:
        return ''.join(block.text for block in response.content if block.type == 'text')
    except (AttributeError, TypeError):
        raise RuntimeError('Anthropic returned invalid text content; no answer was accepted.') from None


def _ollama(system: str, user: str) -> str:
    model = _model('ollama')
    payload = {'model': model, 'messages': [{'role': 'system', 'content': system},
                                           {'role': 'user', 'content': user}],
               'stream': False, 'format': 'json', 'options': {'num_predict': MAX_OUTPUT_TOKENS}}
    request = Request(_ollama_url(), data=json.dumps(payload).encode('utf-8'),
                      headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise _failure('Ollama', exc) from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError('Ollama returned an oversized response; no answer was accepted.')
    try:
        result = json.loads(raw)
        if not isinstance(result, dict) or result.get('done') is not True or result.get('done_reason') not in (None, 'stop'):
            raise ValueError('incomplete response')
        content = result['message']['content']
        if not isinstance(content, str):
            raise ValueError('non-text content')
        return content
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError('Ollama returned an invalid or incomplete chat response; no answer was accepted.') from None


def generate(provider: str, system: str, user: str) -> str:
    """Make one non-streaming request without automatic SDK retries."""
    functions = {'openai': _openai, 'anthropic': _anthropic, 'ollama': _ollama}
    if provider not in functions:
        raise ValueError('Unknown model provider.')
    text = functions[provider](system, user)
    if not isinstance(text, str) or not text.strip() or len(text.encode('utf-8')) > MAX_RESPONSE_BYTES:
        raise RuntimeError('The model returned empty or oversized answer content.')
    return text
