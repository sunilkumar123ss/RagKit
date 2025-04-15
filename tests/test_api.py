import json

import pytest
from fastapi.testclient import TestClient

from ragkit.api import MAX_REQUEST_BYTES, create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path), base_url='http://127.0.0.1') as client:
        yield client


def add(client, title='Leave policy', text='Employees receive 24 days of annual leave. Submit requests five days in advance.'):
    response = client.post('/api/documents', json={'title': title, 'text': text})
    assert response.status_code == 201, response.text
    return response.json()


def test_status_empty_library_and_static_ui(client):
    response = client.get('/api/status')
    assert response.status_code == 200
    assert response.json()['documents'] == 0
    assert response.json()['providers']['extractive']['available'] is True
    assert response.headers['cache-control'] == 'no-store'
    assert client.get('/api/documents').json() == {'documents': []}
    page = client.get('/')
    assert 'Knowledge, with the receipts.' in page.text
    assert "script-src 'self'" in page.headers['content-security-policy']
    assert client.get('/static/app.js').status_code == 200


def test_document_idempotence_get_and_delete(client):
    first = add(client)
    repeated = add(client)
    assert repeated['id'] == first['id']
    assert repeated['unchanged'] is True
    assert client.get('/api/documents/' + first['id']).json()['text'].startswith('Employees')
    assert len(client.get('/api/documents').json()['documents']) == 1
    assert client.delete('/api/documents/' + first['id']).json()['deleted'] is True
    assert client.get('/api/documents/' + first['id']).status_code == 404
    assert client.delete('/api/documents/' + first['id']).status_code == 404


def test_database_persists_across_application_lifetimes(tmp_path):
    with TestClient(create_app(tmp_path), base_url='http://localhost') as first:
        document = add(first)
    with TestClient(create_app(tmp_path), base_url='http://localhost') as second:
        assert second.get('/api/documents/' + document['id']).status_code == 200


@pytest.mark.parametrize('payload', [{'query': ''}, {'query': 'hello', 'top_k': 0}, {'query': 'hello', 'top_k': True},
    {'query': 'hello', 'mode': 'made-up'}, {'query': 'hello', 'rerank': 'yes'}, {'query': 'hello', 'extra': 'value'}, {'query': 'x' * 2001}])
def test_search_validation(client, payload):
    assert client.post('/api/search', json=payload).status_code == 422


def test_document_validation_and_absent_document(client):
    assert client.post('/api/documents', json={'title': '  ', 'text': 'Text'}).status_code == 422
    assert client.post('/api/documents', json={'title': 'Name', 'text': ''}).status_code == 422
    assert client.get('/api/documents/not-here').status_code == 404


def test_search_scope_and_exact_unicode_citation(client):
    text = '👋 Café policy. Annual leave is 24 days. Employees submit requests before travel.'
    doc = add(client, text=text)
    second = add(client, title='Other policy', text='Annual leave is 10 days for this synthetic example.')
    payload = {'query': 'annual leave', 'document_ids': [doc['id']]}
    result = client.post('/api/search', json=payload).json()
    assert result['results']
    assert {row['document_id'] for row in result['results']} == {doc['id']}
    for row in result['results']:
        citation = row['citation']
        assert text[citation['start']:citation['end']] == citation['quote']


def test_offline_answer_and_no_evidence(client):
    add(client)
    response = client.post('/api/answer', json={'query': 'How many days of annual leave?'})
    assert response.status_code == 200
    result = response.json()
    assert result['provider'] == 'extractive' and result['citations']
    assert result['citations'][0]['id'] == 'S1'
    assert '[S1]' in result['answer']
    assert client.post('/api/answer', json={'query': 'unfindablezephyr'}).json()['abstained'] is True


def test_dense_missing_index_is_explicit_and_no_fallback(client):
    add(client)
    response = client.post('/api/search', json={'query': 'leave', 'mode': 'dense'})
    assert response.status_code == 503
    assert 'index' in response.json()['detail']


def test_generation_errors_are_reported_without_network(client, monkeypatch):
    add(client)
    def unavailable(*args, **kwargs):
        raise RuntimeError('Synthetic provider unavailable.')
    monkeypatch.setattr('ragkit.api.answer', unavailable)
    response = client.post('/api/answer', json={'query': 'leave', 'provider': 'openai'})
    assert response.status_code == 503
    assert response.json()['detail'] == 'Synthetic provider unavailable.'


def test_upload_text_json_and_replace_explicit_id(client):
    result = client.post('/api/upload', files={'file': ('notes.md', b'# Notes\nAnnual leave is 24 days.', 'text/markdown')})
    assert result.status_code == 201
    assert result.json()['documents'][0]['title'] == 'notes.md'
    records = [{'id': 'policy-one', 'title': 'Policy', 'text': 'Version one.'}]
    response = client.post('/api/upload', files={'file': ('docs.json', json.dumps(records).encode(), 'application/json')})
    assert response.status_code == 201
    records[0]['text'] = 'Version two.'
    assert client.post('/api/upload', files={'file': ('docs.json', json.dumps(records).encode())}).status_code == 201
    assert client.get('/api/documents/policy-one').json()['text'] == 'Version two.'


def test_upload_validates_all_records_before_ingestion(client):
    records = [{'id': 'valid', 'title': 'Good', 'text': 'Good text'}, {'title': 'Bad', 'text': 'Text', 'source': ''}]
    response = client.post('/api/upload', files={'file': ('docs.json', json.dumps(records).encode())})
    assert response.status_code == 422
    assert client.get('/api/documents').json()['documents'] == []


@pytest.mark.parametrize('field', ['title', 'text', 'source'])
def test_invalid_unicode_upload_cannot_partially_replace_documents(client, field):
    original = {'id': 'existing', 'title': 'Original', 'text': 'Original evidence.'}
    assert client.post('/api/upload', files={'file': ('docs.json', json.dumps([original]).encode())}).status_code == 201
    before = client.get('/api/status').json()['revision']
    replacement = {**original, 'text': 'Replacement evidence.'}
    invalid = {'id': 'invalid', 'title': 'Later record', 'text': 'Later evidence.', 'source': 'docs.json'}
    invalid[field] += '\ud800'
    response = client.post('/api/upload', files={'file': ('docs.json', json.dumps([replacement, invalid]).encode())})
    assert response.status_code == 422
    assert 'valid Unicode' in response.json()['detail']
    assert client.get('/api/documents/existing').json()['text'] == original['text']
    assert client.get('/api/documents/invalid').status_code == 404
    assert client.get('/api/status').json()['revision'] == before


def test_upload_limits_and_unsupported_content(client):
    assert client.post('/api/upload', files={'file': ('notes.py', b'print(1)')}).status_code == 422
    assert client.post('/api/upload', files={'file': ('notes.txt', b'x' * (8 * 1024 * 1024 + 1))}).status_code == 413
    response = client.post('/api/documents', content=b'{}', headers={'content-type': 'application/json', 'content-length': str(MAX_REQUEST_BYTES + 1)})
    assert response.status_code == 413


def test_chunked_request_limit(client):
    def chunks():
        yield b'{"text":"'
        for _ in range(10):
            yield b'x' * 1024 * 1024
    response = client.post('/api/documents', content=chunks(), headers={'content-type': 'application/json'})
    assert response.status_code == 413


def test_same_origin_local_host_and_cross_site_boundaries(client):
    assert client.get('/api/status', headers={'host': 'attacker.example'}).status_code == 403
    assert client.post('/api/sample', headers={'origin': 'https://attacker.example'}).status_code == 403
    assert client.post('/api/sample', headers={'origin': 'null'}).status_code == 403
    assert client.post('/api/sample', headers={'origin': 'http://127.0.0.1:9999'}).status_code == 403
    assert client.post('/api/sample', headers={'sec-fetch-site': 'cross-site'}).status_code == 403
    response = client.post('/api/documents', json={'title': 'Local', 'text': 'Same origin works.'}, headers={'origin': 'http://127.0.0.1'})
    assert response.status_code == 201


def test_sample_uses_explicit_shared_ids_and_is_idempotent(client):
    first = client.post('/api/sample')
    assert first.status_code == 201
    ids = {row['id'] for row in first.json()['documents']}
    assert ids and all(item.startswith('sample-') for item in ids)
    again = client.post('/api/sample').json()['documents']
    assert {row['id'] for row in again} == ids
    assert all(row['unchanged'] for row in again)


def test_index_route_calls_engine_off_event_loop(client, monkeypatch):
    import asyncio
    def build():
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        return {'chunks': 1, 'model': 'injected-test-model', 'revision': 1}
    monkeypatch.setattr(client.app.state.engine, 'build_dense_index', build)
    assert client.post('/api/index').json()['model'] == 'injected-test-model'
