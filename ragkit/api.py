"""Local FastAPI application for the persistent RagKit document workspace."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers

from .answers import answer, provider_status
from .engine import RagKit
from .loaders import MAX_UPLOAD_BYTES, load_documents, load_json_records

STATIC = Path(__file__).parent / 'static'
MAX_REQUEST_BYTES = 9 * 1024 * 1024  # Multipart framing plus an 8 MB file.
LOCAL_HOSTS = {'localhost', '127.0.0.1', '::1'}


class LocalBoundary:
    """Reject nonlocal Hosts and cross-origin browser requests; bound body bytes."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        host = headers.get('host', '')
        try:
            parsed = urlsplit('http://' + host)
            valid_host = parsed.hostname in LOCAL_HOSTS and not parsed.username and not parsed.password
            _ = parsed.port
            valid_host = valid_host and not parsed.path and not parsed.query and not parsed.fragment
        except ValueError:
            valid_host = False
        error = None
        code = 403
        if not valid_host:
            error = 'RagKit accepts only localhost, 127.0.0.1 or ::1 Host headers.'
        elif scope['method'] not in {'GET', 'HEAD', 'OPTIONS'}:
            origin = headers.get('origin')
            if origin:
                try:
                    parsed_origin = urlsplit(origin)
                    allowed_origin = (parsed_origin.scheme == scope.get('scheme', 'http')
                                      and parsed_origin.netloc.lower() == host.lower()
                                      and not parsed_origin.path and not parsed_origin.query
                                      and not parsed_origin.fragment)
                except ValueError:
                    allowed_origin = False
                if not allowed_origin:
                    error = 'Browser writes must come from this RagKit origin.'
            if headers.get('sec-fetch-site') == 'cross-site':
                error = 'Cross-site browser writes are not allowed.'
        try:
            length = int(headers.get('content-length', '0'))
        except ValueError:
            length = -1
        if length < 0:
            error, code = 'Invalid Content-Length.', 400
        elif length > MAX_REQUEST_BYTES:
            error, code = 'Request body exceeds the 9 MB limit.', 413
        if error:
            return await JSONResponse({'detail': error}, status_code=code)(scope, receive, send)
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > MAX_REQUEST_BYTES:
                    raise HTTPException(413, 'Request body exceeds the 9 MB limit.')
            return message

        await self.app(scope, bounded_receive, send)


class DocumentInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=1_000_000)
    source: str | None = Field(default=None, max_length=2048)


class SearchInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20, strict=True)
    mode: Literal['lexical', 'dense', 'hybrid'] = 'lexical'
    rerank: bool = Field(default=False, strict=True)
    document_ids: list[str] | None = Field(default=None, max_length=200)


class AnswerInput(SearchInput):
    provider: Literal['extractive', 'ollama', 'openai', 'anthropic'] = 'extractive'


def create_app(data_dir=None):
    directory = Path(data_dir or os.environ.get('RAGKIT_DATA_DIR', '.ragkit')).expanduser()

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = await run_in_threadpool(RagKit, directory)
        try:
            yield
        finally:
            await run_in_threadpool(app.state.engine.close)

    app = FastAPI(title='RagKit', version='1.0.0', lifespan=lifespan,
                  description='A local document workspace with inspectable retrieval and optional model providers.')
    app.add_middleware(LocalBoundary)

    @app.middleware('http')
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Frame-Options'] = 'DENY'
        if request.url.path == '/' or request.url.path.startswith('/static/'):
            response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(ValueError)
    async def value_error(request, error):
        return JSONResponse({'detail': str(error)}, status_code=422)

    @app.exception_handler(KeyError)
    async def missing_document(request, error):
        return JSONResponse({'detail': 'Document not found.'}, status_code=404)

    @app.exception_handler(RuntimeError)
    async def unavailable(request, error):
        return JSONResponse({'detail': str(error)}, status_code=503)

    @app.get('/api/status')
    async def status(request: Request):
        stats = await run_in_threadpool(request.app.state.engine.stats)
        providers = await run_in_threadpool(provider_status)
        return {**stats, 'providers': providers}

    @app.get('/api/documents')
    async def documents(request: Request):
        return {'documents': await run_in_threadpool(request.app.state.engine.list_documents)}

    @app.get('/api/documents/{document_id}')
    async def document(document_id: str, request: Request):
        return await run_in_threadpool(request.app.state.engine.get_document, document_id)

    @app.delete('/api/documents/{document_id}')
    async def delete_document(document_id: str, request: Request):
        removed = await run_in_threadpool(request.app.state.engine.delete_document, document_id)
        if not removed:
            raise HTTPException(404, 'Document not found.')
        return {'deleted': True, 'id': document_id}

    @app.post('/api/documents', status_code=201)
    async def create_document(payload: DocumentInput, request: Request):
        return await run_in_threadpool(request.app.state.engine.ingest_text, **payload.model_dump())

    def ingest_records(engine, records):
        return {'documents': [engine.ingest_text(record['text'], record['title'], source=record.get('source'),
                                                document_id=record.get('id', record.get('document_id')))
                              for record in records]}

    @app.post('/api/upload', status_code=201)
    async def upload(request: Request, file: UploadFile = File(...)):
        try:
            data = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, 'Uploads must be 8 MB or smaller.')
            records = await run_in_threadpool(load_documents, file.filename, data)
            return await run_in_threadpool(ingest_records, request.app.state.engine, records)
        finally:
            await file.close()

    @app.post('/api/search')
    async def search(payload: SearchInput, request: Request):
        return await run_in_threadpool(request.app.state.engine.search, **payload.model_dump())

    @app.post('/api/answer')
    async def ask(payload: AnswerInput, request: Request):
        return await run_in_threadpool(answer, request.app.state.engine, **payload.model_dump())

    @app.post('/api/index')
    async def build_index(request: Request):
        return await run_in_threadpool(request.app.state.engine.build_dense_index)

    @app.post('/api/sample', status_code=201)
    async def sample(request: Request):
        def load_sample():
            data = json.loads((Path(__file__).parent / 'data/sample.json').read_text(encoding='utf-8'))
            return ingest_records(request.app.state.engine, load_json_records(data))
        return await run_in_threadpool(load_sample)

    @app.get('/', include_in_schema=False)
    async def index():
        return FileResponse(STATIC / 'index.html')

    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    return app
