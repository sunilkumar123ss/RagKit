"""Bounded document loaders. Uploaded content is data, never executable markup."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import PurePosixPath
import re

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_DOCUMENT_CHARACTERS = 1_000_000
MAX_JSON_RECORDS = 200
MAX_PDF_PAGES = 250
SUPPORTED_EXTENSIONS = {'.txt', '.md', '.json', '.pdf'}


def load_documents(filename, content):
    """Public CLI/API loader: return title, text, source and optional explicit id."""
    documents = []
    for record in load_upload(filename, content):
        document = {key: record[key] for key in ('title', 'text', 'source')}
        if record['document_id'] is not None:
            document['id'] = record['document_id']
        documents.append(document)
    return documents


def _string(value, field, *, required=True, limit=None):
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be a nonblank string.')
    if limit is not None and len(value) > limit:
        raise ValueError(f'{field} must contain at most {limit:,} characters.')
    # JSON permits escaped lone surrogates that Python decodes into strings,
    # but SQLite and chunk identifiers require UTF-8. Reject them while the
    # complete upload is still being validated, before replacing any record.
    try:
        value.encode('utf-8')
    except UnicodeEncodeError as error:
        raise ValueError(f'{field} must contain valid Unicode text.') from error
    return value


def load_json_records(value, default_source=None):
    """Accept one {title,text,id?,source?} object, a list, or {documents:[...]}.

    Validate the complete upload before any record is ingested. IDs replace a
    matching document; omitted IDs use the engine's stable identity behavior.
    """
    if isinstance(value, dict) and set(value) == {'documents'}:
        value = value['documents']
    records = value if isinstance(value, list) else [value]
    if not records or len(records) > MAX_JSON_RECORDS:
        raise ValueError(f'JSON must contain between 1 and {MAX_JSON_RECORDS} document records.')
    documents = []
    ids = set()
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f'JSON record {index} must be an object with title and text.')
        if set(record) - {'id', 'title', 'text', 'source'}:
            raise ValueError(f'JSON record {index} has unsupported fields; use id, title, text and source.')
        title = _string(record.get('title'), f'Record {index} title', limit=500)
        text = _string(record.get('text'), f'Record {index} text', limit=MAX_DOCUMENT_CHARACTERS)
        source = _string(record.get('source', default_source), f'Record {index} source', required=False, limit=2048)
        document_id = record.get('id')
        if document_id is not None:
            if not isinstance(document_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,127}', document_id):
                raise ValueError(f'Record {index} id must be 1–128 letters, digits, dots, underscores, colons or hyphens.')
            if document_id in ids:
                raise ValueError(f'Duplicate document id in JSON: {document_id}.')
            ids.add(document_id)
        documents.append({'title': title, 'text': text, 'source': source, 'document_id': document_id})
    return documents


def load_upload(filename, content):
    """Return validated engine input records from bytes; never save upload paths."""
    if not isinstance(content, bytes):
        raise TypeError('Upload content must be bytes.')
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError('Uploads must be 8 MB or smaller.')
    name = PurePosixPath(str(filename or '').replace('\\', '/')).name
    extension = PurePosixPath(name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError('Supported uploads are .txt, .md, .json and .pdf.')
    if not content:
        raise ValueError('The uploaded file is empty.')
    if extension == '.pdf':
        text = _pdf_text(content)
    else:
        try:
            text = content.decode('utf-8-sig')
        except UnicodeDecodeError as error:
            raise ValueError('Text, Markdown and JSON uploads must use UTF-8 encoding.') from error
    if extension == '.json':
        try:
            return load_json_records(json.loads(text), default_source=name)
        except json.JSONDecodeError as error:
            raise ValueError(f'Invalid JSON at line {error.lineno}, column {error.colno}.') from error
    return load_json_records({'title': name, 'text': text, 'source': name})


def _pdf_text(content):
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError('PDF uploads require the optional pypdf package. Install pypdf or RagKit with its web extra.') from error
    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise ValueError('Encrypted PDFs are not supported. Upload a decrypted text copy.')
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ValueError(f'PDF uploads are limited to {MAX_PDF_PAGES} pages.')
        pieces = []
        count = 0
        for page in reader.pages:
            page_text = page.extract_text() or ''
            count += len(page_text) + 2
            if count > MAX_DOCUMENT_CHARACTERS:
                raise ValueError('Extracted PDF text exceeds the one-million-character document limit.')
            pieces.append(page_text)
        text = '\n\n'.join(pieces)
        if not text.strip():
            raise ValueError('No extractable text was found. Scanned PDFs need OCR before upload.')
        return text
    except ValueError:
        raise
    except Exception as error:
        raise ValueError('The PDF could not be read. Use a valid, unencrypted text PDF.') from error
