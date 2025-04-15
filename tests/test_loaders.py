import json
from io import BytesIO
import sys

import pytest

from ragkit.loaders import MAX_UPLOAD_BYTES, load_documents


def test_text_unicode_and_filename_path_are_preserved_safely():
    result = load_documents('../../notes.md', '\ufeffHello 👋 café\n<script>not executable</script>'.encode())
    assert result == [{'title': 'notes.md', 'source': 'notes.md', 'text': 'Hello 👋 café\n<script>not executable</script>'}]


@pytest.mark.parametrize('payload', [
    {'id': 'doc:one', 'title': 'Policy', 'text': 'Original text'},
    [{'id': 'doc:one', 'title': 'Policy', 'text': 'Original text'}],
    {'documents': [{'id': 'doc:one', 'title': 'Policy', 'text': 'Original text'}]},
])
def test_json_record_shapes(payload):
    result = load_documents('records.json', json.dumps(payload).encode())
    assert result == [{'id': 'doc:one', 'title': 'Policy', 'text': 'Original text', 'source': 'records.json'}]


@pytest.mark.parametrize('payload', [[], {'title': '', 'text': 'Text'}, {'title': 'Name', 'text': '  '},
    {'title': 'Name', 'text': 'Text', 'source': ''}, {'title': 'Name', 'text': 'Text', 'id': '../escape'},
    {'title': 'Name', 'text': 'Text', 'code': 'ignored?'}, {'title': 9, 'text': 'Text'},
    [{'id': 'same', 'title': 'A', 'text': 'A'}, {'id': 'same', 'title': 'B', 'text': 'B'}]])
def test_json_rejects_invalid_records(payload):
    with pytest.raises(ValueError):
        load_documents('records.json', json.dumps(payload).encode())


def test_invalid_encoding_json_empty_and_extension():
    for name, content in [('notes.txt', b'\xff'), ('notes.json', b'{invalid'), ('notes.md', b''), ('run.py', b'print(1)')]:
        with pytest.raises(ValueError):
            load_documents(name, content)


def test_size_and_character_limits():
    with pytest.raises(ValueError, match='8 MB'):
        load_documents('notes.txt', b'x' * (MAX_UPLOAD_BYTES + 1))
    with pytest.raises(ValueError, match='1,000,000'):
        load_documents('notes.txt', b'x' * 1_000_001)


def test_pdf_optional_dependency_has_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pypdf', None)
    with pytest.raises(RuntimeError, match='pypdf'):
        load_documents('notes.pdf', b'%PDF-1.4')


def test_blank_and_encrypted_pdf_fail_clearly():
    pypdf = pytest.importorskip('pypdf')
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=300, height=300)
    stream = BytesIO()
    writer.write(stream)
    with pytest.raises(ValueError, match='OCR'):
        load_documents('blank.pdf', stream.getvalue())
    writer.encrypt('synthetic-password')
    encrypted = BytesIO()
    writer.write(encrypted)
    with pytest.raises(ValueError, match='Encrypted'):
        load_documents('encrypted.pdf', encrypted.getvalue())


def test_pdf_extracts_actual_text():
    pypdf = pytest.importorskip('pypdf')
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
    content = DecodedStreamObject()
    content.set_data(b'BT /F1 12 Tf 20 250 Td (Synthetic PDF policy: 24 days.) Tj ET')
    page[NameObject('/Contents')] = content
    stream = BytesIO()
    writer.write(stream)
    assert '24 days' in load_documents('policy.pdf', stream.getvalue())[0]['text']
