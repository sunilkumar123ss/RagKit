"""Transactional SQLite corpus; dense metadata points to a complete generation."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading


class CorpusStore:
    def __init__(self, path: Path, lock=None):
        self.lock = lock or threading.RLock()
        self.connection = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        with self.connection:
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT OR IGNORE INTO metadata VALUES ('schema_version', '1');
                INSERT OR IGNORE INTO metadata VALUES ('revision', '0');
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT,
                    text TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    start INTEGER NOT NULL, end INTEGER NOT NULL, text TEXT NOT NULL,
                    CHECK(start >= 0 AND end > start));
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id, start);
            """)
        if self._metadata("schema_version") != "1":
            self.connection.close()
            raise RuntimeError("unsupported RagKit database schema version")

    def _metadata(self, key):
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def revision(self) -> int:
        with self.lock:
            return int(self._metadata("revision"))

    def dense_state(self) -> dict | None:
        with self.lock:
            value = self._metadata("dense_index")
            return json.loads(value) if value else None

    def set_dense_state(self, state: dict, expected_revision: int):
        with self.lock, self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            if self.revision() != expected_revision:
                raise RuntimeError("corpus changed during indexing; rebuild the dense index")
            self.connection.execute("INSERT OR REPLACE INTO metadata VALUES ('dense_index', ?)",
                                    (json.dumps(state, sort_keys=True),))

    def list_documents(self) -> list[dict]:
        with self.lock:
            documents = [dict(row) for row in self.connection.execute("""
                SELECT d.id, d.title, d.source, d.text,
                       d.updated_at, count(c.id) AS chunks
                FROM documents AS d LEFT JOIN chunks AS c ON c.document_id=d.id
                GROUP BY d.id ORDER BY d.id
            """)]
            for document in documents:
                document["characters"] = len(document.pop("text"))
            return documents

    def get_document(self, document_id: str) -> dict:
        with self.lock:
            row = self.connection.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            if row is None:
                raise KeyError(document_id)
            document = dict(row)
            document["characters"] = len(document["text"])
            document["chunks"] = self.connection.execute(
                "SELECT count(*) FROM chunks WHERE document_id=?", (document_id,)).fetchone()[0]
            return document

    def replace(self, document: dict, chunks: list[dict]) -> tuple[dict, bool]:
        """The previous document survives any failed insert or revision update."""
        with self.lock, self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            old = self.connection.execute("SELECT * FROM documents WHERE id=?", (document["id"],)).fetchone()
            if old and all(old[key] == document[key] for key in ("title", "source", "text")):
                return self.get_document(document["id"]), True
            self.connection.execute("DELETE FROM documents WHERE id=?", (document["id"],))
            self.connection.execute("INSERT INTO documents (id,title,source,text,updated_at) VALUES (?,?,?,?,?)",
                                    tuple(document[key] for key in ("id", "title", "source", "text", "updated_at")))
            self.connection.executemany("INSERT INTO chunks (id,document_id,start,end,text) VALUES (?,?,?,?,?)",
                                        [tuple(chunk[key] for key in ("id", "document_id", "start", "end", "text"))
                                         for chunk in chunks])
            self.connection.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")
            return self.get_document(document["id"]), False

    def delete(self, document_id: str) -> bool:
        with self.lock, self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            deleted = self.connection.execute("DELETE FROM documents WHERE id=?", (document_id,)).rowcount > 0
            if deleted:
                self.connection.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")
            return deleted

    def chunks(self, document_ids=None) -> list[dict]:
        with self.lock:
            if document_ids == []:
                return []
            sql = ("SELECT c.*, d.title, d.source FROM chunks c "
                   "JOIN documents d ON d.id=c.document_id")
            params = ()
            if document_ids is not None:
                sql += " WHERE c.document_id IN (" + ",".join("?" for _ in document_ids) + ")"
                params = tuple(document_ids)
            return [dict(row) for row in self.connection.execute(sql + " ORDER BY c.id", params)]

    def close(self):
        with self.lock:
            self.connection.close()
