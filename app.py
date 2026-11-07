import json
import os
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StringConstraints

from model import ModelUnavailable, Ollama

Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
Body = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]


class Document(BaseModel):
    title: Title
    text: Body


@contextmanager
def database(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def chunks(text, size=120, overlap=24):
    words = text.split()
    for start in range(0, len(words), size - overlap):
        yield " ".join(words[start:start + size])
        if start + size >= len(words):
            break


def create_app(path=None, model=None):
    path = path or os.getenv("RAG_DB", "knowledge.sqlite3")
    model = model or Ollama()

    @asynccontextmanager
    async def lifespan(_app):
        with database(path) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, text TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    position INTEGER, text TEXT NOT NULL, vector TEXT NOT NULL,
                    model TEXT NOT NULL, PRIMARY KEY (document_id, position)
                );
            """)
        yield

    api = FastAPI(title="Enterprise RAG Knowledge Assistant", lifespan=lifespan)

    @api.exception_handler(ModelUnavailable)
    async def model_error(_request: Request, error: ModelUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @api.get("/health")
    def health():
        return {"status": "ok"}

    def index(db, document_id, passages, vectors):
        db.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
            [(document_id, i, passage, json.dumps(vector), model.embedding_model)
             for i, (passage, vector) in enumerate(zip(passages, vectors, strict=True), start=1)],
        )

    @api.post("/documents", status_code=201)
    def create_document(document: Document):
        passages = list(chunks(document.text))
        vectors = model.embed(passages)
        with database(path) as db:
            row = db.execute("INSERT INTO documents(title, text) VALUES (?, ?)",
                             (document.title, document.text))
            index(db, row.lastrowid, passages, vectors)
            return {"id": row.lastrowid, **document.model_dump()}

    @api.get("/documents")
    def list_documents():
        with database(path) as db:
            return [dict(row) for row in db.execute("SELECT id, title FROM documents ORDER BY id")]

    @api.get("/documents/{document_id}")
    def get_document(document_id: int):
        with database(path) as db:
            row = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Document not found")
        return dict(row)

    @api.put("/documents/{document_id}")
    def replace_document(document_id: int, document: Document):
        get_document(document_id)
        passages = list(chunks(document.text))
        vectors = model.embed(passages)
        with database(path) as db:
            row = db.execute("UPDATE documents SET title = ?, text = ? WHERE id = ?",
                             (document.title, document.text, document_id))
            if not row.rowcount:
                raise HTTPException(404, "Document not found")
            db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            index(db, document_id, passages, vectors)
        return {"id": document_id, **document.model_dump()}

    @api.delete("/documents/{document_id}", status_code=204)
    def delete_document(document_id: int):
        with database(path) as db:
            row = db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            if not row.rowcount:
                raise HTTPException(404, "Document not found")
        return Response(status_code=204)

    return api


app = create_app()
