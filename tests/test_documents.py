from fastapi.testclient import TestClient

from app import chunks, create_app
from model import ModelUnavailable


def test_document_lifecycle(client):
    assert client.get("/documents").json() == []
    created = client.post("/documents", json={"title": "Policy", "text": "File expenses within 30 days."})
    assert created.status_code == 201
    doc = created.json()
    url = f"/documents/{doc['id']}"
    assert client.get("/documents").json() == [{"id": doc["id"], "title": "Policy"}]
    assert client.get(url).json() == doc
    update = client.put(url, json={"title": "Policy", "text": "File expenses within 45 days."})
    assert update.status_code == 200
    assert client.get(url).json() == update.json()
    assert client.delete(url).status_code == 204
    assert client.get(url).status_code == 404
    assert client.delete(url).status_code == 404


def test_chunk_overlap():
    text = " ".join(str(number) for number in range(12))
    assert list(chunks(text, size=5, overlap=2)) == ["0 1 2 3 4", "3 4 5 6 7", "6 7 8 9 10", "9 10 11"]


def test_rejects_blank_documents(client):
    assert client.post("/documents", json={"title": "Blank", "text": " "}).status_code == 422


def test_model_failure_leaves_no_partial_document(tmp_path):
    class BrokenModel:
        def embed(self, texts):
            raise ModelUnavailable("Model offline")

    with TestClient(create_app(tmp_path / "test.sqlite3", BrokenModel())) as client:
        response = client.post("/documents", json={"title": "Policy", "text": "File expenses."})
        assert response.status_code == 503
        assert client.get("/documents").json() == []
