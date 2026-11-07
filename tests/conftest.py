import pytest
from fastapi.testclient import TestClient

from app import Selection, create_app


class StubModel:
    embedding_model = "test"

    def embed(self, texts, *, query=False):
        return [[float(term in text.lower()) for term in ("expense", "production", "incident")]
                for text in texts]


    def chat(self, instruction, data, schema):
        assert schema is Selection
        return Selection(source_ids=[1, 1, 999])


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "test.sqlite3", StubModel())) as client:
        yield client
