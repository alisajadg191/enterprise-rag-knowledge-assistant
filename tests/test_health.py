from fastapi.testclient import TestClient

from app import create_app


def test_health(tmp_path):
    with TestClient(create_app(tmp_path / "health.sqlite3")) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
