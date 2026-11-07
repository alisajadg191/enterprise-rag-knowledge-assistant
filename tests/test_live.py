import os

import pytest
from fastapi.testclient import TestClient

from app import create_app

POLICIES = {
    "Expenses": "Submit expense claims within 30 days of purchase. Attach an itemised receipt.",
    "Access": "Production access requires a manager's approval and expires after 24 hours.",
    "Incident response": "Report a suspected security incident to the Security Operations team within one hour.",
}
QUESTIONS = [
    ("When do I need to submit an expense claim?", "Expenses", "30 days"),
    ("Who must approve access to production?", "Access", "manager"),
    ("Who should I tell about a suspected security incident?", "Incident response", "Security Operations"),
]


@pytest.mark.skipif(os.getenv("RUN_LIVE") != "1", reason="Requires a running Ollama service")
def test_live_retrieval_citations_and_abstention(tmp_path):
    with TestClient(create_app(tmp_path / "live.sqlite3")) as client:
        for title, text in POLICIES.items():
            response = client.post("/documents", json={"title": title, "text": text})
            assert response.status_code == 201, response.text

        for question, title, phrase in QUESTIONS:
            payload = {"question": question, "limit": 1}
            search = client.post("/search", json=payload)
            assert search.status_code == 200, search.text
            assert search.json()[0]["title"] == title

            result = client.post("/ask", json=payload)
            assert result.status_code == 200, result.text
            answer = result.json()
            assert answer["status"] == "answered", question
            citation, = answer["citations"]
            assert citation["title"] == title
            assert citation["text"] == POLICIES[title]
            assert phrase in answer["answer"]

        result = client.post("/ask", json={"question": "What time does the cafeteria open?"})
        assert result.status_code == 200, result.text
        assert result.json()["status"] == "insufficient_evidence"
