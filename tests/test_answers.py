def test_answer_cites_only_retrieved_passages(client):
    text = "Production access requires a manager's approval."
    response = client.post("/documents", json={"title": "Access", "text": text})
    assert response.status_code == 201
    document_id = response.json()["id"]

    result = client.post("/ask", json={"question": "Who approves production access?"})
    assert result.status_code == 200
    answer = result.json()
    assert answer["status"] == "answered"
    assert answer["answer"] == f"[1] {text}"
    citation, = answer["citations"]
    assert citation["source_id"] == 1
    assert citation["document_id"] == document_id
    assert citation["position"] == 1
    assert citation["text"] == client.get(f"/documents/{document_id}").json()["text"]


def test_answer_abstains_when_no_passage_matches(client):
    assert client.post("/documents", json={"title": "Access", "text": "Production access requires approval."}).status_code == 201
    result = client.post("/ask", json={"question": "Where is the cafeteria?"})
    assert result.status_code == 200
    assert result.json() == {"answer": None, "citations": [], "status": "insufficient_evidence"}
