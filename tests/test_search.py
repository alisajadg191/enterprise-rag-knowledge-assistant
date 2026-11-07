def test_search_ranks_passages_and_tracks_changes(client):
    docs = [
        {"title": "Access", "text": "Production access requires approval."},
        {"title": "Expenses", "text": "Submit expense claims within 30 days."},
    ]
    for document in docs:
        assert client.post("/documents", json=document).status_code == 201

    query = {"question": "When are expense claims due?", "limit": 1}
    response = client.post("/search", json=query)
    assert response.status_code == 200
    assert response.json() == [{
        "document_id": 2, "title": "Expenses", "position": 1,
        "text": "Submit expense claims within 30 days.", "score": 1.0,
    }]

    update = client.put("/documents/2", json={**docs[1], "text": "Expense claims are due in 45 days."})
    assert update.status_code == 200
    assert client.post("/search", json=query).json()[0]["text"] == update.json()["text"]
    assert client.delete("/documents/2").status_code == 204
    assert client.post("/search", json=query).json()[0]["document_id"] == 1


def test_search_empty_corpus_and_input_validation(client):
    assert client.post("/search", json={"question": "Who approves production access?"}).json() == []
    assert client.post("/search", json={"question": " "}).status_code == 422
    assert client.post("/search", json={"question": "Policy", "limit": 0}).status_code == 422
