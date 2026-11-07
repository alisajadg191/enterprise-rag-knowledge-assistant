# Enterprise RAG Knowledge Assistant

A local FastAPI service that answers questions about internal policies. It retrieves passages using semantic search and links each answer to the document it came from.

## Run locally

You need Python 3.13 and a running Ollama service.

```bash
git clone https://github.com/alisajadg191/enterprise-rag-knowledge-assistant.git
cd enterprise-rag-knowledge-assistant
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen3:8b
.venv/bin/python -m uvicorn app:app --host 127.0.0.1
```

Open [Swagger UI](http://127.0.0.1:8000/docs) and select **Try it out** on each endpoint.

1. Add a policy with **POST /documents** and this body:

   ```json
   {"title":"Production access","text":"Production access requires a manager's approval and expires after 24 hours."}
   ```

2. Send `{"question":"Who must approve access to production?"}` to **POST /search** to inspect the ranked passages and similarity scores.
3. Send the same question to **POST /ask**. The answer cites the production access policy. Open **GET /documents/{document_id}** with the ID from the citation to verify it.
4. Ask when the cafeteria opens. With only the production access policy indexed, the API should return `insufficient_evidence`.
5. Use **PUT /documents/{document_id}** to change the policy or **DELETE /documents/{document_id}** to remove it. Search reflects the new state.

## How it works

1. **Ingest.** Split policy text into overlapping 120-word passages. Embed each passage through Ollama. Save the document and its passage vectors in a single SQLite transaction.
2. **Retrieve.** Embed the question and rank passages by cosine similarity. `/search` exposes the ranked passages so retrieval quality can be debugged independently.
3. **Answer.** Pass sufficiently similar passages to Qwen, which selects passage IDs using a structured response. The server filters out invalid and duplicate IDs and assembles an extractive answer from the selected passages. Each citation includes a document ID, title and passage position.

`RAG_DB` overrides the default `knowledge.sqlite3` path. Use `OLLAMA_URL`, `OLLAMA_EMBED_MODEL`, and `OLLAMA_CHAT_MODEL` to override model settings. Reindex documents after changing the embedding model. The adapter uses Nomic-style query and passage prefixes.

This is a single-user portfolio MVP for loopback use. SQLite keeps setup small, and retrieval scans vectors in memory. Answers quote the selected passages and their relevance still depends on retrieval and model selection. The similarity threshold was checked against a small synthetic corpus.

## Tests

```bash
.venv/bin/python -m pytest -q
RUN_LIVE=1 .venv/bin/python -m pytest -q
```

The first command is also run by GitHub Actions and uses a deterministic model stub. The second runs the same suite plus real Ollama checks for three policy questions and one unsupported question. Both use disposable SQLite databases.
