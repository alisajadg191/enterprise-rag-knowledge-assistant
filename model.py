import json
import math
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

class ModelUnavailable(Exception):
    pass


class Ollama:
    def __init__(self):
        self.url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        self.chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")
        self.embedding_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    def post(self, path, payload):
        request = Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except (OSError, URLError, ValueError) as error:
            raise ModelUnavailable("Ollama request failed") from error

    def embed(self, texts, *, query=False):
        prefix = "search_query: " if query else "search_document: "
        response = self.post("/api/embed", {
            "model": self.embedding_model,
            "input": [prefix + text for text in texts],
            "truncate": False,
        })
        try:
            vectors = response["embeddings"]
            width = len(vectors[0])
            if not width or len(vectors) != len(texts):
                raise ValueError
            normalized = []
            for vector in vectors:
                length = math.hypot(*vector)
                if len(vector) != width or not math.isfinite(length) or length == 0:
                    raise ValueError
                normalized.append([value / length for value in vector])
            return normalized
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelUnavailable("Ollama returned invalid embeddings") from error
