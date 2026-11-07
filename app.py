from fastapi import FastAPI


def create_app():
    api = FastAPI(title="Enterprise RAG Knowledge Assistant")

    @api.get("/health")
    def health():
        return {"status": "ok"}

    return api


app = create_app()
