"""
FastAPI web server
Serves the ingestion UI, chat UI, and about page.
Exposes SSE for ingestion progress and POST /api/chat for RAG queries.

Run with: uvicorn app:app --reload
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ingest import run_pipeline
from query import answer_question, answer_question_stream

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def landing():
    return Path("static/landing.html").read_text(encoding="utf-8")


@app.get("/ingest", response_class=HTMLResponse)
async def ingest():
    return Path("static/index.html").read_text(encoding="utf-8")


@app.get("/chat", response_class=HTMLResponse)
async def chat():
    return Path("static/chat.html").read_text(encoding="utf-8")


@app.get("/about", response_class=HTMLResponse)
async def about():
    return Path("static/about.html").read_text(encoding="utf-8")


@app.get("/chat/about", response_class=HTMLResponse)
async def chat_about():
    return Path("static/chat-about.html").read_text(encoding="utf-8")


@app.get("/chat/about/scoring", response_class=HTMLResponse)
async def chat_about_scoring():
    return Path("static/scoring.html").read_text(encoding="utf-8")


@app.get("/synergies", response_class=HTMLResponse)
async def synergies():
    return Path("static/synergies.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Ingestion SSE stream
# ---------------------------------------------------------------------------

@app.get("/ingest/stream")
async def ingest_stream(
    crawl: bool = True,
    wiki:  bool = True,
    code:  bool = True,
    tests: bool = True,
    repos: str  = "",   # comma-separated repo names selected in the UI
):
    """Server-Sent Events endpoint — streams pipeline progress to the browser."""
    selected_repos = [r.strip() for r in repos.split(",") if r.strip()] if repos else []

    def generate():
        for event in run_pipeline(
            crawl=crawl,
            crawl_wiki=wiki,
            crawl_code=code,
            crawl_tests=tests,
            selected_repos=selected_repos,
        ):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: {\"step\": \"__done__\"}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Index diagnostics
# ---------------------------------------------------------------------------

@app.get("/api/repos")
async def list_repos():
    """Returns all non-disabled, non-empty Git repositories for the repo picker."""
    import base64, requests as req
    from ingest import ORG, PROJECT, _API_BASE, _API_VERSION, _HEADERS
    try:
        r = req.get(
            f"{_API_BASE}/git/repositories",
            headers=_HEADERS,
            params={"api-version": _API_VERSION},
        )
        r.raise_for_status()
        repos = [
            {"id": repo["id"], "name": repo["name"]}
            for repo in r.json().get("value", [])
            if not repo.get("isDisabled") and repo.get("defaultBranch")
        ]
        return {"repos": repos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/index/stats")
async def index_stats():
    """Returns document counts per source type — useful for confirming what is indexed."""
    from azure.search.documents import SearchClient
    from azure.core.credentials import AzureKeyCredential
    from query import SEARCH_ENDPOINT, SEARCH_INDEX_NAME, SEARCH_API_KEY

    client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )

    def count(f=None):
        r = client.search(search_text="*", filter=f, include_total_count=True, top=0)
        return r.get_count()

    try:
        total = count()
        code  = count("path ge '/repos/' and path lt '/repos~'")
        tests = count("path ge '/test-cases/' and path lt '/test-cases~'")
        wiki  = total - code - tests
        return {
            "total":      total,
            "wiki":       wiki,
            "code":       code,
            "test_cases": tests,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# RAG chat endpoint
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = answer_question(req.question)
        return {
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """SSE endpoint — streams per-step progress events then the final answer."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def generate():
        try:
            for event in answer_question_stream(req.question):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
