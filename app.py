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
from query import answer_question

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
async def ingest_stream():
    """Server-Sent Events endpoint — streams pipeline progress to the browser."""

    def generate():
        for event in run_pipeline():
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
