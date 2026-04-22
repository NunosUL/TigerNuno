"""
FastAPI web server
Serves the ingestion UI, chat UI, and about page.
Exposes SSE for ingestion progress and POST /api/chat for RAG queries.

Run with: uvicorn app:app --reload
"""

import json
from pathlib import Path

import requests as req
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
    repos:   str = "",  # comma-separated repo names
    plans:   str = "",  # comma-separated plan IDs
    suites:  str = "",  # comma-separated suite IDs
    tcs:     str = "",  # comma-separated test case IDs
):
    """Server-Sent Events endpoint — streams pipeline progress to the browser."""
    selected_repos     = [r.strip() for r in repos.split(",")  if r.strip()] if repos  else []
    selected_plan_ids  = [int(x)   for x in plans.split(",")  if x.strip()] if plans  else []
    selected_suite_ids = [int(x)   for x in suites.split(",") if x.strip()] if suites else []
    selected_tc_ids    = [int(x)   for x in tcs.split(",")    if x.strip()] if tcs    else []

    def generate():
        for event in run_pipeline(
            crawl=crawl,
            crawl_wiki=wiki,
            crawl_code=code,
            crawl_tests=tests,
            selected_repos=selected_repos,
            selected_plan_ids=selected_plan_ids   or None,
            selected_suite_ids=selected_suite_ids or None,
            selected_tc_ids=selected_tc_ids       or None,
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


@app.get("/api/test-plans")
async def list_test_plans():
    """Returns all test plans for the project."""
    from ingest import _API_BASE, _API_VERSION, _HEADERS
    try:
        plans, params = [], {"api-version": _API_VERSION, "$top": 500}
        while True:
            r = req.get(f"{_API_BASE}/testplan/plans", headers=_HEADERS, params=params)
            r.raise_for_status()
            plans.extend(r.json().get("value", []))
            token = r.headers.get("x-ms-continuationtoken")
            if not token:
                break
            params = {"api-version": _API_VERSION, "$top": 500, "continuationToken": token}
        return {"plans": [{"id": p["id"], "name": p.get("name", f"Plan {p['id']}")} for p in plans]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test-suites")
async def list_all_test_suites():
    """Returns all suites across all plans in one call — used to populate the suite picker upfront."""
    from ingest import _API_BASE, _API_VERSION, _HEADERS
    try:
        # Fetch all plans first
        plans, params = [], {"api-version": _API_VERSION, "$top": 500}
        while True:
            r = req.get(f"{_API_BASE}/testplan/plans", headers=_HEADERS, params=params)
            r.raise_for_status()
            plans.extend(r.json().get("value", []))
            token = r.headers.get("x-ms-continuationtoken")
            if not token:
                break
            params = {"api-version": _API_VERSION, "$top": 500, "continuationToken": token}

        # Fetch suites for every plan
        all_suites = []
        for plan in plans:
            plan_id   = plan["id"]
            plan_name = plan.get("name", f"Plan {plan_id}")
            suites, sp = [], {"api-version": _API_VERSION, "$top": 500}
            while True:
                r = req.get(f"{_API_BASE}/testplan/plans/{plan_id}/suites",
                            headers=_HEADERS, params=sp)
                if not r.ok:
                    break
                suites.extend(r.json().get("value", []))
                token = r.headers.get("x-ms-continuationtoken")
                if not token:
                    break
                sp = {"api-version": _API_VERSION, "$top": 500, "continuationToken": token}
            for s in suites:
                if s.get("name") != plan_name:
                    all_suites.append({
                        "id":        s["id"],
                        "name":      s.get("name", f"Suite {s['id']}"),
                        "plan_id":   plan_id,
                        "plan_name": plan_name,
                    })
        return {"suites": all_suites}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test-plans/{plan_id}/suites")
async def list_test_suites(plan_id: int):
    """Returns all suites for a given test plan (excluding the root suite)."""
    from ingest import _API_BASE, _API_VERSION, _HEADERS
    try:
        suites, params = [], {"api-version": _API_VERSION, "$top": 500}
        while True:
            r = req.get(f"{_API_BASE}/testplan/plans/{plan_id}/suites",
                        headers=_HEADERS, params=params)
            r.raise_for_status()
            suites.extend(r.json().get("value", []))
            token = r.headers.get("x-ms-continuationtoken")
            if not token:
                break
            params = {"api-version": _API_VERSION, "$top": 500, "continuationToken": token}
        # Exclude root suite (same name as plan) — it contains no direct test cases
        plan_r = req.get(f"{_API_BASE}/testplan/plans/{plan_id}",
                         headers=_HEADERS, params={"api-version": _API_VERSION})
        plan_name = plan_r.json().get("name", "") if plan_r.ok else ""
        result = [
            {"id": s["id"], "name": s.get("name", f"Suite {s['id']}"),
             "parent_id": s.get("parentSuite", {}).get("id")}
            for s in suites if s.get("name") != plan_name
        ]
        return {"suites": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test-plans/{plan_id}/suites/{suite_id}/test-cases")
async def list_suite_test_cases(plan_id: int, suite_id: int):
    """Returns test cases in a specific suite."""
    from ingest import _API_BASE, _API_VERSION, _HEADERS
    try:
        tcs, params = [], {"api-version": _API_VERSION, "$top": 500}
        while True:
            r = req.get(
                f"{_API_BASE}/testplan/plans/{plan_id}/suites/{suite_id}/testcase",
                headers=_HEADERS, params=params)
            r.raise_for_status()
            tcs.extend(r.json().get("value", []))
            token = r.headers.get("x-ms-continuationtoken")
            if not token:
                break
            params = {"api-version": _API_VERSION, "$top": 500, "continuationToken": token}
        result = [
            {"id": tc.get("workItem", {}).get("id"),
             "name": tc.get("workItem", {}).get("name", f"TC {tc.get('workItem',{}).get('id')}")}
            for tc in tcs if tc.get("workItem", {}).get("id")
        ]
        return {"test_cases": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test-cases")
async def list_test_cases_batch(suites: str = ""):
    """Returns test cases for multiple plan:suite pairs in one parallel call.
    suites = comma-separated planId:suiteId pairs, e.g. "42:101,42:102,55:200"
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed as cf_completed
    from ingest import _API_BASE, _API_VERSION, _HEADERS

    pairs: list[tuple[int, int]] = []
    for token in (suites or "").split(","):
        token = token.strip()
        if ":" not in token:
            continue
        try:
            pid, sid = token.split(":", 1)
            pairs.append((int(pid), int(sid)))
        except ValueError:
            continue

    if not pairs:
        return {"test_cases": []}

    def fetch_suite(plan_id: int, suite_id: int) -> list[dict]:
        tcs, params = [], {"api-version": _API_VERSION, "$top": 500}
        while True:
            r = req.get(
                f"{_API_BASE}/testplan/plans/{plan_id}/suites/{suite_id}/testcase",
                headers=_HEADERS, params=params)
            if not r.ok:
                break
            tcs.extend(r.json().get("value", []))
            token = r.headers.get("x-ms-continuationtoken")
            if not token:
                break
            params = {"api-version": _API_VERSION, "$top": 500, "continuationToken": token}
        return [
            {"id": tc.get("workItem", {}).get("id"),
             "name": tc.get("workItem", {}).get("name", f"TC {tc.get('workItem', {}).get('id')}")}
            for tc in tcs if tc.get("workItem", {}).get("id")
        ]

    try:
        seen: set[int] = set()
        all_tcs: list[dict] = []
        with ThreadPoolExecutor(max_workers=min(8, len(pairs))) as pool:
            futs = {pool.submit(fetch_suite, pid, sid): (pid, sid) for pid, sid in pairs}
            for fut in cf_completed(futs):
                try:
                    for tc in fut.result():
                        if tc["id"] not in seen:
                            seen.add(tc["id"])
                            all_tcs.append(tc)
                except Exception:
                    pass
        return {"test_cases": all_tcs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/debug/tc/{tc_id}")
async def debug_tc(tc_id: int):
    """Diagnostic: shows every indexed chunk + the raw DevOps fields for one TC.
    Open in browser: /api/debug/tc/296199
    """
    from azure.search.documents import SearchClient
    from azure.core.credentials import AzureKeyCredential
    from query import SEARCH_ENDPOINT, SEARCH_INDEX_NAME, SEARCH_API_KEY
    from ingest import _API_BASE, _API_VERSION, _HEADERS

    # ── 1. What is actually in the search index? ─────────────────────────────
    index_chunks: list[dict] = []
    try:
        sc = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX_NAME, AzureKeyCredential(SEARCH_API_KEY))
        for r in sc.search(search_text="*", filter=f"path eq '/test-cases/{tc_id}'",
                           select=["id", "text"], top=50):
            index_chunks.append({"id": r["id"], "chars": len(r["text"]),
                                  "preview": r["text"][:300]})
    except Exception as e:
        index_chunks = [{"error": str(e)}]

    # ── 2. What does Azure DevOps actually return for this work item? ─────────
    devops: dict = {}
    try:
        r = req.get(f"{_API_BASE}/wit/workitems/{tc_id}",
                    headers=_HEADERS,
                    params={"$expand": "all", "api-version": _API_VERSION})
        if r.ok:
            fields = r.json().get("fields", {})
            steps_raw = fields.get("Microsoft.VSTS.TCM.Steps") or ""
            devops = {
                "title":             fields.get("System.Title"),
                "state":             fields.get("System.State"),
                "description_chars": len(fields.get("System.Description") or ""),
                "steps_chars":       len(steps_raw),
                "steps_preview":     steps_raw[:500] if steps_raw else None,
                "has_description":   bool(fields.get("System.Description")),
                "has_steps":         bool(steps_raw),
                "all_field_keys":    sorted(fields.keys()),
            }
        else:
            devops = {"http_error": r.status_code, "detail": r.text[:300]}
    except Exception as e:
        devops = {"error": str(e)}

    return {"tc_id": tc_id, "chunk_count": len(index_chunks),
            "index_chunks": index_chunks, "devops": devops}


@app.delete("/api/manifest")
async def clear_manifest():
    """Deletes the hash manifest blob so the next run re-indexes everything from scratch."""
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceNotFoundError
    from ingest import STORAGE_CONN, CONTAINER, HASH_MANIFEST_BLOB
    try:
        client = BlobServiceClient.from_connection_string(STORAGE_CONN)
        client.get_blob_client(container=CONTAINER, blob=HASH_MANIFEST_BLOB).delete_blob()
        return {"cleared": True}
    except ResourceNotFoundError:
        return {"cleared": True, "note": "manifest was already absent"}
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
