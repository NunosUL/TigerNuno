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


@app.get("/about/diffs", response_class=HTMLResponse)
async def about_diffs():
    return Path("static/about-diffs.html").read_text(encoding="utf-8")


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
    work_items: bool = True,
    repos:   str = "",  # comma-separated repo names
    plans:   str = "",  # comma-separated plan IDs
    suites:  str = "",  # comma-separated suite IDs
    tcs:     str = "",  # comma-separated test case IDs
    areas:   str = "",  # comma-separated area paths
):
    """Server-Sent Events endpoint — streams pipeline progress to the browser."""
    selected_repos       = [r.strip() for r in repos.split(",")  if r.strip()] if repos  else []
    selected_plan_ids    = [int(x)   for x in plans.split(",")  if x.strip()] if plans  else []
    selected_suite_ids   = [int(x)   for x in suites.split(",") if x.strip()] if suites else []
    selected_tc_ids      = [int(x)   for x in tcs.split(",")    if x.strip()] if tcs    else []
    selected_area_paths  = [a.strip() for a in areas.split(",") if a.strip()] if areas  else []

    def generate():
        for event in run_pipeline(
            crawl=crawl,
            crawl_wiki=wiki,
            crawl_code=code,
            crawl_tests=tests,
            crawl_work_items=work_items,
            selected_repos=selected_repos,
            selected_plan_ids=selected_plan_ids   or None,
            selected_suite_ids=selected_suite_ids or None,
            selected_tc_ids=selected_tc_ids       or None,
            selected_area_paths=selected_area_paths or None,
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

@app.get("/api/areas")
async def list_areas():
    """Returns the full area tree for the project as a flat list with depth info."""
    from ingest import ORG, PROJECT, _API_BASE, _API_VERSION, _HEADERS
    try:
        r = req.get(
            f"https://dev.azure.com/{ORG}/{PROJECT}/_apis/wit/classificationnodes/areas",
            headers=_HEADERS,
            params={"api-version": _API_VERSION, "$depth": 10},
        )
        r.raise_for_status()
        nodes: list[dict] = []

        def walk(node: dict, depth: int = 0, parent_path: str = "") -> None:
            name = node.get("name", "")
            path = f"{parent_path}\\{name}" if parent_path else name
            nodes.append({"path": path, "name": name, "depth": depth})
            for child in node.get("children", []):
                walk(child, depth + 1, path)

        walk(r.json())
        return {"areas": nodes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/areas/counts")
async def area_work_item_counts(areas: str = ""):
    """Returns work item counts broken down by type for the given area paths.
    areas = comma-separated area path strings, e.g. "NetProjects10\\WercsSmart\\Retail"
    Each type is queried in parallel to stay under the 20 000-item WIQL limit.
    Returns truncated=True per type when the result hits the 20K hard limit.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed as cf_completed
    from ingest import _API_BASE, _API_VERSION, _HEADERS, _WI_TYPES

    raw = [a.strip() for a in (areas or "").split(",") if a.strip()]

    # Server-side minimal cover: remove any path already covered by an ancestor.
    # The UI does this too, but defend against direct API calls with redundant paths.
    raw_sorted = sorted(raw, key=len)
    selected: list[str] = []
    for path in raw_sorted:
        if not any(path == p or path.startswith(p + "\\") for p in selected):
            selected.append(path)

    area_clause = ""
    if selected:
        area_parts = " OR ".join(f"[System.AreaPath] UNDER '{p}'" for p in selected)
        area_clause = f" AND ({area_parts})"

    # Returns (wi_type, count, is_truncated)
    def count_type(wi_type: str) -> tuple[str, int, bool]:
        wiql = {
            "query": (
                f"SELECT [System.Id] FROM WorkItems "
                f"WHERE [System.WorkItemType] = '{wi_type}'{area_clause} "
                f"ORDER BY [System.Id]"
            )
        }
        try:
            r = req.post(
                f"{_API_BASE}/wit/wiql?api-version={_API_VERSION}&$top=20000",
                headers=_HEADERS,
                json=wiql,
            )
            if not r.ok:
                try:
                    msg = r.json().get("message", "")
                except Exception:
                    msg = r.text[:300]
                # VS402337 = Azure DevOps 20K hard limit
                if "VS402337" in msg or "20000" in msg:
                    return wi_type, 20000, True
                return wi_type, 0, False
            items = r.json().get("workItems", [])
            # If we got exactly 20K, we may have hit the limit without an error response
            truncated = len(items) >= 20000
            return wi_type, len(items), truncated
        except Exception:
            return wi_type, 0, False

    try:
        counts: dict[str, int] = {}
        truncated: dict[str, bool] = {}
        with ThreadPoolExecutor(max_workers=len(_WI_TYPES)) as pool:
            futs = {pool.submit(count_type, t): t for t in _WI_TYPES}
            for fut in cf_completed(futs):
                wi_type, n, trunc = fut.result()
                counts[wi_type] = n
                truncated[wi_type] = trunc
        # Return in canonical order
        ordered = {t: counts.get(t, 0) for t in _WI_TYPES}
        trunc_ordered = {t: truncated.get(t, False) for t in _WI_TYPES}
        total = sum(ordered.values())
        total_truncated = any(trunc_ordered.values())
        return {
            "counts": ordered,
            "truncated": trunc_ordered,
            "total": total,
            "total_truncated": total_truncated,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/api/repos/counts")
async def list_repo_counts(repos: str = ""):
    """Returns eligible file counts per repo for the selected repo IDs.
    repos = comma-separated repo IDs (GUIDs) as returned by /api/repos.
    Returns {"counts": {"RepoName": N, ...}, "total": N}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed as cf_completed
    from ingest import _API_BASE, _API_VERSION, _HEADERS, _should_index_file, CODE_MAX_FILE_BYTES

    selected_ids = [r.strip() for r in (repos or "").split(",") if r.strip()]
    if not selected_ids:
        return {"counts": {}, "total": 0}

    # Fetch all repos to resolve id→name and get defaultBranch
    try:
        r = req.get(
            f"{_API_BASE}/git/repositories",
            headers=_HEADERS,
            params={"api-version": _API_VERSION},
        )
        r.raise_for_status()
        all_repos = r.json().get("value", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sel_set = set(selected_ids)
    target_repos = [
        repo for repo in all_repos
        if repo["id"] in sel_set
        and not repo.get("isDisabled")
        and repo.get("defaultBranch")
    ]

    def count_repo(repo: dict) -> tuple[str, int]:
        repo_id   = repo["id"]
        repo_name = repo["name"]
        items: list[dict] = []
        cont_token = None
        while True:
            params: dict = {
                "api-version":    _API_VERSION,
                "recursionLevel": "full",
                "scopePath":      "/",
                "versionDescriptor.version": repo["defaultBranch"].replace("refs/heads/", ""),
                "versionDescriptor.versionType": "branch",
                "$top": 10000,
            }
            if cont_token:
                params["continuationToken"] = cont_token
            try:
                resp = req.get(
                    f"{_API_BASE}/git/repositories/{repo_id}/items",
                    headers=_HEADERS,
                    params=params,
                    timeout=30,
                )
            except Exception:
                break
            if not resp.ok:
                break
            items.extend(resp.json().get("value", []))
            cont_token = resp.headers.get("x-ms-continuationtoken")
            if not cont_token:
                break

        count = sum(
            1 for item in items
            if _should_index_file(item) and (item.get("size") or 0) <= CODE_MAX_FILE_BYTES
        )
        return repo_name, count

    try:
        counts: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=min(6, len(target_repos))) as pool:
            futs = {pool.submit(count_repo, repo): repo for repo in target_repos}
            for fut in cf_completed(futs):
                try:
                    name, n = fut.result()
                    counts[name] = n
                except Exception:
                    pass
        return {"counts": counts, "total": sum(counts.values())}
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


@app.get("/api/debug/wi/{wi_id}")
async def debug_wi(wi_id: int):
    """Diagnostic: shows indexed chunks, raw relations, and ArtifactLink regex results for one WI.
    Open in browser: /api/debug/wi/301827
    """
    import re
    from azure.search.documents import SearchClient
    from azure.core.credentials import AzureKeyCredential
    from query import SEARCH_ENDPOINT, SEARCH_INDEX_NAME, SEARCH_API_KEY
    from ingest import _API_BASE, _API_VERSION, _HEADERS

    # ── 1. What is in the search index for this WI? ──────────────────────────
    sha_re = re.compile(r'\*sha:([0-9a-f]{40})\*', re.IGNORECASE)
    index_chunks: list[dict] = []
    found_shas: list[str] = []
    try:
        sc = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX_NAME, AzureKeyCredential(SEARCH_API_KEY))
        for r in sc.search(search_text="*", filter=f"path eq '/work-items/{wi_id}'",
                           select=["id", "text"], top=50):
            text = r["text"]
            shas = sha_re.findall(text)
            found_shas.extend(shas)
            index_chunks.append({
                "id":      r["id"],
                "chars":   len(text),
                "has_commits_header": "## Git Commits" in text,
                "sha_markers": shas,
                "preview": text[:400],
            })
    except Exception as e:
        index_chunks = [{"error": str(e)}]

    # ── 1b. Check whether diff docs exist for the SHAs found in WI chunks ────
    diff_docs: list[dict] = []
    for sha in list(dict.fromkeys(found_shas))[:10]:   # unique, preserve order, cap at 10
        try:
            hits = list(sc.search(search_text="*", filter=f"path eq '/commit-diffs/{sha}'",
                                  select=["id", "text", "path"], top=5))
            if hits:
                diff_docs.append({"sha": sha[:7], "chunk_count": len(hits), "preview": hits[0]["text"][:300]})
            else:
                diff_docs.append({"sha": sha[:7], "chunk_count": 0, "note": "NOT IN INDEX"})
        except Exception as e:
            diff_docs.append({"sha": sha[:7], "error": str(e)})

    # ── 2. Raw Azure DevOps data — relations and ArtifactLinks ───────────────
    artifact_links: list[dict] = []
    wi_meta: dict = {}
    try:
        r = req.get(f"{_API_BASE}/wit/workitems/{wi_id}",
                    headers=_HEADERS,
                    params={"$expand": "all", "api-version": _API_VERSION})
        if r.ok:
            data = r.json()
            fields = data.get("fields", {})
            wi_meta = {
                "title":    fields.get("System.Title"),
                "type":     fields.get("System.WorkItemType"),
                "state":    fields.get("System.State"),
                "relation_count": len(data.get("relations") or []),
            }
            import urllib.parse as _urlparse
            commit_re = re.compile(
                r"vstfs:///Git/Commit/(?:[^/]+/)?([^/]+)/([a-fA-F0-9]{40})",
                re.IGNORECASE
            )
            for rel in (data.get("relations") or []):
                rel_type = rel.get("rel", "")
                raw_url  = rel.get("url", "")
                url      = _urlparse.unquote(raw_url.strip().rstrip("\x00"))
                entry = {
                    "rel":         rel_type,
                    "raw_url":     repr(raw_url),
                    "decoded_url": url,
                }
                if rel_type == "ArtifactLink":
                    m = commit_re.match(url)
                    entry["regex_match"] = bool(m)
                    if m:
                        entry["repo_id"]   = m.group(1)
                        entry["commit_id"] = m.group(2)
                artifact_links.append(entry)
        else:
            wi_meta = {"http_error": r.status_code, "detail": r.text[:300]}
    except Exception as e:
        wi_meta = {"error": str(e)}

    return {
        "wi_id":          wi_id,
        "chunk_count":    len(index_chunks),
        "index_chunks":   index_chunks,
        "diff_docs":      diff_docs,
        "wi_meta":        wi_meta,
        "artifact_links": artifact_links,
    }


@app.get("/api/debug/wi/{wi_id}/dev-info")
async def debug_wi_dev_info(wi_id: int):
    """Step-by-step trace of every commit-resolution stage for a work item.

    Open in browser: /api/debug/wi/301827/dev-info
    """
    import re as _re
    import urllib.parse as _urlparse
    import requests as _requests
    from ingest import _API_BASE, _HEADERS, _API_VERSION

    trace: list[dict] = []

    # Step 1 — fetch WI individually with $expand=all
    try:
        r = _requests.get(
            f"{_API_BASE}/wit/workitems/{wi_id}",
            headers=_HEADERS,
            params={"$expand": "all", "api-version": _API_VERSION},
            timeout=15,
        )
        trace.append({"step": "fetch_wi", "http_status": r.status_code, "ok": r.ok})
        if not r.ok:
            return {"wi_id": wi_id, "trace": trace, "error": f"WI fetch HTTP {r.status_code}"}
        relations = r.json().get("relations") or []
        artifact_links = [rel for rel in relations if rel.get("rel") == "ArtifactLink"]
        trace.append({
            "step": "relations",
            "total_relations": len(relations),
            "artifact_link_count": len(artifact_links),
        })
    except Exception as e:
        return {"wi_id": wi_id, "trace": trace, "error": f"WI fetch exception: {e}"}

    # Step 2 — regex-match each ArtifactLink
    commit_re = _re.compile(
        r"vstfs:///Git/Commit/(?:[^/]+/)?([^/]+)/([a-fA-F0-9]{40})", _re.IGNORECASE
    )
    commit_candidates: list[dict] = []
    for rel in artifact_links:
        raw = rel.get("url", "").strip().rstrip("\x00")
        url = _urlparse.unquote(raw)
        m = commit_re.match(url)
        entry = {"raw_url": raw, "decoded_url": url, "regex_match": bool(m)}
        if m:
            entry["repo_id"]   = m.group(1)
            entry["commit_id"] = m.group(2).lower()
            commit_candidates.append(entry)
        trace.append({"step": "regex", **entry})

    trace.append({"step": "commit_candidates", "count": len(commit_candidates)})

    # Step 3 — for each matched commit, fetch its metadata and changes
    commit_results: list[dict] = []
    for c in commit_candidates:
        repo_id   = c["repo_id"]
        commit_id = c["commit_id"]
        result: dict = {"commit_id": commit_id[:7], "commit_id_full": commit_id, "repo_id": repo_id}
        try:
            cr = _requests.get(
                f"{_API_BASE}/git/repositories/{repo_id}/commits/{commit_id}",
                headers=_HEADERS,
                params={"api-version": _API_VERSION},
                timeout=15,
            )
            result["commit_http_status"] = cr.status_code
            result["commit_ok"] = cr.ok
            if cr.ok:
                c_data = cr.json()
                result["message"] = (c_data.get("comment") or "").splitlines()[0] if c_data.get("comment") else ""
                result["author"]  = (c_data.get("author") or {}).get("name", "")
                result["date"]    = ((c_data.get("author") or {}).get("date") or "")[:10]
                parents = c_data.get("parents") or []
                result["parent_count"] = len(parents)
                # Parents may be plain SHA strings or dicts — handle both
                p0 = parents[0] if parents else None
                result["parent_id"] = (p0.get("objectId") or p0.get("commitId") if isinstance(p0, dict) else str(p0 or ""))
            else:
                result["commit_error_body"] = cr.text[:300]
        except Exception as e:
            result["commit_exception"] = str(e)

        try:
            chg_r = _requests.get(
                f"{_API_BASE}/git/repositories/{repo_id}/commits/{commit_id}/changes",
                headers=_HEADERS,
                params={"api-version": _API_VERSION},
                timeout=15,
            )
            result["changes_http_status"] = chg_r.status_code
            result["changes_ok"] = chg_r.ok
            if chg_r.ok:
                changes = chg_r.json().get("changes", [])
                result["file_count"] = len(changes)
                result["files"] = [
                    {"path": ch.get("item", {}).get("path"), "type": ch.get("changeType")}
                    for ch in changes[:20]
                ]
            else:
                result["changes_error_body"] = chg_r.text[:300]
        except Exception as e:
            result["changes_exception"] = str(e)

        commit_results.append(result)
        trace.append({"step": "commit_detail", **result})

    return {
        "wi_id":            wi_id,
        "commit_count":     len(commit_candidates),
        "commits_fetched":  sum(1 for c in commit_results if c.get("commit_ok")),
        "trace":            trace,
        "commit_results":   commit_results,
    }


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


@app.delete("/api/snapshot")
async def delete_snapshot():
    """Deletes the crawl snapshot JSONL blob so the next crawl=OFF run has no data to load."""
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceNotFoundError
    from ingest import STORAGE_CONN, CONTAINER, ORG, PROJECT
    blob_name = f"rag_{ORG}_{PROJECT}.jsonl"
    try:
        client = BlobServiceClient.from_connection_string(STORAGE_CONN)
        client.get_blob_client(container=CONTAINER, blob=blob_name).delete_blob()
        return {"deleted": True, "blob": blob_name}
    except ResourceNotFoundError:
        return {"deleted": True, "note": "snapshot was already absent", "blob": blob_name}
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
        total      = count()
        code       = count("path ge '/repos/' and path lt '/repos~'")
        tests      = count("path ge '/test-cases/' and path lt '/test-cases~'")
        work_items = count("path ge '/work-items/' and path lt '/work-items~'")
        wiki       = total - code - tests - work_items
        return {
            "total":      total,
            "wiki":       wiki,
            "code":       code,
            "test_cases": tests,
            "work_items": work_items,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# RAG chat endpoint
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question:     str
    source_types: list[str] = []


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = answer_question(req.question,
                                source_types=req.source_types or None)
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
            for event in answer_question_stream(req.question,
                                                source_types=req.source_types or None):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
