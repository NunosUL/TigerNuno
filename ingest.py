"""
RAG Ingestion Pipeline
======================
1. Crawl  — fetches content from three Azure DevOps sources:
              • Wiki pages   (Azure DevOps Wiki REST API)
              • Code files   (Azure DevOps Git repositories)
              • Test cases   (Azure DevOps Test Management)
            Saves a combined JSONL snapshot to Azure Blob Storage.
2. Chunk  — splits changed records into chunks using LangChain's
            RecursiveCharacterTextSplitter; skips unchanged records
            via MD5 hash change detection.
3. Embed  — generates 3072-dim vectors via Azure OpenAI text-embedding-3-large.
4. Index  — creates/ensures Azure AI Search index (HNSW vector + full-text).
5. Upload — upserts documents + vectors into Azure AI Search and saves
            the updated hash manifest.

Source toggles: set CRAWL_WIKI / CRAWL_CODE / CRAWL_TESTS to "false" in .env
to disable individual sources without changing code.
"""

import hashlib
import html as html_lib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Generator
import base64
import difflib
import random
import urllib.parse

import markdown
import requests
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    ScoringProfile,
    TagScoringFunction,
    TagScoringParameters,
)
from azure.storage.blob import BlobServiceClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — Azure DevOps identity
# ---------------------------------------------------------------------------

PAT     = os.environ["AZURE_DEVOPS_PAT"]
ORG     = os.environ.get("DEVOPS_ORG", "ulpsi")
PROJECT = os.environ.get("DEVOPS_PROJECT", "NetProjects10")

_API_BASE    = f"https://dev.azure.com/{ORG}/{PROJECT}/_apis"
_API_VERSION = "7.1"
_encoded     = base64.b64encode(f":{PAT}".encode()).decode()
_HEADERS     = {"Authorization": f"Basic {_encoded}", "Content-Type": "application/json"}
_DL_HEADERS  = {"Authorization": f"Basic {_encoded}"}   # no Content-Type for downloads

# ---------------------------------------------------------------------------
# Config — source toggles
# ---------------------------------------------------------------------------

CRAWL_WIKI  = os.environ.get("CRAWL_WIKI",  "true").lower() == "true"
CRAWL_CODE  = os.environ.get("CRAWL_CODE",  "true").lower() == "true"
CRAWL_TESTS = os.environ.get("CRAWL_TESTS", "true").lower() == "true"
CRAWL_WORK_ITEMS = os.environ.get("CRAWL_WORK_ITEMS", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Config — code file filtering
# ---------------------------------------------------------------------------

CODE_MAX_FILE_BYTES = 100 * 1024  # 100 KB

CODE_EXTENSIONS = frozenset({
    # .NET / C#
    ".cs", ".csproj", ".sln", ".config", ".resx",
    # Web / Razor
    ".razor", ".html", ".htm", ".js", ".ts", ".less", ".css",
    # Data / config
    ".json", ".xml", ".xslt",
})

CODE_FILENAMES: frozenset[str] = frozenset()  # no special filenames for this stack

SKIP_DIRS = frozenset({
    "node_modules", "__pycache__", ".git", "bin", "obj",
    "dist", "build", ".vs", ".idea", ".vscode", "coverage",
    ".terraform", "wwwroot", ".nuget", "packages", "vendor",
    ".next", ".nuxt", "out", "target", ".cache",
})

SKIP_PATH_PARTS = frozenset({
    "migrations", "Migrations", "migrate",
    "generated", "Generated", "auto-generated",
})

SKIP_FILENAMES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock", "packages.lock.json",
    "composer.lock", "Gemfile.lock", "cargo.lock",
})

SKIP_SUFFIXES = (
    ".min.js", ".min.css", ".bundle.js", ".bundle.min.js",
    ".g.cs", ".designer.cs", ".generated.cs", ".g.dart",
)

# Work-item attachment handling
ATTACH_CODE_EXTS = frozenset({
    # Code and configuration
    ".sql", ".config", ".json", ".xml", ".yaml", ".yml",
    ".ps1", ".sh", ".bat", ".cmd",
    ".cs", ".py", ".js", ".ts", ".razor", ".html", ".css",
    # Plain text / structured text (deployment notes, scripts, changelists)
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".log",
})
ATTACH_MAX_BYTES   = 50 * 1024   # skip attachments larger than 50 KB
CRAWL_WI_ATTACHMENTS = os.environ.get("CRAWL_WI_ATTACHMENTS", "true").lower() == "true"

# Commit diff ingestion — store before/after diffs as separate index documents
CRAWL_COMMIT_DIFFS        = os.environ.get("CRAWL_COMMIT_DIFFS", "true").lower() == "true"
COMMIT_DIFF_MAX_FILE_BYTES = 30 * 1024   # skip file versions larger than 30 KB
COMMIT_DIFF_MAX_FILES      = 20          # cap files per commit diff document

AUTO_GENERATED_MARKERS = frozenset({
    "// <auto-generated",
    "// <autogenerated",
    "/* auto-generated",
    "# this file is auto-generated",
    "// generated by",
    "// do not edit",
    "<autogenerated>",
})

# ---------------------------------------------------------------------------
# Config — Storage, OpenAI, Search
# ---------------------------------------------------------------------------

STORAGE_CONN       = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
CONTAINER          = os.environ.get("AZURE_STORAGE_CONTAINER", "wiki-crawl")
HASH_MANIFEST_BLOB = "rag_hashes.json"

AOAI_ENDPOINT             = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_API_KEY              = os.environ["AZURE_OPENAI_API_KEY"]
AOAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
AOAI_API_VERSION          = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

SEARCH_ENDPOINT   = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_API_KEY    = os.environ["AZURE_SEARCH_API_KEY"]
SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "wiki-index")

CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 150
EMBED_BATCH_SIZE  = 100   # Azure OpenAI accepts up to 2048 inputs; smaller batches reduce 429s
EMBED_CONCURRENCY = 2     # parallel workers — keep low to stay under TPM quota
# Minimum pause each worker takes between successful embedding calls.
# Formula: CONCURRENCY / (avg_call_s + DELAY_S) = RPM.
# Override via env var — raise if your quota is higher, lower if you hit 429s:
#   EMBED_INTER_BATCH_DELAY_S=2.0   (very safe, slower)
#   EMBED_INTER_BATCH_DELAY_S=0.1   (faster, needs a higher Azure quota)
EMBED_INTER_BATCH_DELAY_S = float(os.environ.get("EMBED_INTER_BATCH_DELAY_S", "1.0"))
UPLOAD_BATCH_SIZE = 200  # Azure AI Search supports up to 1000; 200 halves round-trips
VECTOR_DIM        = 3072


# ===========================================================================
# STEP 1A — Wiki crawl  (all project + code wikis)
# ===========================================================================

def _list_all_wikis() -> list[dict]:
    """Return every wiki (project wiki + code wikis) registered in this project."""
    r = requests.get(
        f"https://dev.azure.com/{ORG}/{PROJECT}/_apis/wiki/wikis",
        headers=_HEADERS,
        params={"api-version": _API_VERSION},
    )
    r.raise_for_status()
    return r.json().get("value", [])


def _list_all_wiki_pages(wiki_base: str) -> list[dict]:
    params = {"path": "/", "recursionLevel": "full", "includeContent": "false",
              "api-version": _API_VERSION}
    resp = requests.get(f"{wiki_base}/pages", headers=_HEADERS, params=params)
    resp.raise_for_status()
    pages: list[dict] = []
    _collect_wiki_pages(resp.json(), pages)
    return pages


def _collect_wiki_pages(node: dict, acc: list[dict]) -> None:
    acc.append({"path": node.get("path", "/"), "remote_url": node.get("remoteUrl", "")})
    for child in node.get("subPages", []):
        _collect_wiki_pages(child, acc)


def _fetch_wiki_page(page_path: str, wiki_base: str, wiki_name: str) -> dict:
    """Fetch one wiki page and return an ingest record.

    The record path is prefixed with the wiki name so pages from different
    wikis never collide:  /wiki/WERCSmart.Wiki/Processing-Report
    """
    params = {"path": page_path, "includeContent": "true", "api-version": _API_VERSION}
    resp = requests.get(f"{wiki_base}/pages", headers=_HEADERS, params=params)
    resp.raise_for_status()
    data         = resp.json()
    page_id      = data.get("id")
    md_content   = data.get("content", "")

    # Append page discussion comments if any
    if page_id:
        comments = _fetch_wiki_page_comments(page_id, wiki_base)
        if comments:
            comment_lines = ["\n\n## Page Comments\n"]
            for c in comments:
                author = c.get("createdBy", {}).get("displayName") or \
                         c.get("createdBy", {}).get("uniqueName") or "Unknown"
                date   = (c.get("createdDate") or "")[:10]
                text   = BeautifulSoup(c.get("content") or "", "html.parser").get_text("\n").strip()
                if text:
                    comment_lines += [f"**{author}** ({date}):", "", text, ""]
            md_content += "\n".join(comment_lines)

    html_content = markdown.markdown(md_content, extensions=["extra", "toc", "tables", "fenced_code"])
    links        = _extract_wiki_links(html_content, base_path=page_path, wiki_name=wiki_name)

    # Namespace path: /wiki/<WikiName>/page-path so multiple wikis coexist cleanly
    record_path = f"/wiki/{wiki_name}{page_path}"

    return {
        "path":       record_path,
        "remote_url": data.get("remoteUrl", ""),
        "html":       html_content,
        "markdown":   md_content,
        "links":      links,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }


def _fetch_wiki_page_comments(page_id: int, wiki_base: str) -> list[dict]:
    """Fetch all discussion comments for a wiki page."""
    comments: list[dict] = []
    params: dict = {"api-version": "7.1-preview.1", "$top": 100}
    while True:
        try:
            r = requests.get(
                f"{wiki_base}/pages/{page_id}/comments",
                headers=_HEADERS,
                params=params,
            )
            if not r.ok:
                break
            data  = r.json()
            batch = data.get("value", [])
            comments.extend(batch)
            # Wiki comments API uses $skip-based paging, not a continuation token
            if len(batch) < 100:
                break
            params = {"api-version": "7.1-preview.1", "$top": 100,
                      "$skip": len(comments)}
        except Exception as exc:
            log.debug("Could not fetch comments for wiki page %s: %s", page_id, exc)
            break
    return comments


def _extract_wiki_links(html: str, base_path: str, wiki_name: str) -> list[str]:
    soup           = BeautifulSoup(html, "html.parser")
    wiki_web_base  = f"https://dev.azure.com/{ORG}/{PROJECT}/_wiki/wikis/{wiki_name}"
    links          = []
    for tag in soup.find_all("a", href=True):
        href: str = tag["href"]
        if href.startswith("http"):
            links.append(href)
        elif href.startswith("/"):
            links.append(f"{wiki_web_base}{href}")
        else:
            parent = "/".join(base_path.rstrip("/").split("/")[:-1]) or "/"
            links.append(f"{wiki_web_base}{parent}/{href}")
    return list(dict.fromkeys(links))


# ===========================================================================
# STEP 1B — Code crawl
# ===========================================================================

def _should_index_file(item: dict) -> bool:
    """Return True if this git item should be included in the index."""
    if item.get("gitObjectType") != "blob":
        return False

    path     = item.get("path", "")
    parts    = path.strip("/").split("/")
    filename = parts[-1]
    _, ext   = os.path.splitext(filename)

    # Skip by directory component
    if any(p in SKIP_DIRS for p in parts[:-1]):
        return False
    if any(p in SKIP_PATH_PARTS for p in parts):
        return False

    # Skip by exact filename
    if filename in SKIP_FILENAMES:
        return False

    # Skip by suffix (e.g. .min.js, .g.cs)
    if any(filename.endswith(s) for s in SKIP_SUFFIXES):
        return False

    # Skip .lock extension
    if ext == ".lock":
        return False

    # Allow by exact filename (Dockerfile, Makefile, …)
    if filename in CODE_FILENAMES:
        return True

    return ext.lower() in CODE_EXTENSIONS


def _crawl_code_files(selected_repos: list[str] | None = None):
    """
    Generator yielding ("event", message_str) or ("record", dict).
    Crawls git repositories in the project. If selected_repos is provided (non-empty),
    only those repos are crawled; otherwise all non-disabled repos are crawled.
    """
    repos_resp = requests.get(
        f"{_API_BASE}/git/repositories",
        headers=_HEADERS,
        params={"api-version": _API_VERSION},
    )
    repos_resp.raise_for_status()
    all_repos = repos_resp.json().get("value", [])

    # Apply user selection filter — frontend may send repo IDs (GUIDs) or names
    if selected_repos:
        sel_set = set(selected_repos)
        repos = [r for r in all_repos if r["id"] in sel_set or r["name"] in sel_set]
        yield "event", f"[Code] {len(repos)} of {len(all_repos)} repos selected by user"
    else:
        repos = all_repos
        yield "event", f"[Code] Found {len(repos)} repositor{'y' if len(repos)==1 else 'ies'}"

    for idx, repo in enumerate(repos, 1):
        repo_id   = repo["id"]
        repo_name = repo["name"]

        # Skip disabled repos
        if repo.get("isDisabled"):
            yield "event", f"[Code] Skipping {repo_name} (disabled)"
            continue

        # Skip empty repos — they have no defaultBranch and the items API returns 400
        if not repo.get("defaultBranch"):
            yield "event", f"[Code] Skipping {repo_name} (empty — no commits)"
            continue

        default_branch = repo["defaultBranch"].replace("refs/heads/", "")
        yield "event", f"[Code] Scanning {repo_name} ({idx}/{len(repos)}) @ {default_branch}…"

        # List all items from the default branch, with pagination
        items: list[dict] = []
        cont_token = None
        while True:
            params: dict = {
                "recursionLevel": "full",
                "api-version": _API_VERSION,
                "$top": 2000,
                "scopePath": "/",
                "versionDescriptor.version": default_branch,
                "versionDescriptor.versionType": "branch",
            }
            if cont_token:
                params["continuationToken"] = cont_token
            try:
                ir = requests.get(
                    f"{_API_BASE}/git/repositories/{repo_id}/items",
                    headers=_HEADERS,
                    params=params,
                )
                if not ir.ok:
                    try:
                        detail = ir.json().get("message") or ir.json().get("typeKey") or ir.text[:200]
                    except Exception:
                        detail = ir.text[:200]
                    log.warning("Failed to list items for %s: %s — %s", repo_name, ir.status_code, detail)
                    yield "event", f"[Code] Skipping {repo_name} — {ir.status_code}: {detail}"
                    break
                ir.raise_for_status()
            except requests.exceptions.HTTPError:
                break
            except Exception as exc:
                log.warning("Failed to list items for %s: %s", repo_name, exc)
                yield "event", f"[Code] Skipping {repo_name} — {exc}"
                break

            items.extend(ir.json().get("value", []))
            cont_token = ir.headers.get("x-ms-continuationtoken")
            if not cont_token:
                break

        # First pass — identify eligible items without downloading
        eligible = []
        for item in items:
            if not _should_index_file(item):
                continue
            file_size = item.get("size", 0) or 0
            if file_size > CODE_MAX_FILE_BYTES:
                log.debug("Skip (too large %d B): %s%s", file_size, repo_name, item.get("path", ""))
                continue
            eligible.append(item)

        if not eligible:
            yield "event", f"[Code] {repo_name}: no eligible files"
            continue

        yield "event", f"[Code] {repo_name}: {len(eligible)} eligible file(s) — downloading…"

        # Second pass — download and index
        file_count = 0
        for dl_idx, item in enumerate(eligible, 1):
            file_path = item.get("path", "")

            try:
                cr = requests.get(
                    f"{_API_BASE}/git/repositories/{repo_id}/items",
                    headers=_DL_HEADERS,
                    params={"path": file_path, "download": "true", "api-version": _API_VERSION},
                )
                cr.raise_for_status()
                content = cr.text
            except Exception as exc:
                log.warning("Failed to download %s%s: %s", repo_name, file_path, exc)
                continue

            if not content or not content.strip():
                continue

            if len(content.encode("utf-8", errors="replace")) > CODE_MAX_FILE_BYTES:
                continue

            if any(m in content[:400].lower() for m in AUTO_GENERATED_MARKERS):
                continue

            filename = file_path.split("/")[-1]
            _, ext   = os.path.splitext(filename)
            lang     = ext.lstrip(".") if ext else "text"
            safe     = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            file_count += 1
            yield "record", {
                "path":       f"/repos/{repo_name}{file_path}",
                "remote_url": f"https://dev.azure.com/{ORG}/{PROJECT}/_git/{repo_name}?path={file_path}",
                "html":       f'<pre><code class="language-{lang}">{safe}</code></pre>',
                "markdown":   content,
                "links":      [],
                "crawled_at": datetime.now(timezone.utc).isoformat(),
            }

            if dl_idx % 10 == 0 or dl_idx == len(eligible):
                yield "event", f"[Code] {repo_name}: {dl_idx}/{len(eligible)} downloaded ({file_count} indexed)…"

        yield "event", f"[Code] {repo_name}: done — {file_count}/{len(eligible)} file(s) indexed"


# ===========================================================================
# STEP 1C — Test Management crawl
# ===========================================================================

def _ps_text(elem) -> str:
    """Extract clean text from a TCM <parameterizedString> element.

    Azure DevOps stores step content as HTML inside the XML in two different ways
    depending on the DevOps version / editor used:

    Format A — HTML-encoded text node (most common):
        <parameterizedString>&lt;DIV&gt;&lt;P&gt;Do X&lt;/P&gt;&lt;/DIV&gt;</parameterizedString>
        ElementTree decodes entities → elem.text = "<DIV><P>Do X</P></DIV>"
        elem has NO child elements.

    Format B — actual XML child elements:
        <parameterizedString><DIV><P>Do X</P></DIV></parameterizedString>
        elem.text is None/empty; text lives in child nodes.

    Using elem.text alone (the previous approach) silently returns "" for Format B,
    which is why steps disappeared for test cases written in that format.
    """
    if list(elem):
        # Format B: real child elements — itertext() walks the whole subtree
        return " ".join("".join(elem.itertext()).split())

    # Format A: HTML-encoded text node — ElementTree already decoded the entities,
    # so elem.text IS literal HTML like "<DIV><P>Do X</P></DIV>".
    # Pass it through BeautifulSoup to strip the markup.
    raw = (elem.text or "").strip()
    if not raw:
        return ""
    return " ".join(
        BeautifulSoup(html_lib.unescape(raw), "html.parser").get_text(" ").split()
    )


def _parse_test_steps(
    steps_xml: str,
    shared_steps_cache: dict[int, list[tuple[str, str]]] | None = None,
) -> list[tuple[str, str]]:
    """Parse the TCM steps XML blob into [(action, expected_result), …].

    If shared_steps_cache is provided, SharedStep references are resolved inline
    by substituting the referenced shared step's own parsed steps.
    """
    if not steps_xml:
        return []
    try:
        root = ET.fromstring(steps_xml)
    except ET.ParseError as exc:
        log.warning("Could not parse test steps XML: %s", exc)
        return []

    result = []
    for step in root.iter("step"):
        if step.get("type") == "SharedStep":
            ref_id = step.get("ref")
            if shared_steps_cache and ref_id:
                try:
                    resolved = shared_steps_cache.get(int(ref_id))
                    if resolved:
                        result.extend(resolved)
                        continue
                except (ValueError, TypeError):
                    pass
            # Fallback: no cache or not found — emit a readable placeholder
            result.append((f"[Shared steps from work item {ref_id or '?'}]", ""))
            continue

        strings = step.findall("parameterizedString")
        action  = expected = ""

        if strings:
            action = _ps_text(strings[0])
        if len(strings) > 1:
            expected = _ps_text(strings[1])

        if action or expected:
            result.append((action, expected))

    return result


def _collect_shared_step_ids(steps_xml: str) -> set[int]:
    """Return the set of SharedStep work item IDs referenced in a steps XML blob."""
    if not steps_xml or "SharedStep" not in steps_xml:
        return set()
    try:
        root = ET.fromstring(steps_xml)
        ids = set()
        for step in root.iter("step"):
            if step.get("type") == "SharedStep":
                ref = step.get("ref")
                if ref:
                    try:
                        ids.add(int(ref))
                    except ValueError:
                        pass
        return ids
    except ET.ParseError:
        return set()


def _fetch_work_item_comments(wi_id: int) -> list[dict]:
    """Fetch all discussion comments for a work item (paginated)."""
    comments: list[dict] = []
    params: dict = {"api-version": "7.1-preview.4", "$top": 200}
    while True:
        try:
            r = requests.get(
                f"{_API_BASE}/wit/workitems/{wi_id}/comments",
                headers=_HEADERS,
                params=params,
            )
            if not r.ok:
                break
            data  = r.json()
            batch = data.get("comments", [])
            comments.extend(batch)
            token = data.get("continuationToken")
            if not token or not batch:
                break
            params = {"api-version": "7.1-preview.4", "$top": 200, "continuationToken": token}
        except Exception as exc:
            log.debug("Could not fetch comments for work item %s: %s", wi_id, exc)
            break
    return comments


def _identity_display(field_value) -> str:
    """Return the display name from an identity field (object or plain string)."""
    if isinstance(field_value, dict):
        return field_value.get("displayName", "") or field_value.get("uniqueName", "")
    return str(field_value) if field_value else ""


def _fetch_work_item_attachments(wi_id: int, relations: list[dict]) -> list[dict]:
    """Download text-based attachments (code / config files) for a work item.

    Returns a list of dicts:
        {"name": str, "size": int, "ext": str, "is_code": bool, "content": str | None}

    Only files whose extension is in ATTACH_CODE_EXTS and whose size is ≤
    ATTACH_MAX_BYTES are downloaded.  All other attachments are listed with
    content=None so the markdown still shows what files are attached.
    """
    result: list[dict] = []
    for rel in (relations or []):
        if rel.get("rel") != "AttachedFile":
            continue
        attrs  = rel.get("attributes", {})
        fname  = attrs.get("name", "unknown")
        fsize  = attrs.get("resourceSize", 0) or 0
        furl   = rel.get("url", "")
        _, ext = os.path.splitext(fname)
        is_code = ext.lower() in ATTACH_CODE_EXTS

        content: str | None = None
        if is_code and fsize <= ATTACH_MAX_BYTES and furl and CRAWL_WI_ATTACHMENTS:
            try:
                r = requests.get(furl, headers=_DL_HEADERS, timeout=20)
                if r.ok:
                    content = r.text
            except Exception as exc:
                log.debug("Could not download attachment %s for WI %s: %s", fname, wi_id, exc)

        result.append({
            "name":    fname,
            "size":    fsize,
            "ext":     ext.lower(),
            "is_code": is_code,
            "content": content,
            "added":   (attrs.get("resourceCreatedDate") or "")[:10],
        })
    return result


def _format_test_case_markdown(
    work_item: dict,
    plan_name: str,
    suite_name: str,
    comments: list[dict] | None = None,
    shared_steps_cache: dict[int, list[tuple[str, str]]] | None = None,
) -> str:
    """Serialise a test case work item to a rich Markdown document."""
    fields = work_item.get("fields", {})
    wi_id  = work_item.get("id", "")

    title        = fields.get("System.Title", "")
    state        = fields.get("System.State", "")
    area_path    = fields.get("System.AreaPath", "")
    iteration    = fields.get("System.IterationPath", "")
    assigned_to  = _identity_display(fields.get("System.AssignedTo"))
    created_by   = _identity_display(fields.get("System.CreatedBy"))
    created_date = (fields.get("System.CreatedDate") or "")[:10]
    changed_by   = _identity_display(fields.get("System.ChangedBy"))
    changed_date = (fields.get("System.ChangedDate") or "")[:10]
    priority     = fields.get("Microsoft.VSTS.Common.Priority", "")
    auto_status  = fields.get("Microsoft.VSTS.TCM.AutomationStatus", "")
    tags         = fields.get("System.Tags", "") or ""

    raw_desc    = fields.get("System.Description", "") or ""
    description = BeautifulSoup(raw_desc, "html.parser").get_text("\n").strip()

    raw_accept   = fields.get("Microsoft.VSTS.Common.AcceptanceCriteria", "") or ""
    acceptance   = BeautifulSoup(raw_accept, "html.parser").get_text("\n").strip()

    steps_xml = fields.get("Microsoft.VSTS.TCM.Steps", "") or ""
    steps     = _parse_test_steps(steps_xml, shared_steps_cache=shared_steps_cache)

    lines = [
        f"# Test Case: {title}",
        "",
        f"**ID:** {wi_id}",
        f"**Test Plan:** {plan_name}",
        f"**Test Suite:** {suite_name}",
    ]
    if state:
        lines.append(f"**State:** {state}")
    if priority:
        lines.append(f"**Priority:** {priority}")
    if auto_status:
        lines.append(f"**Automation Status:** {auto_status}")
    if assigned_to:
        lines.append(f"**Assigned To:** {assigned_to}")
    if area_path:
        lines.append(f"**Area Path:** {area_path}")
    if iteration:
        lines.append(f"**Iteration:** {iteration}")
    if tags:
        lines.append(f"**Tags / Feature Areas:** {tags}")
    if created_by:
        lines.append(f"**Created By:** {created_by}" + (f" ({created_date})" if created_date else ""))
    if changed_by:
        lines.append(f"**Last Modified By:** {changed_by}" + (f" ({changed_date})" if changed_date else ""))

    if description:
        lines += ["", "## Description", "", description]

    if acceptance:
        lines += ["", "## Acceptance Criteria", "", acceptance]

    if steps:
        lines += ["", "## Test Steps", "",
                  "| Step | Action | Expected Result |",
                  "|------|--------|-----------------|"]
        for i, (action, expected) in enumerate(steps, 1):
            a = action.replace("|", "\\|")
            e = (expected or "—").replace("|", "\\|")
            lines.append(f"| {i} | {a} | {e} |")

    if comments:
        lines += ["", "## Discussion", ""]
        for comment in comments:
            author   = _identity_display(comment.get("createdBy")) or "Unknown"
            date     = (comment.get("createdDate") or "")[:10]
            raw_text = comment.get("text", "") or ""
            text     = BeautifulSoup(raw_text, "html.parser").get_text("\n").strip()
            if text:
                lines += [f"**{author}** ({date}):", "", text, ""]

    return "\n".join(lines)


def _crawl_test_cases(
    selected_plan_ids: set[int] | None = None,
    selected_suite_ids: set[int] | None = None,
    selected_tc_ids: set[int] | None = None,
):
    """
    Generator yielding ("event", message_str) or ("record", dict).
    Crawls test plans → suites → test cases and batch-fetches work items.
    Test cases appearing in multiple suites are deduplicated by work item ID.

    selected_plan_ids:  if provided, only crawl these plan IDs
    selected_suite_ids: if provided, only crawl these suite IDs
    selected_tc_ids:    if provided, only fetch these work item IDs
    """
    # List all test plans (paginated)
    plans = []
    plan_params = {"api-version": _API_VERSION, "$top": 500}
    while True:
        plans_resp = requests.get(
            f"{_API_BASE}/testplan/plans",
            headers=_HEADERS,
            params=plan_params,
        )
        plans_resp.raise_for_status()
        plans.extend(plans_resp.json().get("value", []))
        token = plans_resp.headers.get("x-ms-continuationtoken")
        if not token:
            break
        plan_params = {"api-version": _API_VERSION, "$top": 500, "continuationToken": token}

    # Filter to selected plans if provided
    if selected_plan_ids:
        plans = [p for p in plans if p["id"] in selected_plan_ids]

    yield "event", f"[Tests] Processing {len(plans)} test plan(s)"

    # Collect unique test case IDs → (plan_name, suite_name)
    tc_map: dict[int, tuple[str, str]] = {}

    for p_idx, plan in enumerate(plans, 1):
        plan_id   = plan["id"]
        plan_name = plan.get("name", f"Plan {plan_id}")

        yield "event", f"[Tests] Plan {p_idx}/{len(plans)}: '{plan_name}' — scanning suites…"

        # Paginate suites — Azure returns x-ms-continuationtoken when more pages exist
        try:
            suites = []
            params = {"api-version": _API_VERSION, "$top": 500}
            while True:
                sr = requests.get(
                    f"{_API_BASE}/testplan/plans/{plan_id}/suites",
                    headers=_HEADERS,
                    params=params,
                )
                sr.raise_for_status()
                suites.extend(sr.json().get("value", []))
                token = sr.headers.get("x-ms-continuationtoken")
                if not token:
                    break
                params = {"api-version": _API_VERSION, "$top": 500, "continuationToken": token}
        except Exception as exc:
            log.warning("Failed to list suites for plan %s: %s", plan_name, exc)
            yield "event", f"[Tests] Could not list suites for '{plan_name}' — skipping"
            continue

        # Filter to selected suites if provided
        if selected_suite_ids:
            suites = [s for s in suites if s["id"] in selected_suite_ids]

        non_root = [s for s in suites if s.get("name") != plan_name]
        yield "event", f"[Tests] '{plan_name}' — found {len(non_root)} suite(s)"

        for s_idx, suite in enumerate(non_root, 1):
            suite_id   = suite["id"]
            suite_name = suite.get("name", f"Suite {suite_id}")

            # Paginate test cases within each suite
            try:
                test_cases = []
                tc_params = {"api-version": _API_VERSION, "$top": 500}
                while True:
                    tcr = requests.get(
                        f"{_API_BASE}/testplan/plans/{plan_id}/suites/{suite_id}/testcase",
                        headers=_HEADERS,
                        params=tc_params,
                    )
                    tcr.raise_for_status()
                    test_cases.extend(tcr.json().get("value", []))
                    token = tcr.headers.get("x-ms-continuationtoken")
                    if not token:
                        break
                    tc_params = {"api-version": _API_VERSION, "$top": 500, "continuationToken": token}
            except Exception as exc:
                log.warning("Failed to list test cases for %s/%s: %s", plan_name, suite_name, exc)
                continue

            # Filter to selected test cases if provided
            if selected_tc_ids:
                test_cases = [tc for tc in test_cases
                              if tc.get("workItem", {}).get("id") in selected_tc_ids]

            added = 0
            for tc in test_cases:
                wi_id = tc.get("workItem", {}).get("id")
                if wi_id and wi_id not in tc_map:
                    tc_map[wi_id] = (plan_name, suite_name)
                    added += 1

            yield "event", (
                f"[Tests] Suite {s_idx}/{len(non_root)}: '{suite_name}' "
                f"→ {len(test_cases)} test case(s) ({added} new)"
            )

    yield "event", f"[Tests] Found {len(tc_map)} unique test case(s) — fetching details…"

    # Batch-fetch work items (≤ 200 per call)
    wi_ids    = list(tc_map.keys())
    work_items_all: list[dict] = []
    n_batches = max(1, (len(wi_ids) + 199) // 200)

    for i in range(0, len(wi_ids), 200):
        batch_num = i // 200 + 1
        batch   = wi_ids[i: i + 200]
        ids_str = ",".join(str(x) for x in batch)
        yield "event", f"[Tests] Fetching work item details — batch {batch_num}/{n_batches}…"
        try:
            wr = requests.get(
                f"{_API_BASE}/wit/workitems",
                headers=_HEADERS,
                params={"ids": ids_str, "$expand": "all", "api-version": _API_VERSION},
            )
            wr.raise_for_status()
            work_items_all.extend(wr.json().get("value", []))
        except Exception as exc:
            # Batch failed because one or more IDs are deleted/inaccessible.
            # Fall back to fetching each ID individually so valid items are kept.
            log.warning("Batch %d failed (%s) — retrying individually…", batch_num, exc)
            yield "event", f"[Tests] Batch {batch_num} failed — retrying {len(batch)} items individually…"
            skipped = 0
            for wi_id_single in batch:
                try:
                    sr = requests.get(
                        f"{_API_BASE}/wit/workitems/{wi_id_single}",
                        headers=_HEADERS,
                        params={"$expand": "all", "api-version": _API_VERSION},
                    )
                    sr.raise_for_status()
                    work_items_all.append(sr.json())
                except Exception:
                    log.warning("Skipping inaccessible work item %s", wi_id_single)
                    skipped += 1
            if skipped:
                yield "event", f"[Tests] Skipped {skipped} inaccessible work item(s) in batch {batch_num}"

    # Collect all SharedStep work item IDs referenced across all test cases
    all_shared_ids: set[int] = set()
    for wi in work_items_all:
        steps_xml = wi.get("fields", {}).get("Microsoft.VSTS.TCM.Steps", "") or ""
        all_shared_ids |= _collect_shared_step_ids(steps_xml)

    # Batch-fetch shared step work items and parse their steps into a cache
    shared_steps_cache: dict[int, list[tuple[str, str]]] = {}
    if all_shared_ids:
        shared_id_list = list(all_shared_ids)
        yield "event", f"[Tests] Resolving {len(shared_id_list)} shared step(s)…"
        for i in range(0, len(shared_id_list), 200):
            batch = shared_id_list[i: i + 200]
            try:
                sr = requests.get(
                    f"{_API_BASE}/wit/workitems",
                    headers=_HEADERS,
                    params={"ids": ",".join(str(x) for x in batch),
                            "$expand": "all", "api-version": _API_VERSION},
                )
                sr.raise_for_status()
                for shared_wi in sr.json().get("value", []):
                    sid = shared_wi.get("id")
                    sx  = shared_wi.get("fields", {}).get("Microsoft.VSTS.TCM.Steps", "") or ""
                    if sid and sx:
                        shared_steps_cache[sid] = _parse_test_steps(sx)
            except Exception as exc:
                log.warning("Failed to fetch shared steps batch: %s", exc)

    # Fetch discussion threads in parallel (one request per test case)
    yield "event", f"[Tests] Fetching discussion threads for {len(work_items_all)} test case(s)…"
    comments_map: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_fetch_work_item_comments, wi.get("id")): wi.get("id")
                for wi in work_items_all if wi.get("id")}
        for fut in as_completed(futs):
            wid = futs[fut]
            try:
                comments_map[wid] = fut.result()
            except Exception:
                comments_map[wid] = []

    records: list[dict] = []
    for wi in work_items_all:
        wi_id                 = wi.get("id")
        plan_name, suite_name = tc_map.get(wi_id, ("", ""))
        md_content            = _format_test_case_markdown(
            wi, plan_name, suite_name,
            comments=comments_map.get(wi_id, []),
            shared_steps_cache=shared_steps_cache,
        )
        html_content = markdown.markdown(md_content, extensions=["extra", "tables"])
        records.append({
            "path":       f"/test-cases/{wi_id}",
            "remote_url": f"https://dev.azure.com/{ORG}/{PROJECT}/_workitems/edit/{wi_id}",
            "html":       html_content,
            "markdown":   md_content,
            "links":      [],
            "crawled_at": datetime.now(timezone.utc).isoformat(),
        })

    for r in records:
        yield "record", r


# ===========================================================================
# STEP 1D — Work Items crawl (Epics, Features, User Stories, Tasks, Bugs,
#            Change Requests)
# ===========================================================================

_WI_TYPES = ("Epic", "Feature", "User Story", "Task", "Bug", "Change Request")

# Fields fetched for every work item type
_WI_COMMON_FIELDS = [
    "System.Id", "System.Title", "System.WorkItemType", "System.State",
    "System.AreaPath", "System.IterationPath", "System.Reason",
    "System.AssignedTo", "System.CreatedBy", "System.CreatedDate",
    "System.ChangedBy", "System.ChangedDate", "System.Tags",
    "System.Description",
    # Epic / Feature / User Story status fields
    "Microsoft.VSTS.Common.StateChangeDate",
    "Microsoft.VSTS.CMMI.RequirementsRequestReason",
    # Collaboration / request-reason custom fields (may not exist on all instances)
    "Custom.CollaborationStatus",
    "Custom.RequestReason",
    "Custom.ResolvedReason",
    # Acceptance criteria (User Stories)
    "Microsoft.VSTS.Common.AcceptanceCriteria",
    # Bug-specific
    "Microsoft.VSTS.TCM.ReproSteps",
    "Microsoft.VSTS.Common.Priority",
    "Microsoft.VSTS.Common.Severity",
    # Details / description variants used in Epics/Features
    "Microsoft.VSTS.Common.ValueArea",
    "Microsoft.VSTS.Common.BusinessValue",
    "Microsoft.VSTS.Common.TimeCriticality",
    "Microsoft.VSTS.Scheduling.Effort",
    "Microsoft.VSTS.Scheduling.StoryPoints",
    # CMMI Change Request fields
    "Microsoft.VSTS.CMMI.Justification",
    "Microsoft.VSTS.CMMI.Analysis",
    "Microsoft.VSTS.CMMI.Symptoms",
    "Microsoft.VSTS.CMMI.FixedInChangesetId",
    "Microsoft.VSTS.CMMI.ImpactOnArchitecture",
    "Microsoft.VSTS.CMMI.ImpactOnDevelopment",
    "Microsoft.VSTS.CMMI.ImpactOnUserExperience",
    "Microsoft.VSTS.CMMI.ImpactOnTest",
    "Microsoft.VSTS.CMMI.ImpactOnTechnicalPublications",
]


def _format_work_item_markdown(
    wi: dict,
    comments: list[dict] | None = None,
    dev_info: dict | None = None,
    wi_lookup: dict[int, tuple[str, str]] | None = None,
    attachments: list[dict] | None = None,
) -> str:
    """Serialise a work item (Epic/Feature/User Story/Task/Bug/Change Request) to Markdown."""
    fields  = wi.get("fields", {})
    wi_id   = wi.get("id", "")
    wi_type = fields.get("System.WorkItemType", "Work Item")

    def txt(field_name: str) -> str:
        val = fields.get(field_name)
        if not val:
            return ""
        if isinstance(val, dict):
            return val.get("displayName") or val.get("uniqueName") or ""
        return str(val)

    def html_txt(field_name: str) -> str:
        raw = fields.get(field_name) or ""
        if not raw:
            return ""
        soup = BeautifulSoup(raw, "html.parser")
        # Insert space between table cells so columns don't merge
        for tag in soup.find_all(["td", "th"]):
            tag.insert_after(" ")
        # Insert newlines at block boundaries
        for tag in soup.find_all(["br"]):
            tag.replace_with("\n")
        for tag in soup.find_all(["tr", "p", "div", "li", "h1", "h2", "h3", "h4"]):
            tag.insert_after("\n")
        text = soup.get_text(separator="")
        # Strip non-breaking spaces / Unicode replacement chars that ADO inserts
        text = text.replace("\xa0", " ").replace("\ufffd", " ")
        # Join split "LabelWord\n: Value" → "LabelWord: Value"
        # Handles patterns like <b>Name</b>: value → "Name\n: value"
        text = re.sub(r"(\w[\w. ]*)\n[ \t]*:\s*", lambda m: m.group(1).rstrip() + ": ", text)
        # Normalise each line (collapse intra-line whitespace)
        lines = [" ".join(line.split()) for line in text.split("\n")]
        # After per-line normalisation, join "Short label:\n<value>" lines.
        # Handles ADO descriptions where Path:, Build ID#, Build ID#:, etc. have
        # their value on the next non-blank line (e.g. "Path:\n\\server\share").
        joined: list[str] = []
        for line in lines:
            last = joined[-1].rstrip() if joined else ""
            if (last and line
                    and len(last) <= 40
                    and (last.endswith(":") or last.endswith("#"))):
                joined[-1] = last + " " + line
            else:
                joined.append(line)
        lines = joined
        # Collapse consecutive blank lines to a single blank
        result: list[str] = []
        prev_blank = False
        for line in lines:
            is_blank = not line
            if is_blank and prev_blank:
                continue
            result.append(line)
            prev_blank = is_blank
        return "\n".join(result).strip()

    title       = txt("System.Title")
    state       = txt("System.State")
    area        = txt("System.AreaPath")
    iteration   = txt("System.IterationPath")
    reason      = txt("System.Reason")
    assigned_to = txt("System.AssignedTo")
    created_by  = txt("System.CreatedBy")
    created_dt  = txt("System.CreatedDate")[:10]
    changed_by  = txt("System.ChangedBy")
    changed_dt  = txt("System.ChangedDate")[:10]
    tags        = txt("System.Tags")
    priority    = txt("Microsoft.VSTS.Common.Priority")

    lines = [
        f"# {wi_type}: {title}",
        "",
        f"**ID:** {wi_id}",
        f"**Type:** {wi_type}",
        f"**State:** {state}",
        f"**Area Path:** {area}",
        f"**Iteration:** {iteration}",
    ]
    if reason:
        lines.append(f"**Reason:** {reason}")
    if priority:
        lines.append(f"**Priority:** {priority}")
    if assigned_to:
        lines.append(f"**Assigned To:** {assigned_to}")
    if created_by:
        lines.append(f"**Created By:** {created_by}" + (f" ({created_dt})" if created_dt else ""))
    if changed_by:
        lines.append(f"**Last Modified By:** {changed_by}" + (f" ({changed_dt})" if changed_dt else ""))
    if tags:
        lines.append(f"**Tags:** {tags}")

    # Type-specific status fields
    for label, field in [
        ("Request Reason",       "Microsoft.VSTS.CMMI.RequirementsRequestReason"),
        ("Request Reason",       "Custom.RequestReason"),
        ("Collaboration Status", "Custom.CollaborationStatus"),
        ("Resolved Reason",      "Custom.ResolvedReason"),
        ("Value Area",           "Microsoft.VSTS.Common.ValueArea"),
        ("Business Value",       "Microsoft.VSTS.Common.BusinessValue"),
        ("Time Criticality",     "Microsoft.VSTS.Common.TimeCriticality"),
        ("Effort",               "Microsoft.VSTS.Scheduling.Effort"),
        ("Story Points",         "Microsoft.VSTS.Scheduling.StoryPoints"),
        ("Severity",             "Microsoft.VSTS.Common.Severity"),
    ]:
        val = txt(field)
        if val:
            lines.append(f"**{label}:** {val}")

    # Description / Details
    description = html_txt("System.Description")
    if description:
        lines += ["", "## Description", "", description]

    # Acceptance Criteria (User Stories)
    acceptance = html_txt("Microsoft.VSTS.Common.AcceptanceCriteria")
    if acceptance:
        lines += ["", "## Acceptance Criteria", "", acceptance]

    # Bug repro steps
    repro = html_txt("Microsoft.VSTS.TCM.ReproSteps")
    if repro:
        lines += ["", "## Repro Steps", "", repro]

    # CMMI Change Request fields
    for label, field in [
        ("Justification",                    "Microsoft.VSTS.CMMI.Justification"),
        ("Analysis",                         "Microsoft.VSTS.CMMI.Analysis"),
        ("Symptoms",                         "Microsoft.VSTS.CMMI.Symptoms"),
        ("Fixed In Changeset",               "Microsoft.VSTS.CMMI.FixedInChangesetId"),
        ("Impact on Architecture",           "Microsoft.VSTS.CMMI.ImpactOnArchitecture"),
        ("Impact on Development",            "Microsoft.VSTS.CMMI.ImpactOnDevelopment"),
        ("Impact on User Experience",        "Microsoft.VSTS.CMMI.ImpactOnUserExperience"),
        ("Impact on Test",                   "Microsoft.VSTS.CMMI.ImpactOnTest"),
        ("Impact on Technical Publications", "Microsoft.VSTS.CMMI.ImpactOnTechnicalPublications"),
    ]:
        val = html_txt(field) or txt(field)
        if val:
            lines += ["", f"## {label}", "", val]

    # Linked work items (parent, children, related) — from $expand=all relations
    relations = wi.get("relations") or []
    if relations:
        # rel type → friendly section label.  ArtifactLink (git) is handled separately below.
        _CHILD_REL   = "System.LinkTypes.Hierarchy-Forward"
        _PARENT_REL  = "System.LinkTypes.Hierarchy-Reverse"
        _RELATED_REL = "System.LinkTypes.Related"

        def _wi_label(rel_url: str) -> str:
            """Extract WI ID from a REST URL and look up type + title."""
            try:
                ref_id = int(rel_url.rstrip("/").split("/")[-1])
            except (ValueError, TypeError):
                return ""
            if wi_lookup is not None:
                ref_type, ref_title = wi_lookup.get(ref_id, ("Work Item", ""))
                label = f"{ref_type} {ref_id}"
                if ref_title:
                    label += f" · {ref_title}"
            else:
                label = f"Work Item {ref_id}"
            return label

        children_labels = []
        parent_label    = ""
        related_labels  = []

        for rel in relations:
            rel_type = rel.get("rel", "")
            if rel_type not in (_CHILD_REL, _PARENT_REL, _RELATED_REL):
                continue
            label = _wi_label(rel.get("url", ""))
            if not label:
                continue
            if rel_type == _CHILD_REL:
                children_labels.append(label)
            elif rel_type == _PARENT_REL:
                parent_label = label
            elif rel_type == _RELATED_REL:
                related_labels.append(label)

        if parent_label or children_labels or related_labels:
            lines += ["", "## Linked Work Items", ""]
            if parent_label:
                lines.append(f"**Parent:** {parent_label}")
            if children_labels:
                lines.append(f"**Children ({len(children_labels)}):**")
                for lbl in children_labels:
                    lines.append(f"- {lbl}")
            if related_labels:
                lines.append(f"**Related ({len(related_labels)}):**")
                for lbl in related_labels:
                    lines.append(f"- {lbl}")

    # Discussion
    if comments:
        lines += ["", "## Discussion", ""]
        for c in comments:
            author = ""
            cb = c.get("createdBy")
            if isinstance(cb, dict):
                author = cb.get("displayName") or cb.get("uniqueName") or "Unknown"
            else:
                author = str(cb) if cb else "Unknown"
            date = (c.get("createdDate") or "")[:10]
            raw_text = c.get("text", "") or ""
            text = BeautifulSoup(raw_text, "html.parser").get_text("\n").strip()
            if text:
                lines += [f"**{author}** ({date}):", "", text, ""]

    # Git commits (Bugs and Tasks only)
    if dev_info:
        commits  = dev_info.get("commits") or []
        branches = dev_info.get("branches") or []

        if branches:
            lines += ["", "## Branches", ""]
            for b in branches:
                lines.append(f"- {b}")

        if commits:
            lines += ["", "## Git Commits", ""]
            for commit in commits:
                short_id = commit.get("commitId", "")
                message  = commit.get("message", "")
                author   = commit.get("author", "")
                date     = commit.get("date", "")
                changes  = commit.get("changes") or []

                header_parts = [f"**{short_id}**"]
                if message:
                    header_parts.append(f'"{message}"')
                header = " — ".join(header_parts)
                meta_parts = []
                if author:
                    meta_parts.append(f"**Author:** {author}")
                if date:
                    meta_parts.append(f"**Date:** {date}")

                lines.append(f"### {header}")
                full_sha = commit.get("commitIdFull", "")
                if full_sha:
                    lines.append(f"*sha:{full_sha}*")
                if meta_parts:
                    lines.append(" | ".join(meta_parts))
                if changes:
                    lines += ["",
                              "| Change Type | File |",
                              "|-------------|------|"]
                    for chg in changes:
                        ct   = chg.get("changeType", "edit")
                        path = chg.get("path", "")
                        lines.append(f"| {ct} | {path} |")
                    for chg in changes:
                        if chg.get("pkg_diff"):
                            fname = chg["path"].split("/")[-1]
                            lines += [
                                "",
                                f"**NuGet package changes in `{fname}`:**",
                                "",
                                chg["pkg_diff"],
                            ]
                lines.append("")

    # Attachments — rendered as a markdown table so the LLM can deduplicate
    # across work items by Date Added when consolidating multiple CRs.
    if attachments:
        lines += ["", "## Attachments", ""]
        lines.append("| File Name | Size | Date Added | Work Item |")
        lines.append("|-----------|------|------------|-----------|")
        for att in attachments:
            size_str  = f"{att['size'] // 1024} KB" if att.get("size") else "—"
            date_str  = att.get("added") or "—"
            name_cell = att["name"].replace("|", "\\|")
            lines.append(f"| {name_cell} | {size_str} | {date_str} | {wi_id} |")
        # Embed content of downloaded code/config attachments
        for att in attachments:
            if not att.get("content"):
                continue
            lang = att["ext"].lstrip(".")
            lines += [
                "",
                f"### Attachment: {att['name']}",
                "",
                f"```{lang}",
                att["content"][:3000],   # cap per file to stay within chunk budget
                "```",
            ]

    return "\n".join(lines)


def _fetch_file_at_commit(repo_id: str, file_path: str, commit_sha: str) -> str | None:
    """Return the text content of a file at a specific commit, or None if unavailable/too large."""
    try:
        r = requests.get(
            f"{_API_BASE}/git/repositories/{repo_id}/items",
            headers=_DL_HEADERS,   # no Content-Type so API returns raw file bytes
            params={
                "path": file_path,
                "version": commit_sha,
                "versionType": "commit",
                "$format": "text",
                "api-version": _API_VERSION,
            },
            timeout=15,
        )
        if not r.ok:
            log.debug("_fetch_file_at_commit %s@%s → HTTP %s", file_path, commit_sha[:7], r.status_code)
            return None
        if len(r.content) > COMMIT_DIFF_MAX_FILE_BYTES:
            log.debug("_fetch_file_at_commit %s@%s → skipped (%d bytes > limit)", file_path, commit_sha[:7], len(r.content))
            return None
        return r.text
    except Exception as exc:
        log.debug("_fetch_file_at_commit %s@%s → exception: %s", file_path, commit_sha[:7], exc)
    return None


def _build_commit_diff_record(
    repo_id: str,
    commit_id: str,       # full 40-char SHA
    parent_id: str,
    commit_meta: dict,    # {message, author, date, parent_count}
    file_changes: list[dict],
    wi_id: int,
) -> dict | None:
    """Build an index record containing the unified diff for every text file in a commit.

    Each diff is between the commit SHA and its first-parent SHA — both immutable git
    objects.  The diff is therefore identical regardless of when ingestion runs; running
    it today, tomorrow, or six months later always produces the same result for the same
    commit SHA.

    Stored at path /commit-diffs/{commitSha} so query.py can retrieve it by exact SHA.
    Returns None when no diffable files are found.
    """
    short        = commit_id[:7]
    parent_count = commit_meta.get("parent_count", 1)
    merge_note   = (
        "\n> **Merge commit** — diff shown against first parent only. "
        "Full feature-branch changes may span multiple commits."
        if parent_count > 1 else ""
    )

    lines: list[str] = [
        f"# Commit {short}",
        f"*sha:{commit_id}*",
        f"**Message:** {commit_meta.get('message', '')}",
        f"**Author:** {commit_meta.get('author', '')}  |  **Date:** {commit_meta.get('date', '')}",
        f"**Compared against parent:** {parent_id[:7] if parent_id else 'none (first commit)'}",
        f"**Linked Work Item:** {wi_id}",
    ]
    if merge_note:
        lines.append(merge_note)
    lines += ["", "## Changed Files", ""]

    diffable = 0
    for chg in file_changes[:COMMIT_DIFF_MAX_FILES]:
        fpath       = chg.get("path", "")
        change_type = chg.get("changeType", "edit")
        fname       = fpath.split("/")[-1]
        ext         = os.path.splitext(fname)[1].lower()

        lines.append(f"### `{fpath}` — {change_type}")
        lines.append("")

        if ext not in CODE_EXTENSIONS:
            lines.append("*Non-text file — diff not available.*")
            lines.append("")
            continue

        after  = _fetch_file_at_commit(repo_id, fpath, commit_id)  if change_type != "delete" else None
        before = _fetch_file_at_commit(repo_id, fpath, parent_id)  if parent_id and change_type != "add" else None

        if after is None and before is None:
            lines.append("*Content unavailable (file may be too large or access restricted).*")
            lines.append("")
            continue

        lang = ext.lstrip(".")
        if change_type == "add" or before is None:
            lines += [f"**New file:**", f"```{lang}", (after or "")[:2000], "```", ""]
        elif change_type == "delete" or after is None:
            lines += [f"**Deleted file:**", f"```{lang}", (before or "")[:1000], "```", ""]
        else:
            diff_lines = list(difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"{fname} (before)",
                tofile=f"{fname} (after)",
                n=4,
            ))
            if diff_lines:
                lines += ["```diff"] + [l.rstrip("\n") for l in diff_lines[:200]] + ["```", ""]
            else:
                lines.append("*No text changes detected (whitespace-only or identical).*")
                lines.append("")

        diffable += 1

    if diffable == 0:
        log.warning(
            "Commit diff %s (WI %s): no diffable files found — all %d changed files were "
            "non-text, too large, or inaccessible via API",
            commit_id[:7], wi_id, len(file_changes),
        )
        return None

    log.info("Commit diff %s (WI %s): built diff for %d file(s)", commit_id[:7], wi_id, diffable)
    md       = "\n".join(lines)
    html_out = markdown.markdown(md, extensions=["extra", "tables"])
    return {
        "path":       f"/commit-diffs/{commit_id}",
        "remote_url": f"https://dev.azure.com/{ORG}/{PROJECT}/_git/{repo_id}/commit/{commit_id}",
        "html":       html_out,
        "markdown":   md,
        "links":      [],
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }


def _fetch_csproj_package_diff(
    repo_id: str, file_path: str, commit_id: str, parent_id: str
) -> str | None:
    """Fetch a .csproj at two commits and return a markdown table of changed PackageReferences.

    Returns a markdown string (before/after table) or None if nothing changed or the file
    couldn't be fetched.  Only called when a commit changes a .csproj file.
    """

    def _get_content(version: str) -> str | None:
        try:
            r = requests.get(
                f"{_API_BASE}/git/repositories/{repo_id}/items",
                headers=_DL_HEADERS,   # no Content-Type for raw file download
                params={
                    "path": file_path,
                    "version": version,
                    "versionType": "commit",
                    "$format": "text",
                    "api-version": _API_VERSION,
                },
                timeout=15,
            )
            if r.ok:
                return r.text
        except Exception:
            pass
        return None

    def _parse_packages(xml_text: str) -> tuple[dict[str, str], str]:
        pkgs: dict[str, str] = {}
        framework = ""
        try:
            root = ET.fromstring(xml_text)
            for elem in root.iter("PackageReference"):
                name = elem.get("Include") or elem.get("include", "")
                ver  = elem.get("Version") or elem.get("version", "")
                if name:
                    pkgs[name] = ver or ""
            for tag in ("TargetFramework", "TargetFrameworks"):
                for elem in root.iter(tag):
                    framework = (elem.text or "").strip()
                    break
                if framework:
                    break
        except ET.ParseError:
            pass
        return pkgs, framework

    after_text  = _get_content(commit_id)
    if not after_text:
        return None
    before_text = _get_content(parent_id) if parent_id else None

    after_pkgs,  after_fw  = _parse_packages(after_text)
    before_pkgs, before_fw = _parse_packages(before_text) if before_text else ({}, "")

    rows: list[str] = []
    if before_fw != after_fw and (before_fw or after_fw):
        rows.append(f"| TargetFramework | `{before_fw or '—'}` | `{after_fw or '—'}` |")

    for name in sorted(set(before_pkgs) | set(after_pkgs)):
        bv = before_pkgs.get(name)
        av = after_pkgs.get(name)
        if bv == av:
            continue
        if bv is None:
            rows.append(f"| {name} | *(new)* | `{av}` |")
        elif av is None:
            rows.append(f"| {name} | `{bv}` | *(removed)* |")
        else:
            rows.append(f"| {name} | `{bv}` | `{av}` |")

    if not rows:
        return None
    return "\n".join(
        ["| Package | Before | After |", "|---------|--------|-------|"] + rows
    )


def _fetch_wi_dev_info(wi_id: int, relations: list[dict]) -> dict:
    """Resolve git commit and branch details from ArtifactLink relations.

    Called for Bug and Task work items only.  Always re-fetches the work item
    individually to guarantee ArtifactLinks are present — the batch
    /wit/workitems?ids=...&$expand=all endpoint silently omits ArtifactLinks
    from the relations array even when $expand=all is specified.

    Returns:
        {
            "commits": [{"commitId": "a1b2c3d", "message": "...", "author": "...",
                          "date": "YYYY-MM-DD", "changes": [{"path": ..., "changeType": ...}]},
                         ...],
            "branches": ["feature/my-branch", ...],
        }
    """
    # Fetch the WI individually with $expand=all — the batch API silently omits
    # ArtifactLinks even with $expand=all; only the individual endpoint is reliable.
    try:
        r = requests.get(
            f"{_API_BASE}/wit/workitems/{wi_id}",
            headers=_HEADERS,
            params={"$expand": "all", "api-version": _API_VERSION},
            timeout=15,
        )
        if r.ok:
            relations = r.json().get("relations") or []
            artifact_count = sum(1 for rel in relations if rel.get("rel") == "ArtifactLink")
            log.info("_fetch_wi_dev_info WI %s: %d relations, %d ArtifactLink(s)", wi_id, len(relations), artifact_count)
        else:
            log.warning("_fetch_wi_dev_info WI %s: individual fetch returned HTTP %s — using batch relations", wi_id, r.status_code)
    except Exception as exc:
        log.warning("_fetch_wi_dev_info WI %s: individual fetch failed (%s) — using batch relations", wi_id, exc)

    commits: list[dict]  = []
    branches: list[str]  = []
    seen_commits: set[str] = set()
    seen_branches: set[str] = set()

    for rel in (relations or []):
        if rel.get("rel") != "ArtifactLink":
            continue
        # Strip whitespace / null bytes then URL-decode (%2f → /) before matching.
        # Azure DevOps encodes the vstfs URL as:
        #   vstfs:///Git/Commit/{projectId}%2f{repoId}%2f{sha}
        # which after unquoting becomes the 3-segment form:
        #   vstfs:///Git/Commit/{projectId}/{repoId}/{sha}
        url = rel.get("url", "").strip().rstrip("\x00")
        url = urllib.parse.unquote(url)

        # ── Git Commit ─────────────────────────────────────────────────────────
        # Handles both 3-segment (projectId/repoId/sha) and legacy 2-segment (repoId/sha).
        # The optional (?:[^/]+/) prefix absorbs the projectId when present.
        commit_match = re.match(
            r"vstfs:///Git/Commit/(?:[^/]+/)?([^/]+)/([a-fA-F0-9]{40})",
            url, re.IGNORECASE
        )
        if commit_match:
            repo_id   = commit_match.group(1)
            commit_id = commit_match.group(2).lower()
            if commit_id not in seen_commits:
                seen_commits.add(commit_id)
                try:
                    cr = requests.get(
                        f"{_API_BASE}/git/repositories/{repo_id}/commits/{commit_id}",
                        headers=_HEADERS,
                        params={"api-version": _API_VERSION},
                    )
                    chg_r = requests.get(
                        f"{_API_BASE}/git/repositories/{repo_id}/commits/{commit_id}/changes",
                        headers=_HEADERS,
                        params={"api-version": _API_VERSION},
                    )
                    file_changes: list[dict] = []
                    if chg_r.ok:
                        for chg in chg_r.json().get("changes", []):
                            item        = chg.get("item", {})
                            change_type = chg.get("changeType", "edit")
                            item_path   = item.get("path", "")
                            if item_path and not item.get("isFolder"):
                                file_changes.append({"path": item_path, "changeType": change_type})
                    if cr.ok:
                        c = cr.json()
                        parents = c.get("parents") or []
                        # Azure DevOps returns parents as either plain SHA strings
                        # or dicts {"objectId": "sha..."} depending on API version.
                        # Normalise to a plain SHA string.
                        def _parent_sha(p) -> str:
                            if isinstance(p, dict):
                                return p.get("objectId", "") or p.get("commitId", "")
                            return str(p) if p else ""
                        parent_id    = _parent_sha(parents[0]) if parents else ""
                        parent_count = len(parents)
                        # For .csproj files, fetch before/after package diff
                        for fc in file_changes:
                            if fc["path"].lower().endswith(".csproj"):
                                diff = _fetch_csproj_package_diff(
                                    repo_id, fc["path"], commit_id, parent_id
                                )
                                if diff:
                                    fc["pkg_diff"] = diff
                        # Only take the first line of the commit message
                        first_line = (c.get("comment") or "").strip().splitlines()
                        commits.append({
                            "commitId":     commit_id[:7],
                            "commitIdFull": commit_id,
                            "repoId":       repo_id,
                            "parentId":     parent_id,
                            "parentCount":  parent_count,
                            "message":      first_line[0] if first_line else "",
                            "author":       (c.get("author") or {}).get("name", ""),
                            "date":         ((c.get("author") or {}).get("date") or "")[:10],
                            "changes":      file_changes,
                        })
                except Exception as exc:
                    log.warning("Could not fetch commit %s for WI %s: %s", commit_id[:7], wi_id, exc)
            continue

        # ── Git Branch (Ref) ───────────────────────────────────────────────────
        # vstfs:///Git/Ref/{projectId}/{repoId}/{hexEncodedRefPath}  (3-segment)
        ref_match = re.match(
            r"vstfs:///Git/Ref/(?:[^/]+/)?([^/]+)/([a-fA-F0-9]+)", url, re.IGNORECASE
        )
        if ref_match:
            hex_ref = ref_match.group(2)
            try:
                ref_path    = bytes.fromhex(hex_ref).decode("utf-8", errors="replace")
                branch_name = re.sub(r"^refs/heads/", "", ref_path).strip("\x00")
                if branch_name and branch_name not in seen_branches:
                    seen_branches.add(branch_name)
                    branches.append(branch_name)
            except Exception:
                pass
            continue

        # ── Known link types we intentionally skip (Build, Release, TFVC) ────
        # vstfs:///Build/Build/NNN        — linked CI/CD build run
        # vstfs:///ReleaseManagement/...  — release pipeline link
        # vstfs:///VersionControl/...     — TFVC changeset link
        if re.match(r"vstfs:///(Build|ReleaseManagement|VersionControl)/", url, re.IGNORECASE):
            log.debug("ArtifactLink skipped (non-Git) for WI %s — %r", wi_id, url)
            continue

        # ── Truly unrecognised — log at DEBUG to avoid terminal clutter ───────
        log.debug("ArtifactLink not recognised for WI %s — decoded URL: %r", wi_id, url)

    return {"commits": commits, "branches": branches}


def _crawl_work_items(selected_area_paths: list[str] | None = None):
    """
    Generator yielding ("event", message_str) or ("record", dict).
    Crawls Epics, Features, User Stories, Tasks and Bugs from the project.
    If selected_area_paths is provided, only items under those area paths are fetched.
    Discussion threads are fetched in parallel.
    """
    area_clause = ""
    if selected_area_paths:
        area_parts = " OR ".join(
            f"[System.AreaPath] UNDER '{p}'" for p in selected_area_paths
        )
        area_clause = f" AND ({area_parts})"

    # Run one WIQL query per work item type to stay well under the 20 000-item
    # limit enforced by Azure DevOps.  A single broad query across all types and
    # a large area tree easily exceeds that limit and returns HTTP 400.
    yield "event", "[WorkItems] Running per-type WIQL queries…"
    wi_ids: list[int] = []
    type_counts: dict[str, int] = {}
    for wi_type in _WI_TYPES:
        wiql = {
            "query": (
                f"SELECT [System.Id] FROM WorkItems "
                f"WHERE [System.WorkItemType] = '{wi_type}'{area_clause} "
                f"ORDER BY [System.Id]"
            )
        }
        try:
            r = requests.post(
                f"{_API_BASE}/wit/wiql?api-version={_API_VERSION}&$top=20000",
                headers=_HEADERS,
                json=wiql,
            )
            if not r.ok:
                try:
                    detail = r.json().get("message") or r.text[:500]
                except Exception:
                    detail = r.text[:500]
                yield "event", f"[WorkItems] WIQL failed for {wi_type} ({r.status_code}): {detail}"
                continue
            refs = r.json().get("workItems", [])
            type_counts[wi_type] = len(refs)
            wi_ids.extend(ref["id"] for ref in refs)
            yield "event", f"[WorkItems] {wi_type}: {len(refs)} item(s)"
        except Exception as exc:
            yield "event", f"[WorkItems] WIQL failed for {wi_type}: {exc}"

    # Deduplicate (a work item should only appear in one type, but be safe)
    seen_ids: set[int] = set()
    unique_ids: list[int] = []
    for wid in wi_ids:
        if wid not in seen_ids:
            seen_ids.add(wid)
            unique_ids.append(wid)
    wi_ids = unique_ids

    if not wi_ids:
        yield "event", "[WorkItems] No work items found for selected areas."
        return

    summary_parts = ", ".join(f"{t}: {type_counts.get(t, 0)}" for t in _WI_TYPES)
    yield "event", f"[WorkItems] Total {len(wi_ids)} item(s) — {summary_parts}"

    # Use $expand=all (same as test-case crawl) — returns whatever fields exist on
    # each item without failing if a field name in an explicit list is absent on this instance.
    work_items_all: list[dict] = []
    n_batches = max(1, (len(wi_ids) + 199) // 200)
    for i in range(0, len(wi_ids), 200):
        batch_num = i // 200 + 1
        batch = wi_ids[i: i + 200]
        yield "event", f"[WorkItems] Fetching batch {batch_num}/{n_batches}…"
        try:
            wr = requests.get(
                f"{_API_BASE}/wit/workitems",
                headers=_HEADERS,
                params={
                    "ids": ",".join(str(x) for x in batch),
                    "$expand": "all",
                    "api-version": _API_VERSION,
                },
            )
            wr.raise_for_status()
            work_items_all.extend(wr.json().get("value", []))
        except Exception as exc:
            log.warning("WorkItems batch %d failed: %s — retrying individually", batch_num, exc)
            yield "event", f"[WorkItems] Batch {batch_num} failed — retrying {len(batch)} individually…"
            skipped = 0
            for wid in batch:
                try:
                    sr = requests.get(
                        f"{_API_BASE}/wit/workitems/{wid}",
                        headers=_HEADERS,
                        params={"$expand": "all", "api-version": _API_VERSION},
                    )
                    sr.raise_for_status()
                    work_items_all.append(sr.json())
                except Exception:
                    skipped += 1
            if skipped:
                yield "event", f"[WorkItems] Skipped {skipped} inaccessible item(s) in batch {batch_num}"

    # For Bugs and Tasks: resolve git commits + branches from relations in parallel
    _DEV_TYPES = {"Bug", "Task"}
    dev_wi_ids = [
        wi.get("id") for wi in work_items_all
        if wi.get("fields", {}).get("System.WorkItemType") in _DEV_TYPES
    ]
    if dev_wi_ids:
        yield "event", f"[WorkItems] Fetching dev links (commits/branches) for {len(dev_wi_ids)} Bug/Task item(s)…"
    dev_info_map: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {
            pool.submit(_fetch_wi_dev_info, wi.get("id"), wi.get("relations") or []): wi.get("id")
            for wi in work_items_all
            if wi.get("fields", {}).get("System.WorkItemType") in _DEV_TYPES and wi.get("id")
        }
        for fut in as_completed(futs):
            wid = futs[fut]
            try:
                dev_info_map[wid] = fut.result()
            except Exception:
                dev_info_map[wid] = {"commits": [], "branches": []}

    # Generate commit diff records — one per unique commit SHA, across all work items
    if CRAWL_COMMIT_DIFFS:
        seen_diff_shas: set[str] = set()
        diff_pairs: list[tuple[int, dict]] = []   # (wi_id, commit_dict)
        for wi in work_items_all:
            wid = wi.get("id")
            for commit in dev_info_map.get(wid, {}).get("commits", []):
                full_sha = commit.get("commitIdFull", "")
                if full_sha and full_sha not in seen_diff_shas:
                    seen_diff_shas.add(full_sha)
                    diff_pairs.append((wid, commit))
        total_commits = sum(len(dev_info_map.get(wi.get("id"), {}).get("commits", [])) for wi in work_items_all)
        log.info("[WorkItems] dev_info_map: %d WI(s) with commits, %d unique commit(s) to diff", total_commits, len(diff_pairs))
        if diff_pairs:
            yield "event", f"[WorkItems] Building commit diffs for {len(diff_pairs)} unique commit(s)…"
            for wi_id, commit in diff_pairs:
                rec = _build_commit_diff_record(
                    repo_id=commit["repoId"],
                    commit_id=commit["commitIdFull"],
                    parent_id=commit.get("parentId", ""),
                    commit_meta={
                        "message":      commit.get("message", ""),
                        "author":       commit.get("author", ""),
                        "date":         commit.get("date", ""),
                        "parent_count": commit.get("parentCount", 1),
                    },
                    file_changes=commit.get("changes", []),
                    wi_id=wi_id,
                )
                if rec:
                    yield "record", rec

    # Fetch discussion threads in parallel
    yield "event", f"[WorkItems] Fetching discussion threads for {len(work_items_all)} item(s)…"
    comments_map: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_fetch_work_item_comments, wi.get("id")): wi.get("id")
                for wi in work_items_all if wi.get("id")}
        for fut in as_completed(futs):
            wid = futs[fut]
            try:
                comments_map[wid] = fut.result()
            except Exception:
                comments_map[wid] = []

    # Fetch attachments in parallel (code/config files are downloaded and embedded)
    attachments_map: dict[int, list[dict]] = {}
    if CRAWL_WI_ATTACHMENTS:
        wi_with_attachments = [
            wi for wi in work_items_all
            if wi.get("id") and any(
                r.get("rel") == "AttachedFile" for r in (wi.get("relations") or [])
            )
        ]
        if wi_with_attachments:
            yield "event", f"[WorkItems] Fetching attachments for {len(wi_with_attachments)} item(s)…"
            with ThreadPoolExecutor(max_workers=6) as pool:
                futs = {
                    pool.submit(_fetch_work_item_attachments, wi.get("id"), wi.get("relations") or []): wi.get("id")
                    for wi in wi_with_attachments
                }
                for fut in as_completed(futs):
                    wid = futs[fut]
                    try:
                        attachments_map[wid] = fut.result()
                    except Exception:
                        attachments_map[wid] = []

    # Build ID → (type, title) lookup so each work item's markdown can reference
    # its linked children/parent by name instead of just bare ID numbers.
    wi_lookup: dict[int, tuple[str, str]] = {
        w.get("id"): (
            w.get("fields", {}).get("System.WorkItemType", "Work Item"),
            w.get("fields", {}).get("System.Title", ""),
        )
        for w in work_items_all
        if w.get("id")
    }

    total_wi = len(work_items_all)
    for wi_idx, wi in enumerate(work_items_all, 1):
        wi_id    = wi.get("id")
        wi_type  = wi.get("fields", {}).get("System.WorkItemType", "Work Item")
        wi_title = wi.get("fields", {}).get("System.Title", "")
        yield "event", f"[WorkItems] ({wi_idx}/{total_wi}) {wi_type} {wi_id} · {wi_title}"
        md       = _format_work_item_markdown(
            wi,
            comments=comments_map.get(wi_id, []),
            dev_info=dev_info_map.get(wi_id),
            wi_lookup=wi_lookup,
            attachments=attachments_map.get(wi_id) or None,
        )
        html_out = markdown.markdown(md, extensions=["extra", "tables"])
        yield "record", {
            "path":       f"/work-items/{wi_id}",
            "remote_url": f"https://dev.azure.com/{ORG}/{PROJECT}/_workitems/edit/{wi_id}",
            "html":       html_out,
            "markdown":   md,
            "links":      [],
            "crawled_at": datetime.now(timezone.utc).isoformat(),
        }

    yield "event", f"[WorkItems] Done — {total_wi} work item(s) crawled"


# ===========================================================================
# Blob storage helpers
# ===========================================================================

def _save_snapshot(records: list[dict], blob_name: str) -> None:
    client           = BlobServiceClient.from_connection_string(STORAGE_CONN)
    container_client = client.get_container_client(CONTAINER)
    try:
        container_client.create_container()
    except Exception:
        pass
    jsonl_bytes = "\n".join(json.dumps(r, ensure_ascii=False) for r in records).encode("utf-8")
    container_client.get_blob_client(blob_name).upload_blob(jsonl_bytes, overwrite=True)


def _load_snapshot(blob_name: str) -> list[dict]:
    """Load previously crawled records from blob storage."""
    client = BlobServiceClient.from_connection_string(STORAGE_CONN)
    blob   = client.get_blob_client(container=CONTAINER, blob=blob_name)
    try:
        data = blob.download_blob().readall().decode("utf-8")
        return [json.loads(line) for line in data.splitlines() if line.strip()]
    except ResourceNotFoundError:
        return []


# ===========================================================================
# Hash manifest — change detection between runs
# ===========================================================================

def load_hash_manifest() -> dict[str, str]:
    client = BlobServiceClient.from_connection_string(STORAGE_CONN)
    blob   = client.get_blob_client(container=CONTAINER, blob=HASH_MANIFEST_BLOB)
    try:
        return json.loads(blob.download_blob().readall().decode("utf-8"))
    except ResourceNotFoundError:
        return {}


def save_hash_manifest(manifest: dict[str, str]) -> None:
    client = BlobServiceClient.from_connection_string(STORAGE_CONN)
    blob   = client.get_blob_client(container=CONTAINER, blob=HASH_MANIFEST_BLOB)
    blob.upload_blob(json.dumps(manifest, indent=2).encode("utf-8"), overwrite=True)


def page_hash(record: dict) -> str:
    content = record.get("markdown", "") or record.get("html", "")
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def filter_changed_records(
    records: list[dict], manifest: dict[str, str]
) -> tuple[list[dict], dict[str, str]]:
    changed      = []
    new_manifest = {}
    for record in records:
        path = record.get("path", "/")
        h    = page_hash(record)
        new_manifest[path] = h
        if manifest.get(path) != h:
            changed.append(record)
    return changed, new_manifest


# ===========================================================================
# STEP 2 — Chunk
# ===========================================================================

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(["p", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr"]):
        tag.insert_before("\n")
    return " ".join(soup.get_text(separator=" ").split())


def _path_source_type(path: str) -> tuple[str, list[str]]:
    """Return (source_type string, source_tags list) for a given path."""
    if path.startswith("/repos/"):
        return "code", ["code"]
    if path.startswith("/test-cases/"):
        return "test", ["test"]
    if path.startswith("/work-items/"):
        return "workitem", ["workitem"]
    if path.startswith("/commit-diffs/"):
        return "commit-diff", ["commit-diff"]
    # Covers both new-style /wiki/<WikiName>/... and any legacy bare /PageName paths
    return "wiki", ["wiki"]


def _tc_identity_prefix(text: str) -> str:
    """Extract a short identity header from a test case markdown document.

    Returns a prefix like:
        Test Case 296199 · isMarketPlaceClient set to true…
    so that every chunk produced from a split test case document still
    contains the TC ID and title and remains discoverable by ID.
    """
    wi_id = title = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("**ID:**"):
            wi_id = line.replace("**ID:**", "").strip()
        elif line.startswith("# Test Case:"):
            title = line.replace("# Test Case:", "").strip()
        if wi_id and title:
            break
    if wi_id:
        label = f"Test Case {wi_id}"
        if title:
            label += f" · {title}"
        return label + "\n\n"
    return ""


def chunk_records(records: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
    )
    chunks: list[dict] = []
    for record in records:
        path        = record.get("path", "/")
        url         = record.get("remote_url", "")
        crawled_at  = record.get("crawled_at", "")
        source_type, source_tags = _path_source_type(path)
        text        = record.get("markdown", "").strip() or html_to_text(record.get("html", ""))
        if not text:
            continue

        # For test cases: build a short identity prefix to prepend to every
        # non-first chunk so all chunks remain queryable by TC ID / title.
        is_test_case  = path.startswith("/test-cases/")
        tc_prefix     = _tc_identity_prefix(text) if is_test_case else ""

        lc_docs = splitter.create_documents(
            texts=[text],
            metadatas=[{"path": path, "url": url, "crawled_at": crawled_at}],
        )
        for idx, doc in enumerate(lc_docs):
            safe_id   = re.sub(r"[^a-zA-Z0-9_\-=]", "_", path.strip("/").replace("/", "__")) or "root"
            chunk_text = (tc_prefix + doc.page_content) if (is_test_case and idx > 0) else doc.page_content
            chunks.append({
                "id":          f"{safe_id}__{idx}",
                "text":        chunk_text,
                "path":        path,
                "url":         url,
                "crawled_at":  crawled_at,
                "source_type": source_type,
                "source_tags": source_tags,
            })
    return chunks


# ===========================================================================
# STEP 3 — Embed
# ===========================================================================

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed all chunks in batches. Returns chunks with 'embedding' field populated."""
    model = AzureOpenAIEmbeddings(
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        azure_deployment=AOAI_EMBEDDING_DEPLOYMENT,
        openai_api_version=AOAI_API_VERSION,
        max_retries=0,  # disable SDK retries — our caller handles back-off
    )
    texts       = [c["text"] for c in chunks]
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        all_vectors.extend(model.embed_documents(texts[i: i + EMBED_BATCH_SIZE]))
    for chunk, vector in zip(chunks, all_vectors):
        chunk["embedding"] = vector
    return chunks


# ===========================================================================
# STEP 4 — Index
# ===========================================================================

def ensure_search_index() -> bool:
    """Create or update the Azure AI Search index.

    Adds semantic configuration, tag-based scoring profile, and source-type fields.
    Returns True when the index is newly created, False when an existing index was updated.
    """
    index_client = SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )
    existing = [idx.name for idx in index_client.list_indexes()]
    is_new   = SEARCH_INDEX_NAME not in existing

    index = SearchIndex(
        name=SEARCH_INDEX_NAME,
        fields=[
            SimpleField(name="id",          type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="text",    type=SearchFieldDataType.String),
            SimpleField(name="path",        type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="url",         type=SearchFieldDataType.String),
            SimpleField(name="crawled_at",  type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(
                name="source_tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=VECTOR_DIM,
                vector_search_profile_name="hnsw-profile",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
            profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-config")],
        ),
        semantic_search=SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="tigerchat-semantic",
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=[SemanticField(field_name="text")],
                        keywords_fields=[SemanticField(field_name="path")],
                    ),
                )
            ]
        ),
        scoring_profiles=[
            ScoringProfile(
                name="source-boost",
                function_aggregation="sum",
                functions=[
                    TagScoringFunction(
                        field_name="source_tags",
                        boost=3.0,
                        parameters=TagScoringParameters(tags_parameter="boostTags"),
                        interpolation="linear",
                    )
                ],
            )
        ],
    )

    if is_new:
        index_client.create_index(index)
        return True

    # Existing index — try full update first, then fall back if the tier
    # doesn't support semantic search or scoring profiles.
    try:
        index_client.create_or_update_index(index)
    except Exception as e:
        err = str(e)
        if "semantic" in err.lower() or "scoringProfile" in err.lower() or "ScoringProfile" in err:
            log.warning("Semantic/scoring profile not supported by this tier — updating index without them.")
            index.semantic_search = None
            index.scoring_profiles = []
            index_client.create_or_update_index(index)
        else:
            raise
    return False


# ===========================================================================
# STEP 5 — Upload
# ===========================================================================

def upload_to_search(chunks: list[dict]) -> int:
    """Upload chunks to Azure AI Search in batches. Returns count of failed documents."""
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )
    failed = 0
    for i in range(0, len(chunks), UPLOAD_BATCH_SIZE):
        results = search_client.upload_documents(documents=chunks[i: i + UPLOAD_BATCH_SIZE])
        failed += sum(1 for r in results if not r.succeeded)
    return failed


# ===========================================================================
# Pipeline generator — used by the web server for SSE streaming
# ===========================================================================

def run_pipeline(
    crawl: bool = True,
    crawl_wiki: bool | None = None,
    crawl_code: bool | None = None,
    crawl_tests: bool | None = None,
    crawl_work_items: bool | None = None,
    selected_repos: list[str] | None = None,
    selected_plan_ids: list[int] | None = None,
    selected_suite_ids: list[int] | None = None,
    selected_tc_ids: list[int] | None = None,
    selected_area_paths: list[str] | None = None,
) -> Generator[dict, None, None]:
    """
    Runs the full pipeline, yielding SSE-ready event dicts:
      { "step": str, "status": "active"|"done"|"error", "message": str }
    The final upload event includes a "summary" key with run stats.

    crawl=False: skip crawling and re-process the existing snapshot from blob storage.
    crawl_wiki/code/tests: override the CRAWL_* env vars when crawl=True.
    """
    # Resolve per-source toggles — UI params take priority over env vars
    do_wiki       = crawl_wiki       if crawl_wiki       is not None else CRAWL_WIKI
    do_code       = crawl_code       if crawl_code       is not None else CRAWL_CODE
    do_tests      = crawl_tests      if crawl_tests      is not None else CRAWL_TESTS
    do_work_items = crawl_work_items if crawl_work_items is not None else CRAWL_WORK_ITEMS

    blob_name = f"rag_{ORG}_{PROJECT}.jsonl"
    summary   = {}
    force_reprocess = False

    def event(step, status, message, **extra):
        d = {"step": step, "status": status, "message": message}
        d.update(extra)
        return d

    # ── Step 1: Crawl ─────────────────────────────────────────────────────────
    if not crawl:
        # Skip crawl — load the last snapshot from blob and force re-processing
        yield event("crawl", "active", "Crawl skipped — loading existing snapshot from blob storage…")
        try:
            all_records = _load_snapshot(blob_name)
        except Exception as exc:
            yield event("crawl", "error", f"Could not load snapshot: {exc}")
            return
        if not all_records:
            yield event("crawl", "error",
                        "No snapshot found in blob storage — run with crawling enabled first.")
            return
        force_reprocess = True
        # Count records by path prefix so the summary footer shows real numbers
        _wiki_count = sum(1 for r in all_records if r.get("path", "").startswith("/wiki/"))
        _code_count = sum(1 for r in all_records if r.get("path", "").startswith("/repos/"))
        _test_count = sum(1 for r in all_records if r.get("path", "").startswith("/test-cases/"))
        _wi_count   = sum(1 for r in all_records if r.get("path", "").startswith("/work-items/"))
        summary.update({
            "pages":      len(all_records),
            "wiki_count": _wiki_count,
            "code_count": _code_count,
            "test_count": _test_count,
            "wi_count":   _wi_count,
            "blob":       blob_name,
        })
        yield event("crawl", "done",
                    f"Loaded {len(all_records)} records from existing snapshot (crawl skipped) "
                    f"(wiki: {_wiki_count}, code: {_code_count}, "
                    f"tests: {_test_count}, work items: {_wi_count})")
    else:
        try:
            all_records: list[dict] = []
            wiki_count = code_count = test_count = wi_count = 0

            # ── Wiki ──
            if do_wiki:
                yield event("crawl", "active", f"[Wiki] Discovering all wikis in {ORG}/{PROJECT}…")
                try:
                    all_wikis = _list_all_wikis()
                except Exception as exc:
                    all_wikis = []
                    yield event("crawl", "active", f"[Wiki] Could not list wikis: {exc} — skipping")

                if not all_wikis:
                    yield event("crawl", "active", "[Wiki] No wikis found.")
                else:
                    wiki_names = ", ".join(w.get("name", "?") for w in all_wikis)
                    yield event("crawl", "active",
                                f"[Wiki] Found {len(all_wikis)} wiki(s): {wiki_names}")

                    for wiki_meta in all_wikis:
                        wiki_name = wiki_meta.get("name", "")
                        wiki_base = (
                            f"https://dev.azure.com/{ORG}/{PROJECT}"
                            f"/_apis/wiki/wikis/{wiki_name}"
                        )
                        yield event("crawl", "active",
                                    f"[Wiki] '{wiki_name}' — listing pages…")
                        try:
                            pages = _list_all_wiki_pages(wiki_base)
                        except Exception as exc:
                            log.warning("Failed to list pages for wiki %s: %s", wiki_name, exc)
                            yield event("crawl", "active",
                                        f"[Wiki] '{wiki_name}' — failed to list pages: {exc}")
                            continue

                        yield event("crawl", "active",
                                    f"[Wiki] '{wiki_name}' — {len(pages)} page(s) — fetching…")
                        failed_pages: list[str] = []
                        for i, page in enumerate(pages, 1):
                            try:
                                all_records.append(
                                    _fetch_wiki_page(page["path"], wiki_base, wiki_name)
                                )
                                wiki_count += 1
                            except Exception as exc:
                                log.warning("Wiki page fetch failed %s/%s: %s",
                                            wiki_name, page["path"], exc)
                                failed_pages.append(page["path"])
                            if i % 20 == 0 or i == len(pages):
                                yield event("crawl", "active",
                                            f"[Wiki] '{wiki_name}': {i}/{len(pages)} fetched…")
                        msg = f"[Wiki] '{wiki_name}' done — {wiki_count} page(s) total so far"
                        if failed_pages:
                            msg += f" ({len(failed_pages)} failed)"
                        yield event("crawl", "active", msg)

            # ── Code ──
            if do_code:
                yield event("crawl", "active", "[Code] Starting repository crawl…")
                try:
                    for kind, value in _crawl_code_files(selected_repos=selected_repos):
                        if kind == "event":
                            yield event("crawl", "active", value)
                        else:
                            all_records.append(value)
                            code_count += 1
                except Exception as exc:
                    log.warning("Code crawl error: %s", exc)
                    yield event("crawl", "active", f"[Code] Warning: {exc} — continuing")
                yield event("crawl", "active", f"[Code] Done — {code_count} file(s)")

            # ── Tests ──
            if do_tests:
                yield event("crawl", "active", "[Tests] Starting test management crawl…")
                plan_set  = set(selected_plan_ids)  if selected_plan_ids  else None
                suite_set = set(selected_suite_ids) if selected_suite_ids else None
                tc_set    = set(selected_tc_ids)    if selected_tc_ids    else None
                try:
                    for kind, value in _crawl_test_cases(plan_set, suite_set, tc_set):
                        if kind == "event":
                            yield event("crawl", "active", value)
                        else:
                            all_records.append(value)
                            test_count += 1
                except Exception as exc:
                    log.warning("Test crawl error: %s", exc)
                    yield event("crawl", "active", f"[Tests] Warning: {exc} — continuing")
                yield event("crawl", "active", f"[Tests] Done — {test_count} test case(s)")

            # ── Work Items ──
            if do_work_items:
                yield event("crawl", "active", "[WorkItems] Starting work items crawl…")
                try:
                    for kind, value in _crawl_work_items(selected_area_paths=selected_area_paths):
                        if kind == "event":
                            yield event("crawl", "active", value)
                        else:
                            all_records.append(value)
                            wi_count += 1
                except Exception as exc:
                    log.warning("WorkItems crawl error: %s", exc)
                    yield event("crawl", "active", f"[WorkItems] Warning: {exc} — continuing")
                yield event("crawl", "active", f"[WorkItems] Done — {wi_count} work item(s)")

            # Save combined snapshot
            _save_snapshot(all_records, blob_name)

            summary.update({
                "pages":      len(all_records),
                "wiki_count": wiki_count,
                "code_count": code_count,
                "test_count": test_count,
                "wi_count":   wi_count,
                "blob":       blob_name,
            })
            yield event("crawl", "done",
                        f"Crawled {len(all_records)} total records "
                        f"(wiki: {wiki_count}, code: {code_count}, tests: {test_count}, work items: {wi_count})")

        except Exception as exc:
            yield event("crawl", "error", str(exc))
            return

    records = all_records

    # ── Step 2: Chunk (with optional change detection) ────────────────────────
    try:
        if force_reprocess:
            # Crawl was skipped — re-process all records regardless of hashes
            yield event("chunk", "active",
                        f"Re-processing {len(records)} records from snapshot (bypassing change detection)…")
            changed_records = records
            new_manifest    = {r["path"]: page_hash(r) for r in records}
            skipped = 0
        else:
            yield event("chunk", "active", "Loading hash manifest…")
            manifest = load_hash_manifest()
            changed_records, new_manifest = filter_changed_records(records, manifest)
            skipped = len(records) - len(changed_records)

        summary["skipped"] = skipped

        if not changed_records:
            summary.update({"chunks": 0, "uploaded": 0, "failed": 0})
            yield event("chunk",  "done", f"All {len(records)} records unchanged — nothing to do")
            yield event("embed",  "done", "Skipped (no changes)")
            yield event("index",  "done", "Skipped (no changes)")
            yield event("upload", "done", "Skipped (no changes)", summary=summary)
            return

        yield event("chunk", "active",
                    f"Chunking {len(changed_records)} records ({skipped} unchanged, skipped)…")
        # Chunk records in parallel — each record is independent CPU work
        chunks = []
        CHUNK_BATCH = 200  # records per progress report
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(chunk_records, [r]): idx for idx, r in enumerate(changed_records)}
            results_map: dict[int, list] = {}
            for fut in as_completed(futs):
                idx = futs[fut]
                results_map[idx] = fut.result()
                done_count = len(results_map)
                if done_count % CHUNK_BATCH == 0 or done_count == len(changed_records):
                    partial = sum(len(v) for v in results_map.values())
                    yield event("chunk", "active",
                                f"Chunked {done_count}/{len(changed_records)} records → {partial} chunks so far…")
        # Reassemble in original order
        for idx in range(len(changed_records)):
            chunks.extend(results_map[idx])
        summary["chunks"] = len(chunks)
        yield event("chunk", "done",
                    f"Produced {len(chunks)} chunks from {len(changed_records)} changed records ({skipped} skipped)")

    except Exception as exc:
        yield event("chunk", "error", str(exc))
        return

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    try:
        texts = [c["text"] for c in chunks]
        batches = [texts[i: i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)]
        n_embed_batches = len(batches)
        yield event("embed", "active",
                    f"Embedding {len(chunks)} chunks in {n_embed_batches} batch(es) — {EMBED_CONCURRENCY} parallel…")

        model = AzureOpenAIEmbeddings(
            azure_endpoint=AOAI_ENDPOINT,
            api_key=AOAI_API_KEY,
            azure_deployment=AOAI_EMBEDDING_DEPLOYMENT,
            openai_api_version=AOAI_API_VERSION,
            max_retries=0,  # disable SDK retries — _embed_batch handles back-off
        )

        def _embed_batch(batch_idx: int, batch_texts: list[str]) -> tuple[int, list]:
            """Embed one batch with retry + exponential back-off that honours
            the API's own 'retry after N seconds' hint when present."""
            _retry_after_re = re.compile(r'retry after (\d+)', re.IGNORECASE)
            max_retries = 8
            for attempt in range(max_retries):
                try:
                    result = model.embed_documents(batch_texts)
                    # Proactive throttle: sleep before releasing the worker so the
                    # pool can't immediately start the next batch.  This keeps
                    # sustained throughput below the Azure OpenAI RPM quota.
                    if EMBED_INTER_BATCH_DELAY_S > 0:
                        time.sleep(EMBED_INTER_BATCH_DELAY_S)
                    return batch_idx, result
                except Exception as exc:
                    is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
                    if attempt < max_retries - 1 and is_rate_limit:
                        # Honour the API's Retry-After hint when present, otherwise
                        # use exponential back-off (2^attempt seconds) + random jitter
                        # so concurrent workers don't all retry at the same instant.
                        m = _retry_after_re.search(str(exc))
                        hint = int(m.group(1)) if m else 0
                        base_wait = max(hint, 2 ** attempt)
                        wait = base_wait + random.uniform(0.5, 3.0)
                        log.warning(
                            "Rate limited on embed batch %d — waiting %.0fs (attempt %d/%d)",
                            batch_idx + 1, wait, attempt + 1, max_retries,
                        )
                        time.sleep(wait)
                    else:
                        raise

        # Submit all batches to a thread pool; yield keepalives while waiting
        all_vectors: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
        completed = 0
        with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as pool:
            futures = {
                pool.submit(_embed_batch, idx, batch): idx
                for idx, batch in enumerate(batches)
            }
            while futures:
                done = set()
                for fut in list(futures):
                    if fut.done():
                        batch_idx = futures[fut]
                        batch_vectors = fut.result()[1]  # raises if _embed_batch raised
                        start = batch_idx * EMBED_BATCH_SIZE
                        for j, vec in enumerate(batch_vectors):
                            all_vectors[start + j] = vec
                        completed += 1
                        done.add(fut)
                        yield event("embed", "active",
                                    f"Embedded batch {completed}/{n_embed_batches}…")
                for fut in done:
                    del futures[fut]
                if futures:
                    # Brief sleep + keepalive to avoid busy-spin and keep SSE alive
                    time.sleep(1)
                    yield ": keepalive\n"

        for chunk, vector in zip(chunks, all_vectors):
            chunk["embedding"] = vector
        yield event("embed", "done", f"Embedded {len(chunks)} chunks")
    except Exception as exc:
        yield event("embed", "error", str(exc))
        return

    # ── Step 4: Index ─────────────────────────────────────────────────────────
    try:
        yield event("index", "active", "Ensuring search index exists…")
        created = ensure_search_index()
        summary.update({"index": SEARCH_INDEX_NAME, "index_created": created})
        yield event("index", "done",
                    f"Created index '{SEARCH_INDEX_NAME}'" if created
                    else f"Index '{SEARCH_INDEX_NAME}' already exists")
    except Exception as exc:
        yield event("index", "error", str(exc))
        return

    # ── Step 5: Upload ────────────────────────────────────────────────────────
    try:
        n_upload_batches = max(1, (len(chunks) + UPLOAD_BATCH_SIZE - 1) // UPLOAD_BATCH_SIZE)
        yield event("upload", "active",
                    f"Uploading {len(chunks)} documents in {n_upload_batches} batch(es)…")
        search_client = SearchClient(
            endpoint=SEARCH_ENDPOINT,
            index_name=SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(SEARCH_API_KEY),
        )
        failed = 0
        completed_uploads = 0

        def _upload_batch(batch_docs):
            res = search_client.upload_documents(documents=batch_docs)
            return sum(1 for r in res if not r.succeeded)

        upload_batches = [chunks[i: i + UPLOAD_BATCH_SIZE] for i in range(0, len(chunks), UPLOAD_BATCH_SIZE)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {pool.submit(_upload_batch, b): b_idx for b_idx, b in enumerate(upload_batches)}
            for fut in as_completed(futs):
                failed += fut.result()
                completed_uploads += 1
                yield event("upload", "active",
                            f"Uploading batch {completed_uploads}/{n_upload_batches}…")

        summary.update({"uploaded": len(chunks) - failed, "failed": failed})

        if failed == 0:
            save_hash_manifest(new_manifest)

        yield event("upload", "done",
                    f"Uploaded {len(chunks) - failed} documents ({failed} failed)",
                    summary=summary)
    except Exception as exc:
        yield event("upload", "error", str(exc))
        return


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    for evt in run_pipeline():
        print(evt)
