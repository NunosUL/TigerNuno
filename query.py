"""
RAG Query Engine
================
Implements the query side of the RAG pipeline per the architecture diagram:

  1. Embed query    — Azure OpenAI embeddings (same model as ingestion)
  2. Hybrid search  — Azure AI Search: vector similarity + full-text keywords
  3. Build context  — rank, deduplicate, fit token budget, attach source metadata
  4. LLM generation — Azure OpenAI GPT-4o with system prompt
  5. Format         — answer text + source citations + confidence signal
"""

import logging
import os
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AOAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
AOAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
AOAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_API_KEY = os.environ["AZURE_SEARCH_API_KEY"]
SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "wiki-index")

TOP_K = 20             # number of chunks retrieved from semantic search
MAX_CONTEXT_CHARS     = 16000  # ~4000 tokens of context sent to the LLM

# Regex: matches "test case 296199", "TC 296199", "TC#296199", "test case #296199"
_TC_ID_RE = re.compile(r'\b(?:test[\s_-]*case|tc)\s*#?\s*(\d{4,})\b', re.IGNORECASE)

# Matches the full 40-char commit SHA embedded as *sha:{sha}* in WI markdown
_FULL_SHA_RE = re.compile(r'\*sha:([0-9a-f]{40})\*', re.IGNORECASE)

# Regex: matches the keyword + first ID, then group 2 captures any comma-separated IDs
# that follow on the same phrase (e.g. "work items 300340, 301889, 300950, 301867").
# Also handles plural forms ("work items", "change requests") and "change request".
_WI_ID_RE = re.compile(
    r'\b(?:epic|feature|user[\s_-]*story|task|bug|change[\s_-]*requests?|work[\s_-]*items?|wi)'
    r'\s*#?\s*(\d{4,})((?:\s*[,&]\s*\d{4,})*)',
    re.IGNORECASE,
)

SEMANTIC_CONFIG_NAME  = "tigerchat-semantic"
SCORING_PROFILE_NAME  = "source-boost"
MIN_RERANKER_SCORE    = 1.8   # drop chunks below this threshold (reranker_score 0–4 scale)
                               # raised from 1.0 — filters BM25-only keyword hits that the
                               # semantic reranker considers only weakly relevant
SEMANTIC_ENABLED      = os.environ.get("AZURE_SEARCH_SEMANTIC_ENABLED", "true").lower() == "true"
BOOST_SOURCE_TYPE     = os.environ.get("BOOST_SOURCE_TYPE", "")  # "wiki" | "code" | "test" | "" = no boost

# When a question pins specific TC IDs, supplementary (non-pinned) chunks must clear
# a higher reranker bar and are capped at this many distinct source paths.
PINNED_SUPPLEMENTARY_MIN_SCORE = 2.5
PINNED_SUPPLEMENTARY_MAX_PATHS = 3

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about an internal
Azure DevOps project. You have access to four types of source material:

- **Wiki pages** — internal documentation, guides, and process descriptions.
- **Source code** — C# classes, Razor views, configuration files, and other repository files.
- **Test cases** — structured test plans with step-by-step actions and expected results.
- **Work items** — Epics, Features, User Stories, Tasks, and Bugs with their full details and discussion threads.

Rules:
- Answer using ONLY the provided context chunks.
- If the context does not contain enough information, say so clearly.
- When answering about test cases, list them clearly with their ID, title, and steps.
- When answering about work items, include the ID, type, title, state, and relevant details.
- When answering about documentation, reference the wiki page.
- Do NOT append a "Sources:" or "References:" list — sources are shown separately in the UI.
- Be concise and structured. Use bullet points or numbered lists where appropriate.
- Do not invent information not present in the context.
- IMPORTANT — semantic precision: only include results that are genuinely and centrally about
  the named concept. If in doubt, exclude and say so rather than listing loosely related items.

TABLE FORMATTING — use markdown tables for any list of items:
- Whenever you list deployment packages, scripts, files, attachments, work items, test cases,
  API endpoints, or any other repeating structured data with 2+ items, present them as a
  markdown table rather than a bullet list.
- Use clear, short header names (e.g. Name | Path | Build ID, or File | Size | Date Added).
- Keep cell values concise — no markdown inside table cells.

ATTACHMENT DEDUPLICATION (applies when consolidating attachments from multiple work items):
- For attachments: if the same file name appears in more than one work item, keep only the
  row with the LATEST Date Added value. Discard older duplicates entirely.
- Always include the Work Item ID column so the source WI is traceable.
- If Date Added is unknown for all duplicates, keep the row from the highest-numbered Work Item ID.
- Consolidated attachments table columns: File Name | Size | Date Added | Work Item

CODE CHANGE FORMATTING — use this exact structure for every changed file:

**FILE** `<file path>`

---

**🔴 BEFORE**

<one-line plain-text description of what the code did before>

---

**🟢 AFTER**

<one-line plain-text description of what the code does after>

```<language>
<changed code block — properly indented, no truncation>
```

---

Rules for code change responses:
- Never inline code changes into a sentence. Always use the FILE / BEFORE / AFTER structure above.
- Always put the full file path in a backtick span on the FILE line.
- Each changed file gets its own FILE block, separated by a blank line.
- Code blocks must use triple backticks with the correct language tag (csharp, json, xml, etc.).
- Do not add explanatory prose inside the code block — keep descriptions on the BEFORE/AFTER label lines.
- Keep the horizontal rules (---) exactly as shown — they create visual separation when pasted into Word."""


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    answer: str
    sources: list[dict]   # [{"path": str, "url": str}]
    confidence: str       # "high" | "medium" | "low"


# ---------------------------------------------------------------------------
# Step 1: Embed query
# ---------------------------------------------------------------------------

def embed_query(question: str) -> list[float]:
    model = AzureOpenAIEmbeddings(
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        azure_deployment=AOAI_EMBEDDING_DEPLOYMENT,
        openai_api_version=AOAI_API_VERSION,
    )
    return model.embed_query(question)


# ---------------------------------------------------------------------------
# Step 2: Hybrid search (vector + full-text)
# ---------------------------------------------------------------------------

def _extract_tc_ids(question: str) -> list[int]:
    """Return any test-case numeric IDs explicitly mentioned in the question."""
    return [int(m.group(1)) for m in _TC_ID_RE.finditer(question)]


def _extract_wi_ids(question: str) -> list[int]:
    """Return any work-item numeric IDs explicitly mentioned in the question.

    Matches patterns like 'TASK 297723', 'BUG 12345', 'USER STORY 297707',
    'EPIC 1000', 'FEATURE 2000', 'work item(s) 1234', 'change request(s) 300340',
    'WI#1234', and comma/ampersand-separated lists:
    'work items 300340, 301889, 300950, 301867'.
    """
    ids: list[int] = []
    seen: set[int] = set()
    for m in _WI_ID_RE.finditer(question):
        first = int(m.group(1))
        if first not in seen:
            seen.add(first)
            ids.append(first)
        # Parse the comma/ampersand-separated tail (group 2), e.g. ", 301889, 300950"
        for extra in re.findall(r'\d{4,}', m.group(2) or ""):
            n = int(extra)
            if n not in seen:
                seen.add(n)
                ids.append(n)
    return ids


def _fetch_all_tc_chunks(tc_ids: list[int]) -> list[dict]:
    """Fetch EVERY indexed chunk for the given TC IDs using a path filter.

    This bypasses TOP_K and the semantic reranker threshold so that all sections
    of a test case (metadata, Description, Test Steps, Discussion) are guaranteed
    to appear in context when the user explicitly names the TC.
    """
    if not tc_ids:
        return []
    client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )
    all_chunks: list[dict] = []
    for tc_id in tc_ids:
        try:
            results = client.search(
                search_text="*",
                filter=f"path eq '/test-cases/{tc_id}'",
                select=["id", "text", "path", "url", "crawled_at"],
                top=50,           # a single TC won't produce more than 50 chunks
                # NOTE: do NOT use order_by — the id field is not sortable and
                # Azure AI Search would throw, causing a silent empty return.
            )
            for r in results:
                all_chunks.append({
                    "id":             r["id"],
                    "text":           r["text"],
                    "path":           r.get("path", ""),
                    "url":            r.get("url", ""),
                    "crawled_at":     r.get("crawled_at", ""),
                    "source_type":    "test",
                    "score":          99.0,   # pin above any semantic result
                    "reranker_score": 4.0,
                })
        except Exception as exc:
            log.warning("Could not fetch TC chunks for id=%s: %s", tc_id, exc)
    return all_chunks


def _fetch_all_wi_chunks(wi_ids: list[int]) -> list[dict]:
    """Fetch EVERY indexed chunk for the given work-item IDs using a path filter.

    Same pattern as _fetch_all_tc_chunks — bypasses TOP_K and the reranker
    threshold so that all sections of a work item (Description, Discussion,
    Commits, etc.) are guaranteed to appear when the user explicitly names it.
    """
    if not wi_ids:
        return []
    client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )
    all_chunks: list[dict] = []
    for wi_id in wi_ids:
        try:
            results = client.search(
                search_text="*",
                filter=f"path eq '/work-items/{wi_id}'",
                select=["id", "text", "path", "url", "crawled_at"],
                top=50,
            )
            for r in results:
                all_chunks.append({
                    "id":             r["id"],
                    "text":           r["text"],
                    "path":           r.get("path", ""),
                    "url":            r.get("url", ""),
                    "crawled_at":     r.get("crawled_at", ""),
                    "source_type":    "workitem",
                    "score":          99.0,
                    "reranker_score": 4.0,
                })
        except Exception as exc:
            log.warning("Could not fetch WI chunks for id=%s: %s", wi_id, exc)
    return all_chunks


# Matches rows of the commit file-change table written by _format_work_item_markdown:
#   | edit | /Services/ProgramHealth/FilterService.cs |
_COMMIT_FILE_ROW_RE = re.compile(
    r'\|\s*[\w][\w ,/\-]*\|\s*(/[\w/.\-_]+\.\w+)\s*\|'
)

# Matches child WI lines in ## Linked Work Items:
#   - Task 297435 · Update the frontend…
#   - Bug 12345 · Some bug title
_CHILD_WI_LINE_RE = re.compile(
    r'^-\s+(?:Epic|Feature|User Story|Task|Bug)\s+(\d{4,})',
    re.IGNORECASE | re.MULTILINE,
)


def _fetch_linked_child_wi_ids(wi_chunks: list[dict]) -> list[int]:
    """Parse child work-item IDs from the '## Linked Work Items / Children' section
    of any pinned WI chunk.  This enables single-entry-point questions:

        "Tell me about USER STORY 297706 and all its child tasks"

    The parent WI markdown contains lines like:
        **Children (2):**
        - Task 297435 · Update the frontend
        - Task 297437 · Update the backend

    We extract all child IDs and return them so the caller can pin those chunks too.
    Skips IDs that are already pinned (the caller deduplicates).
    """
    child_ids: list[int] = []
    seen: set[int] = set()
    for chunk in wi_chunks:
        text = chunk.get("text", "")
        if "## Linked Work Items" not in text:
            continue
        # Isolate the Linked Work Items block so we don't catch sibling IDs
        block_start = text.find("## Linked Work Items")
        # End at the next ## heading or end of text
        next_heading = text.find("\n## ", block_start + 4)
        block = text[block_start: next_heading if next_heading != -1 else len(text)]

        # Only scan lines after "**Children"
        in_children = False
        for line in block.splitlines():
            if "**Children" in line:
                in_children = True
            elif line.startswith("**") and in_children:
                in_children = False  # moved past children block
            elif in_children:
                m = _CHILD_WI_LINE_RE.match(line.strip())
                if m:
                    wid = int(m.group(1))
                    if wid not in seen:
                        seen.add(wid)
                        child_ids.append(wid)
    return child_ids


def _fetch_commit_code_chunks(wi_chunks: list[dict]) -> list[dict]:
    """Pull source-code chunks that correspond to files touched in WI commits.

    Two strategies, tried in order:

    Strategy A — Commit file table (precise):
        Parse '## Git Commits' file-change tables from the WI markdown.
        Search the code index for each filename exactly.

    Strategy B — Title-keyword fallback (when no commit table is present):
        If a Task/Bug WI has no commit data indexed (ingested before the
        dev_info feature was added, or the DevOps artifact links use a
        non-standard format), extract 2-3 key noun phrases from the WI title
        and run a semantic BM25 search against /repos/ paths.

    Caps at 10 distinct files to stay within the context budget.
    """
    client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )
    results: list[dict] = []
    seen_ids: set[str] = set()

    def _add_code_result(r: dict) -> None:
        if r["id"] not in seen_ids and len(results) < 10:
            seen_ids.add(r["id"])
            results.append({
                "id":             r["id"],
                "text":           r["text"],
                "path":           r.get("path", ""),
                "url":            r.get("url", ""),
                "crawled_at":     r.get("crawled_at", ""),
                "source_type":    "code",
                "score":          90.0,
                "reranker_score": 3.5,
            })

    # ── Strategy A: commit file table ────────────────────────────────────────
    file_paths: list[str] = []
    seen_fps: set[str] = set()
    wi_with_no_commits: list[dict] = []

    for chunk in wi_chunks:
        text = chunk.get("text", "")
        has_commits = "## Git Commits" in text
        if has_commits:
            for m in _COMMIT_FILE_ROW_RE.finditer(text):
                fp = m.group(1).strip()
                if fp and fp not in seen_fps:
                    seen_fps.add(fp)
                    file_paths.append(fp)
        else:
            # Check if this looks like a Task or Bug (type is in the heading)
            first_line = text.splitlines()[0] if text else ""
            if first_line.startswith("# Task:") or first_line.startswith("# Bug:"):
                wi_with_no_commits.append(chunk)

    for fp in file_paths[:10]:
        filename = fp.split("/")[-1]          # e.g. "FilterService.cs"
        stem     = filename.rsplit(".", 1)[0]  # e.g. "FilterService"
        if len(stem) < 4:
            continue
        # Dotted stems like "RetailEvents.Api" tokenize poorly in phrase search —
        # use only the first segment so Azure Search finds the file reliably.
        search_term = stem.split(".")[0] if "." in stem else stem
        try:
            sr = client.search(
                search_text=f'"{search_term}"',
                filter="path ge '/repos/' and path lt '/repos~'",
                select=["id", "text", "path", "url", "crawled_at"],
                top=5,  # increased from 3 — config/project files rank lower than .cs files
            )
            for r in sr:
                if filename.lower() in r.get("path", "").lower():
                    _add_code_result(r)
        except Exception as exc:
            log.debug("Strategy A: could not fetch code for %s: %s", fp, exc)

    # ── Strategy B: title-keyword fallback ───────────────────────────────────
    # Only kick in for Tasks/Bugs that had no commits, and only if A found nothing
    if not results and wi_with_no_commits:
        for chunk in wi_with_no_commits[:3]:  # cap: at most 3 WIs to fallback on
            text  = chunk.get("text", "")
            # Extract title from "# Task: Some Title Here" heading
            title_match = re.match(r'^#\s+(?:Task|Bug):\s+(.+)$', text.splitlines()[0].strip())
            if not title_match:
                continue
            title = title_match.group(1).strip()
            # Strip noise words — keep meaningful nouns/verbs
            stopwords = {
                "the","a","an","and","or","to","for","in","on","of","with",
                "is","are","was","were","be","been","by","at","from","that",
                "this","it","as","its","into","through","update","updates",
            }
            keywords = [
                w for w in re.split(r'\W+', title)
                if len(w) >= 4 and w.lower() not in stopwords
            ]
            if not keywords:
                continue
            # CamelCase/PascalCase words (e.g. "RetailEvents") are component names
            # that actually appear in code files.  Generic action words like "Upgrade"
            # or "Migrate" do not — combining them in an AND-query makes matches fail.
            # Use the longest CamelCase word alone; fall back to multi-keyword only if
            # no component name is present.
            camel = [w for w in keywords if re.search(r'[A-Z][a-z]', w)]
            if camel:
                query = f'"{max(camel, key=len)}"'
            else:
                query = " ".join(f'"{kw}"' for kw in keywords[:4])
            try:
                sr = client.search(
                    search_text=query,
                    filter="path ge '/repos/' and path lt '/repos~'",
                    select=["id", "text", "path", "url", "crawled_at"],
                    top=5,
                )
                for r in sr:
                    _add_code_result(r)
            except Exception as exc:
                log.debug("Strategy B: keyword search failed for '%s': %s", title, exc)

    return results


def _fetch_commit_diff_chunks(wi_chunks: list[dict]) -> list[dict]:
    """Fetch diff documents for every commit SHA embedded in pinned WI chunks.

    During ingestion, each commit heading in the WI markdown contains a line
    *sha:{full40CharSHA}* so we can do an exact path lookup against
    /commit-diffs/{sha} without any structural parsing of the diff document itself.
    """
    shas: list[str] = []
    seen: set[str] = set()
    for chunk in wi_chunks:
        for m in _FULL_SHA_RE.finditer(chunk.get("text", "")):
            sha = m.group(1).lower()
            if sha not in seen:
                seen.add(sha)
                shas.append(sha)
    if not shas:
        return []

    client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )
    results: list[dict] = []
    for sha in shas[:15]:   # cap: at most 15 commits per question
        try:
            for r in client.search(
                search_text="*",
                filter=f"path eq '/commit-diffs/{sha}'",
                select=["id", "text", "path", "url", "crawled_at"],
                top=10,
            ):
                results.append({
                    "id":            r["id"],
                    "text":          r["text"],
                    "path":          r["path"],
                    "url":           r.get("url", ""),
                    "crawled_at":    r.get("crawled_at", ""),
                    "source_type":   "commit-diff",
                    "score":         95.0,   # Pass 1b: above code (90), below pinned WI (99)
                    "reranker_score": 4.0,
                })
        except Exception as exc:
            log.debug("Could not fetch diff for commit %s: %s", sha[:7], exc)
    return results


def _filter_supplementary(
    semantic_chunks: list[dict],
    pinned_paths: set[str],
) -> list[dict]:
    """When items are pinned, apply a stricter reranker threshold and path cap to
    any supplementary semantic results so only highly relevant additional sources
    appear alongside the explicitly requested items.

    Chunks whose path is already covered by a pinned item pass through unchanged
    (they just add more sections of the same document).  New, distinct paths must
    clear PINNED_SUPPLEMENTARY_MIN_SCORE and are capped at PINNED_SUPPLEMENTARY_MAX_PATHS.
    """
    filtered: list[dict] = []
    extra_paths: set[str] = set()
    for c in semantic_chunks:
        score = c.get("reranker_score") or 0
        if c["path"] in pinned_paths:
            # Extra chunk from an already-pinned document — always keep
            if score >= MIN_RERANKER_SCORE:
                filtered.append(c)
        elif score >= PINNED_SUPPLEMENTARY_MIN_SCORE:
            if len(extra_paths) < PINNED_SUPPLEMENTARY_MAX_PATHS:
                extra_paths.add(c["path"])
                filtered.append(c)
    return filtered


def _source_type_filter(source_types: list[str] | None) -> str | None:
    """Build an OData filter expression for Azure AI Search from a source_types list."""
    if not source_types:
        return None
    if len(source_types) == 1:
        return f"source_type eq '{source_types[0]}'"
    clauses = " or ".join(f"source_type eq '{t}'" for t in source_types)
    return f"({clauses})"


def _execute_search(question: str, query_vector: list[float],
                    source_filter: str | None = None) -> list[dict]:
    """
    Run the Azure AI Search query (vector + BM25 + RRF + optional semantic reranker
    and scoring profile) and return raw results WITHOUT the min-score filter applied.
    Used by both hybrid_search() and answer_question_stream().
    """
    client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=TOP_K,
        fields="embedding",
    )

    search_kwargs: dict = dict(
        search_text=question,
        vector_queries=[vector_query],
        select=["id", "text", "path", "url", "crawled_at"],
        top=TOP_K,
    )

    if SEMANTIC_ENABLED:
        search_kwargs["query_type"] = "semantic"
        search_kwargs["semantic_configuration_name"] = SEMANTIC_CONFIG_NAME

    if BOOST_SOURCE_TYPE:
        search_kwargs["scoring_profile"] = SCORING_PROFILE_NAME
        search_kwargs["scoring_parameters"] = [f"boostTags-{BOOST_SOURCE_TYPE}"]

    if source_filter:
        search_kwargs["filter"] = source_filter

    def _parse(results) -> list[dict]:
        return [
            {
                "id":             r["id"],
                "text":           r["text"],
                "path":           r.get("path", ""),
                "url":            r.get("url", ""),
                "crawled_at":     r.get("crawled_at", ""),
                "source_type":    _source_type_from_path(r.get("path", "")),
                "score":          r.get("@search.score", 0),
                "reranker_score": r.get("@search.reranker_score"),
            }
            for r in results
        ]

    try:
        return _parse(client.search(**search_kwargs))
    except Exception as e:
        err = str(e)
        # Index predates semantic config or scoring profile — fall back gracefully
        if "semanticConfiguration" in err or "scoringProfile" in err or "source_type" in err:
            search_kwargs.pop("query_type", None)
            search_kwargs.pop("semantic_configuration_name", None)
            search_kwargs.pop("scoring_profile", None)
            search_kwargs.pop("scoring_parameters", None)
            return _parse(client.search(**search_kwargs))
        raise


def hybrid_search(question: str, query_vector: list[float],
                  source_types: list[str] | None = None) -> list[dict]:
    """Hybrid search with TC/WI-ID pinning, recursive child resolution,
    commit-code auto-fetch, and strict supplementary filtering.

    1. Pin ALL chunks for explicitly named TC and WI IDs (bypass reranker).
    2. Auto-resolve children: if a pinned WI lists child IDs in its markdown
       '## Linked Work Items / Children' section, fetch those too — one level deep.
       This means naming only USER STORY 297706 automatically pulls TASK 297435
       and TASK 297437 into context without the user having to know their IDs.
    3. Fetch source-code chunks for files touched in any Bug/Task commit (two
       strategies: exact commit file table → keyword fallback).
    4. Restrict supplementary semantic results to high-scoring, non-redundant sources.

    source_types: when provided, restricts the semantic search to those source types
    and skips pinning logic for source types not in the list.
    """
    source_filter = _source_type_filter(source_types)
    include_wi    = not source_types or "workitem" in source_types
    include_code  = not source_types or "code" in source_types
    include_diffs = not source_types or "commit-diff" in source_types

    tc_chunks    = _fetch_all_tc_chunks(_extract_tc_ids(question))
    wi_ids       = _extract_wi_ids(question) if include_wi else []
    wi_chunks    = _fetch_all_wi_chunks(wi_ids) if wi_ids else []

    # Auto-resolve children one level deep
    child_ids    = _fetch_linked_child_wi_ids(wi_chunks) if include_wi else []
    # Only fetch children not already explicitly named
    new_child_ids = [cid for cid in child_ids if cid not in wi_ids]
    child_chunks  = _fetch_all_wi_chunks(new_child_ids) if new_child_ids else []

    all_wi_chunks = wi_chunks + child_chunks
    code_chunks   = _fetch_commit_code_chunks(all_wi_chunks) if include_code else []
    diff_chunks   = _fetch_commit_diff_chunks(all_wi_chunks) if include_diffs else []
    pinned        = tc_chunks + all_wi_chunks + diff_chunks + code_chunks
    has_pinned    = bool(tc_chunks or all_wi_chunks)

    semantic_chunks = _execute_search(question, query_vector, source_filter=source_filter)
    if SEMANTIC_ENABLED:
        if has_pinned:
            pinned_paths = {c["path"] for c in pinned}
            semantic_chunks = _filter_supplementary(semantic_chunks, pinned_paths)
        else:
            semantic_chunks = [c for c in semantic_chunks
                               if (c.get("reranker_score") or 0) >= MIN_RERANKER_SCORE]

    seen_ids = {c["id"] for c in pinned}
    merged   = list(pinned)
    for c in semantic_chunks:
        if c["id"] not in seen_ids:
            seen_ids.add(c["id"])
            merged.append(c)
    return merged


# ---------------------------------------------------------------------------
# Step 3: Build context
# ---------------------------------------------------------------------------

def _source_type_from_path(path: str) -> str:
    """Return short source type key derived from the path (works without index field)."""
    if path.startswith("/repos/"):
        return "code"
    if path.startswith("/test-cases/"):
        return "test"
    if path.startswith("/work-items/"):
        return "workitem"
    return "wiki"


def _source_type(path: str) -> str:
    """Classify a chunk path into a human-readable source type label."""
    if path.startswith("/repos/"):
        return "Source Code"
    if path.startswith("/test-cases/"):
        return "Test Case"
    if path.startswith("/work-items/"):
        return "Work Item"
    return "Wiki"


def build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Two-pass context builder.

    Pass 1a — explicitly pinned IDs (score ≥ 99, i.e. named TC/WI IDs):
               ALL their chunks are included with NO total-budget cap and
               NO per-path cap.  This guarantees commits, branches, and
               discussion sections of a named work item always reach the LLM.

    Pass 1b — commit-code chunks (score ≈ 90): included with normal per-path
               cap but still outside the total-budget gate, so they always
               accompany pinned items.

    Pass 2  — supplementary semantic results: fill whatever budget remains.
               Uses both per-path cap and total-budget gate; switches to
               `continue` (not `break`) so a single over-large chunk does
               not cut off smaller later ones.
    """
    seen_paths: dict[str, int] = {}
    seen_ids: set[str] = set()
    context_parts: list[str] = []
    sources: list[dict] = []
    total_chars = 0

    def _add(chunk: dict, *, ignore_path_cap: bool = False) -> None:
        nonlocal total_chars
        cid  = chunk["id"]
        path = chunk["path"]
        text = chunk["text"].strip()
        if not text or cid in seen_ids:
            return
        seen_ids.add(cid)

        if not ignore_path_cap:
            rich_item = path.startswith("/test-cases/") or path.startswith("/work-items/")
            path_cap  = MAX_CONTEXT_CHARS // 2 if rich_item else MAX_CONTEXT_CHARS // 4
            if seen_paths.get(path, 0) > path_cap:
                return

        source_label = f"[{_source_type(path)}: {path}]"
        context_parts.append(f"{source_label}\n{text}")
        total_chars += len(text)
        seen_paths[path] = seen_paths.get(path, 0) + len(text)

        if not any(s["path"] == path for s in sources):
            sources.append({"path": path, "url": chunk["url"]})

    # Pass 1a: pinned IDs — no cap of any kind so commits/branches always appear
    for chunk in chunks:
        if (chunk.get("score") or 0) >= 99.0:
            _add(chunk, ignore_path_cap=True)

    # Pass 1b: commit-code chunks — respect per-path cap, bypass total budget
    for chunk in chunks:
        score = chunk.get("score") or 0
        if 90.0 <= score < 99.0:
            _add(chunk, ignore_path_cap=False)

    # Pass 2: supplementary semantic results — respect total budget
    for chunk in chunks:
        if (chunk.get("score") or 0) >= 90.0:
            continue
        if total_chars >= MAX_CONTEXT_CHARS:
            break
        text = chunk["text"].strip()
        if total_chars + len(text) <= MAX_CONTEXT_CHARS:
            _add(chunk)

    return "\n\n---\n\n".join(context_parts), sources


# ---------------------------------------------------------------------------
# Step 4: LLM generation
# ---------------------------------------------------------------------------

def call_llm(question: str, context: str) -> str:
    client = AzureOpenAI(
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        api_version=AOAI_API_VERSION,
    )

    user_message = f"""Context:
{context}

Question: {question}"""

    response = client.chat.completions.create(
        model=AOAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=4096,
    )

    return response.choices[0].message.content.strip()


def _strip_sources_block(answer: str) -> str:
    """Remove any trailing Sources / References block the LLM appended despite instructions."""
    return re.sub(
        r"\n{1,2}(?:#+\s*)?(?:sources?|references?)[:\s]*\n.*",
        "",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    ).rstrip()


# ---------------------------------------------------------------------------
# Step 5: Confidence signal
# ---------------------------------------------------------------------------

def estimate_confidence(chunks: list[dict], answer: str) -> str:
    """
    Heuristic confidence.
    - When specific items were pinned (score >= 99): the data is authoritative.
      HIGH if >= 2 pinned chunks were found; skip uncertainty-phrase check because
      "no relevant changes" / "not applicable" are VALID PFQ answers, not gaps.
    - When semantic reranker is active: uses @search.reranker_score (0–4 scale).
      Thresholds: ≥ 2.5 + 3 chunks → HIGH; ≥ 1.5 + 2 chunks → MEDIUM; else LOW.
    - Fallback when semantic is disabled: uses @search.score (RRF, ~0.01–0.06).
      Thresholds: > 0.03 + 4 chunks → HIGH; > 0.01 + 2 chunks → MEDIUM; else LOW.
    Gate 1 (no chunks) always overrides to LOW.
    Gate 2 (LLM uncertainty language) overrides to LOW only for non-pinned answers.
    """
    if not chunks:
        return "low"

    # When the user explicitly named items (IDs pinned), their chunks have score=99.
    # Multiple pinned chunks = grounded, authoritative answer.
    # Do NOT apply the uncertainty-phrase gate — "No schema changes in this story"
    # is a HIGH-confidence answer, not an "I don't know".
    pinned_count = sum(1 for c in chunks if (c.get("score") or 0) >= 99.0)
    if pinned_count >= 2:
        return "high"
    if pinned_count == 1:
        return "medium"   # found but thin — only one chunk pinned

    # Non-pinned path: check for uncertainty language
    uncertainty_phrases = [
        "i don't know", "not enough information", "cannot find",
        "no information", "not mentioned", "unclear",
        "context does not include", "context does not contain",
        "no context", "unable to find",
    ]
    if any(phrase in answer.lower() for phrase in uncertainty_phrases):
        return "low"

    top = chunks[0]
    reranker_score = top.get("reranker_score")

    if reranker_score is not None:
        if reranker_score >= 2.5 and len(chunks) >= 3:
            return "high"
        if reranker_score >= 1.5 and len(chunks) >= 2:
            return "medium"
        return "low"

    top_score = top.get("score", 0)
    if top_score > 0.03 and len(chunks) >= 4:
        return "high"
    if top_score > 0.01 and len(chunks) >= 2:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer_question(question: str,
                    source_types: list[str] | None = None) -> ChatResponse:
    """Full RAG query pipeline. Returns a ChatResponse."""
    # 1. Embed
    query_vector = embed_query(question)

    # 2. Hybrid search
    chunks = hybrid_search(question, query_vector, source_types=source_types)

    # 3. Build context
    context, sources = build_context(chunks)

    # 4. LLM
    answer = _strip_sources_block(call_llm(question, context))

    # 5. Confidence
    confidence = estimate_confidence(chunks, answer)

    return ChatResponse(answer=answer, sources=sources, confidence=confidence)


def answer_question_stream(question: str,
                           source_types: list[str] | None = None):
    """
    Generator version of the RAG pipeline that yields SSE progress dicts so the
    UI can show a live step-by-step indicator while processing.

    Each yielded dict has:
      {"type": "status", "step": "<id>", "message": "<human text>"}
    Final yield:
      {"type": "done", "answer": ..., "sources": [...], "confidence": ...}
    """
    # Step 1 — Embed
    yield {"type": "status", "step": "embed",
           "message": "Embedding your question…"}
    query_vector = embed_query(question)

    # Step 2 — Azure AI Search (vector + BM25 + RRF + semantic reranker in one call)
    yield {"type": "status", "step": "search",
           "message": "Querying Azure AI Search (vector + BM25 + RRF)…"}
    source_filter = _source_type_filter(source_types)
    raw_chunks = _execute_search(question, query_vector, source_filter=source_filter)

    # Step 3 — Semantic reranker filter + TC/WI-ID pinning + child resolution + code fetch
    include_wi    = not source_types or "workitem" in source_types
    include_code  = not source_types or "code" in source_types
    include_diffs = not source_types or "commit-diff" in source_types

    tc_ids       = _extract_tc_ids(question)
    wi_ids       = _extract_wi_ids(question) if include_wi else []
    tc_chunks    = _fetch_all_tc_chunks(tc_ids)
    wi_chunks    = _fetch_all_wi_chunks(wi_ids) if wi_ids else []

    # Auto-resolve children one level deep
    child_ids     = _fetch_linked_child_wi_ids(wi_chunks) if include_wi else []
    new_child_ids = [cid for cid in child_ids if cid not in wi_ids]
    child_chunks  = _fetch_all_wi_chunks(new_child_ids) if new_child_ids else []

    all_wi_chunks = wi_chunks + child_chunks
    code_chunks   = _fetch_commit_code_chunks(all_wi_chunks) if include_code else []
    diff_chunks   = _fetch_commit_diff_chunks(all_wi_chunks) if include_diffs else []
    pinned        = tc_chunks + all_wi_chunks + diff_chunks + code_chunks
    has_pinned    = bool(tc_chunks or all_wi_chunks)

    if SEMANTIC_ENABLED:
        if has_pinned:
            pinned_paths = {c["path"] for c in pinned}
            kept         = _filter_supplementary(raw_chunks, pinned_paths)
        else:
            kept = [c for c in raw_chunks if (c.get("reranker_score") or 0) >= MIN_RERANKER_SCORE]
    else:
        kept = raw_chunks

    dropped = len(raw_chunks) - len(kept)

    pin_parts = []
    if tc_chunks:
        pin_parts.append(f"{len(tc_chunks)} chunk(s) from {len(tc_ids)} TC ID(s)")
    if all_wi_chunks:
        n_direct   = len(wi_chunks)
        n_children = len(child_chunks)
        wi_label   = f"{n_direct} explicit"
        if n_children:
            wi_label += f" + {n_children} child WI(s) auto-resolved"
        pin_parts.append(f"{len(all_wi_chunks)} WI chunk(s) ({wi_label})")
    if code_chunks:
        pin_parts.append(f"{len(code_chunks)} code file chunk(s)")
    pin_msg = (" + pinned: " + ", ".join(pin_parts)) if pin_parts else ""

    supp_note = " (strict supplementary filter active)" if has_pinned else ""
    rerank_msg = (
        f"Semantic reranker scored {len(raw_chunks)} results — "
        f"kept {len(kept)}, dropped {dropped}{supp_note}{pin_msg}"
        if SEMANTIC_ENABLED
        else f"Retrieved {len(kept)} results (semantic reranker disabled){pin_msg}"
    )
    yield {"type": "status", "step": "rerank", "message": rerank_msg}

    # Merge: pinned first, then filtered semantic (deduped by id)
    seen_ids = {c["id"] for c in pinned}
    merged   = list(pinned)
    for c in kept:
        if c["id"] not in seen_ids:
            seen_ids.add(c["id"])
            merged.append(c)
    chunks = merged

    # Step 4 — Context builder
    source_types = set(c.get("source_type", "wiki") for c in chunks)
    labels = " · ".join(sorted(source_types)) if source_types else "mixed"
    yield {"type": "status", "step": "context",
           "message": f"Building context from {len(chunks)} chunk{'s' if len(chunks) != 1 else ''} ({labels})…"}
    context, sources = build_context(chunks)

    # Step 5 — LLM
    yield {"type": "status", "step": "generate",
           "message": "GPT-4o is generating your answer…"}
    answer = _strip_sources_block(call_llm(question, context))

    confidence = estimate_confidence(chunks, answer)
    yield {
        "type":       "done",
        "answer":     answer,
        "sources":    sources,
        "confidence": confidence,
    }
