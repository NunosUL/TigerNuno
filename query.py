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

import os
import re
from dataclasses import dataclass

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

TOP_K = 15             # number of chunks retrieved from search
MAX_CONTEXT_CHARS     = 16000  # ~4000 tokens of context sent to the LLM

SEMANTIC_CONFIG_NAME  = "tigerchat-semantic"
SCORING_PROFILE_NAME  = "source-boost"
MIN_RERANKER_SCORE    = 1.0   # drop chunks below this threshold (reranker_score 0–4 scale)
SEMANTIC_ENABLED      = os.environ.get("AZURE_SEARCH_SEMANTIC_ENABLED", "true").lower() == "true"
BOOST_SOURCE_TYPE     = os.environ.get("BOOST_SOURCE_TYPE", "")  # "wiki" | "code" | "test" | "" = no boost

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about an internal
Azure DevOps project. You have access to three types of source material:

- **Wiki pages** — internal documentation, guides, and process descriptions.
- **Source code** — C# classes, Razor views, configuration files, and other repository files.
- **Test cases** — structured test plans with step-by-step actions and expected results.

Rules:
- Answer using ONLY the provided context chunks.
- If the context does not contain enough information, say so clearly.
- When answering about test cases, list them clearly with their ID, title, and steps.
- When answering about code, reference the file path and relevant logic.
- When answering about documentation, reference the wiki page.
- Do NOT append a "Sources:" or "References:" list — sources are shown separately in the UI.
- Be concise and structured. Use bullet points or numbered lists where appropriate.
- Do not invent information not present in the context."""


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

def _execute_search(question: str, query_vector: list[float]) -> list[dict]:
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


def hybrid_search(question: str, query_vector: list[float]) -> list[dict]:
    """Search + optional semantic reranker, with min-score filter applied."""
    chunks = _execute_search(question, query_vector)
    if SEMANTIC_ENABLED:
        chunks = [c for c in chunks if (c.get("reranker_score") or 0) >= MIN_RERANKER_SCORE]
    return chunks


# ---------------------------------------------------------------------------
# Step 3: Build context
# ---------------------------------------------------------------------------

def _source_type_from_path(path: str) -> str:
    """Return short source type key derived from the path (works without index field)."""
    if path.startswith("/repos/"):
        return "code"
    if path.startswith("/test-cases/"):
        return "test"
    return "wiki"


def _source_type(path: str) -> str:
    """Classify a chunk path into a human-readable source type label."""
    if path.startswith("/repos/"):
        return "Source Code"
    if path.startswith("/test-cases/"):
        return "Test Case"
    return "Wiki"


def build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    - Rank by score (already sorted by Azure AI Search)
    - Deduplicate: skip chunks from the same path if already well-represented
    - Fit within MAX_CONTEXT_CHARS token budget
    - Label each chunk with its source type so the LLM knows what it's reading
    - Return (context_string, unique_sources)
    """
    seen_paths: dict[str, int] = {}   # path → chars already included
    seen_ids: set[str] = set()
    context_parts: list[str] = []
    sources: list[dict] = []
    total_chars = 0

    for chunk in chunks:
        cid = chunk["id"]
        path = chunk["path"]
        text = chunk["text"].strip()

        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        # Soft-cap per path: don't let a single file/page dominate context
        if seen_paths.get(path, 0) > MAX_CONTEXT_CHARS // 4:
            continue

        if total_chars + len(text) > MAX_CONTEXT_CHARS:
            break

        source_label = f"[{_source_type(path)}: {path}]"
        context_parts.append(f"{source_label}\n{text}")
        total_chars += len(text)
        seen_paths[path] = seen_paths.get(path, 0) + len(text)

        # Track unique sources for citations
        if not any(s["path"] == path for s in sources):
            sources.append({"path": path, "url": chunk["url"]})

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
        max_tokens=1024,
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
    - When semantic reranker is active: uses @search.reranker_score (0–4 scale).
      Thresholds: ≥ 2.5 + 3 chunks → HIGH; ≥ 1.5 + 2 chunks → MEDIUM; else LOW.
    - Fallback when semantic is disabled: uses @search.score (RRF, ~0.01–0.06).
      Thresholds: > 0.03 + 4 chunks → HIGH; > 0.01 + 2 chunks → MEDIUM; else LOW.
    Gate 1 (no chunks) and Gate 2 (LLM uncertainty language) always override to LOW.
    """
    if not chunks:
        return "low"

    uncertainty_phrases = [
        "i don't know", "not enough information", "cannot find",
        "no information", "not mentioned", "unclear",
    ]
    if any(phrase in answer.lower() for phrase in uncertainty_phrases):
        return "low"

    top = chunks[0]
    reranker_score = top.get("reranker_score")

    if reranker_score is not None:
        # Semantic reranker active — 0 to 4 scale
        if reranker_score >= 2.5 and len(chunks) >= 3:
            return "high"
        if reranker_score >= 1.5 and len(chunks) >= 2:
            return "medium"
        return "low"

    # Fallback: RRF @search.score scale
    top_score = top.get("score", 0)
    if top_score > 0.03 and len(chunks) >= 4:
        return "high"
    if top_score > 0.01 and len(chunks) >= 2:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer_question(question: str) -> ChatResponse:
    """Full RAG query pipeline. Returns a ChatResponse."""
    # 1. Embed
    query_vector = embed_query(question)

    # 2. Hybrid search
    chunks = hybrid_search(question, query_vector)

    # 3. Build context
    context, sources = build_context(chunks)

    # 4. LLM
    answer = _strip_sources_block(call_llm(question, context))

    # 5. Confidence
    confidence = estimate_confidence(chunks, answer)

    return ChatResponse(answer=answer, sources=sources, confidence=confidence)


def answer_question_stream(question: str):
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
    raw_chunks = _execute_search(question, query_vector)

    # Step 3 — Semantic reranker filter (happened server-side; we now apply min-score)
    kept = ([c for c in raw_chunks if (c.get("reranker_score") or 0) >= MIN_RERANKER_SCORE]
            if SEMANTIC_ENABLED else raw_chunks)
    dropped = len(raw_chunks) - len(kept)
    rerank_msg = (
        f"Semantic reranker scored {len(raw_chunks)} results — "
        f"kept {len(kept)}, dropped {dropped} below threshold"
        if SEMANTIC_ENABLED
        else f"Retrieved {len(kept)} results (semantic reranker disabled)"
    )
    yield {"type": "status", "step": "rerank", "message": rerank_msg}
    chunks = kept

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
