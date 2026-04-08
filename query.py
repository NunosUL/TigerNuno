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

TOP_K = 8              # number of chunks retrieved from search
MAX_CONTEXT_CHARS = 12000  # ~3000 tokens of context sent to the LLM

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about internal
documentation from an Azure DevOps Wiki.

Rules:
- Answer using ONLY the provided context chunks.
- If the context does not contain enough information, say so clearly.
- Always cite the wiki pages you used at the end of your answer.
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

def hybrid_search(question: str, query_vector: list[float]) -> list[dict]:
    """
    Combines vector similarity search with full-text keyword search.
    Azure AI Search merges both result sets using Reciprocal Rank Fusion (RRF).
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

    results = client.search(
        search_text=question,       # full-text keywords
        vector_queries=[vector_query],  # vector similarity
        select=["id", "text", "path", "url", "crawled_at"],
        top=TOP_K,
    )

    return [
        {
            "id": r["id"],
            "text": r["text"],
            "path": r.get("path", ""),
            "url": r.get("url", ""),
            "crawled_at": r.get("crawled_at", ""),
            "score": r.get("@search.score", 0),
            "reranker_score": r.get("@search.reranker_score"),
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Step 3: Build context
# ---------------------------------------------------------------------------

def build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    - Rank by score (already sorted by Azure AI Search)
    - Deduplicate: skip chunks from the same page if text is very similar
    - Fit within MAX_CONTEXT_CHARS token budget
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

        # Soft-cap per page: don't let one page dominate context
        if seen_paths.get(path, 0) > MAX_CONTEXT_CHARS // 3:
            continue

        if total_chars + len(text) > MAX_CONTEXT_CHARS:
            break

        context_parts.append(f"[Source: {path}]\n{text}")
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


# ---------------------------------------------------------------------------
# Step 5: Confidence signal
# ---------------------------------------------------------------------------

def estimate_confidence(chunks: list[dict], answer: str) -> str:
    """
    Heuristic confidence based on:
    - Number of chunks retrieved
    - Top chunk score
    - Whether the LLM flagged uncertainty in its answer
    """
    if not chunks:
        return "low"

    uncertainty_phrases = [
        "i don't know", "not enough information", "cannot find",
        "no information", "not mentioned", "unclear",
    ]
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in uncertainty_phrases):
        return "low"

    top_score = chunks[0].get("score", 0)
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
    answer = call_llm(question, context)

    # 5. Confidence
    confidence = estimate_confidence(chunks, answer)

    return ChatResponse(answer=answer, sources=sources, confidence=confidence)
