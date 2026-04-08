"""
RAG Ingestion Pipeline
======================
1. Fetch  — downloads the most recent wiki JSONL blob from Azure Blob Storage
2. Chunk  — splits pages into chunks using LangChain's RecursiveCharacterTextSplitter
3. Embed  — generates vectors via Azure OpenAI (only for changed/new pages)
4. Index  — creates/ensures Azure AI Search index
5. Upload — upserts documents + vectors into Azure AI Search

Change detection: a hash manifest (wiki_hashes.json) is stored in the blob
container after each run. On subsequent runs, only pages whose content has
changed are re-embedded and re-uploaded. Unchanged pages are skipped entirely.

Can be used as a module (run_pipeline generator) or standalone CLI.
"""

import hashlib
import json
import logging
import os
import re
from typing import Generator

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
# Config
# ---------------------------------------------------------------------------

STORAGE_CONN = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER", "wiki-crawl")
HASH_MANIFEST_BLOB = "wiki_hashes.json"

AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AOAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
AOAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_API_KEY = os.environ["AZURE_SEARCH_API_KEY"]
SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "wiki-index")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 100
UPLOAD_BATCH_SIZE = 100
VECTOR_DIM = 3072  # text-embedding-3-large; use 1536 for ada-002 / text-embedding-3-small


# ---------------------------------------------------------------------------
# Step 1: Fetch
# ---------------------------------------------------------------------------

def get_latest_blob_name() -> str:
    client = BlobServiceClient.from_connection_string(STORAGE_CONN)
    blobs = [b for b in client.get_container_client(CONTAINER).list_blobs() if b.name.endswith(".jsonl")]
    if not blobs:
        raise FileNotFoundError(f"No .jsonl blobs found in container '{CONTAINER}'")
    latest = max(blobs, key=lambda b: b.last_modified)
    return latest.name


def download_blob(blob_name: str) -> list[dict]:
    client = BlobServiceClient.from_connection_string(STORAGE_CONN)
    data = client.get_blob_client(container=CONTAINER, blob=blob_name).download_blob().readall().decode("utf-8")
    return [json.loads(line) for line in data.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Hash manifest — tracks page content hashes between runs
# ---------------------------------------------------------------------------

def load_hash_manifest() -> dict[str, str]:
    """Load {path: md5_hash} manifest from blob storage. Returns {} if not found."""
    client = BlobServiceClient.from_connection_string(STORAGE_CONN)
    blob = client.get_blob_client(container=CONTAINER, blob=HASH_MANIFEST_BLOB)
    try:
        data = blob.download_blob().readall().decode("utf-8")
        return json.loads(data)
    except ResourceNotFoundError:
        return {}


def save_hash_manifest(manifest: dict[str, str]) -> None:
    """Persist the updated manifest back to blob storage."""
    client = BlobServiceClient.from_connection_string(STORAGE_CONN)
    blob = client.get_blob_client(container=CONTAINER, blob=HASH_MANIFEST_BLOB)
    blob.upload_blob(json.dumps(manifest, indent=2).encode("utf-8"), overwrite=True)


def page_hash(record: dict) -> str:
    """MD5 of the page's markdown (or HTML as fallback)."""
    content = record.get("markdown", "") or record.get("html", "")
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def filter_changed_records(records: list[dict], manifest: dict[str, str]) -> tuple[list[dict], dict[str, str]]:
    """
    Compare each record against the manifest.
    Returns:
      - changed_records: pages that are new or have different content
      - new_manifest: updated {path: hash} for ALL pages (to be saved after upload)
    """
    changed = []
    new_manifest = {}
    for record in records:
        path = record.get("path", "/")
        h = page_hash(record)
        new_manifest[path] = h
        if manifest.get(path) != h:
            changed.append(record)
    return changed, new_manifest


# ---------------------------------------------------------------------------
# Step 2: Chunk
# ---------------------------------------------------------------------------

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(["p", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr"]):
        tag.insert_before("\n")
    return " ".join(soup.get_text(separator=" ").split())


def chunk_records(records: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
    )
    chunks: list[dict] = []
    for record in records:
        path = record.get("path", "/")
        url = record.get("remote_url", "")
        crawled_at = record.get("crawled_at", "")
        text = record.get("markdown", "").strip() or html_to_text(record.get("html", ""))
        if not text:
            continue
        lc_docs = splitter.create_documents(
            texts=[text],
            metadatas=[{"path": path, "url": url, "crawled_at": crawled_at}],
        )
        for idx, doc in enumerate(lc_docs):
            safe_id = re.sub(r"[^a-zA-Z0-9_\-=]", "_", path.strip("/").replace("/", "__")) or "root"
            chunks.append({
                "id": f"{safe_id}__{idx}",
                "text": doc.page_content,
                "path": path,
                "url": url,
                "crawled_at": crawled_at,
            })
    return chunks


# ---------------------------------------------------------------------------
# Step 3: Embed
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[dict], progress_cb=None) -> list[dict]:
    embeddings_model = AzureOpenAIEmbeddings(
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        azure_deployment=AOAI_EMBEDDING_DEPLOYMENT,
        openai_api_version=AOAI_API_VERSION,
    )
    texts = [c["text"] for c in chunks]
    total = len(texts)
    all_vectors: list[list[float]] = []
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[i: i + EMBED_BATCH_SIZE]
        all_vectors.extend(embeddings_model.embed_documents(batch))
        if progress_cb:
            progress_cb(min(i + EMBED_BATCH_SIZE, total), total)
    for chunk, vector in zip(chunks, all_vectors):
        chunk["embedding"] = vector
    return chunks


# ---------------------------------------------------------------------------
# Step 4: Index
# ---------------------------------------------------------------------------

def ensure_search_index() -> bool:
    """Returns True if index was created, False if it already existed."""
    index_client = SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )
    existing = [idx.name for idx in index_client.list_indexes()]
    if SEARCH_INDEX_NAME in existing:
        return False

    index = SearchIndex(
        name=SEARCH_INDEX_NAME,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="text", type=SearchFieldDataType.String),
            SimpleField(name="path", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="url", type=SearchFieldDataType.String),
            SimpleField(name="crawled_at", type=SearchFieldDataType.String, filterable=True),
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
    )
    index_client.create_index(index)
    return True


# ---------------------------------------------------------------------------
# Step 5: Upload
# ---------------------------------------------------------------------------

def upload_to_search(chunks: list[dict], progress_cb=None) -> int:
    """Returns number of failed documents."""
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )
    total = len(chunks)
    failed = 0
    for i in range(0, total, UPLOAD_BATCH_SIZE):
        batch = chunks[i: i + UPLOAD_BATCH_SIZE]
        results = search_client.upload_documents(documents=batch)
        failed += sum(1 for r in results if not r.succeeded)
        if progress_cb:
            progress_cb(min(i + UPLOAD_BATCH_SIZE, total), total)
    return failed


# ---------------------------------------------------------------------------
# Generator — used by the web server for streaming progress
# ---------------------------------------------------------------------------

def run_pipeline() -> Generator[dict, None, None]:
    """
    Runs the full pipeline, yielding progress events as dicts:
      { "step": str, "status": "active"|"done"|"error", "message": str }
    Final event also includes "summary" with stats.
    """
    summary = {}

    def event(step, status, message, **extra):
        d = {"step": step, "status": status, "message": message}
        d.update(extra)
        return d

    # --- Step 1: Fetch ---
    try:
        yield event("fetch", "active", "Finding latest blob…")
        blob_name = get_latest_blob_name()
        yield event("fetch", "active", f"Downloading {blob_name}…")
        records = download_blob(blob_name)
        summary["blob"] = blob_name
        summary["pages"] = len(records)
        yield event("fetch", "done", f"Fetched {len(records)} pages from {blob_name}")
    except Exception as e:
        yield event("fetch", "error", str(e))
        return

    # --- Step 2: Chunk (with change detection) ---
    try:
        yield event("chunk", "active", "Loading hash manifest…")
        manifest = load_hash_manifest()

        changed_records, new_manifest = filter_changed_records(records, manifest)
        skipped = len(records) - len(changed_records)
        summary["skipped"] = skipped

        if not changed_records:
            summary["chunks"] = 0
            summary["uploaded"] = 0
            summary["failed"] = 0
            yield event("chunk", "done", f"All {len(records)} pages unchanged — nothing to do")
            yield event("embed", "done", "Skipped (no changes)")
            yield event("index", "done", "Skipped (no changes)")
            yield event("upload", "done", "Skipped (no changes)", summary=summary)
            return

        yield event("chunk", "active", f"Chunking {len(changed_records)} changed pages ({skipped} unchanged, skipped)…")
        chunks = chunk_records(changed_records)
        summary["chunks"] = len(chunks)
        yield event("chunk", "done", f"Produced {len(chunks)} chunks from {len(changed_records)} changed pages ({skipped} skipped)")
    except Exception as e:
        yield event("chunk", "error", str(e))
        return

    # --- Step 3: Embed ---
    try:
        total_chunks = len(chunks)
        yield event("embed", "active", f"Embedding {total_chunks} chunks…")

        def embed_progress(done, total):
            pass  # could yield intermediate events here if needed

        chunks = embed_chunks(chunks, progress_cb=embed_progress)
        yield event("embed", "done", f"Embedded {total_chunks} chunks")
    except Exception as e:
        yield event("embed", "error", str(e))
        return

    # --- Step 4: Index ---
    try:
        yield event("index", "active", "Ensuring search index exists…")
        created = ensure_search_index()
        msg = f"Created index '{SEARCH_INDEX_NAME}'" if created else f"Index '{SEARCH_INDEX_NAME}' already exists"
        summary["index"] = SEARCH_INDEX_NAME
        summary["index_created"] = created
        yield event("index", "done", msg)
    except Exception as e:
        yield event("index", "error", str(e))
        return

    # --- Step 5: Upload ---
    try:
        yield event("upload", "active", f"Uploading {len(chunks)} documents…")

        failed = upload_to_search(chunks)
        summary["uploaded"] = len(chunks) - failed
        summary["failed"] = failed

        # Only save manifest if upload fully succeeded
        if failed == 0:
            save_hash_manifest(new_manifest)

        yield event("upload", "done", f"Uploaded {len(chunks) - failed} documents ({failed} failed)", summary=summary)
    except Exception as e:
        yield event("upload", "error", str(e))
        return


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for evt in run_pipeline():
        print(evt)
