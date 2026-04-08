"""
Wiki RAG Parser
Reads the most recent wiki JSONL blob from Azure Blob Storage,
parses and cleans each page, chunks by heading sections, and
outputs a list of RAG-ready documents:

  [
    {
      "id":       "Home__0",
      "text":     "# Home\nWelcome to the wiki...",
      "metadata": { "path": "/Home", "url": "...", "heading": "Home", "crawled_at": "..." }
    },
    ...
  ]

Usage:
  from parser import load_rag_documents
  docs = load_rag_documents()          # fetches latest blob automatically
"""

import io
import json
import logging
import os
import re

from azure.storage.blob import BlobServiceClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

STORAGE_CONN = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER", "wiki-crawl")

# Max characters per chunk before it gets split further.
# ~1500 chars ≈ ~300-400 tokens — comfortable for most embedding models.
MAX_CHUNK_CHARS = 1500


# ---------------------------------------------------------------------------
# Blob helpers
# ---------------------------------------------------------------------------

def get_latest_blob_name() -> str:
    """Return the name of the most recently modified wiki blob in the container."""
    client = BlobServiceClient.from_connection_string(STORAGE_CONN)
    container_client = client.get_container_client(CONTAINER)

    blobs = [b for b in container_client.list_blobs() if b.name.endswith(".jsonl")]
    if not blobs:
        raise FileNotFoundError(f"No .jsonl blobs found in container '{CONTAINER}'")

    latest = max(blobs, key=lambda b: b.last_modified)
    log.info("Latest blob: %s (modified %s)", latest.name, latest.last_modified)
    return latest.name


def download_blob(blob_name: str) -> list[dict]:
    """Download a JSONL blob and return its records as a list of dicts."""
    client = BlobServiceClient.from_connection_string(STORAGE_CONN)
    blob_client = client.get_blob_client(container=CONTAINER, blob=blob_name)

    data = blob_client.download_blob().readall().decode("utf-8")
    records = [json.loads(line) for line in data.splitlines() if line.strip()]
    log.info("Loaded %d records from %s", len(records), blob_name)
    return records


# ---------------------------------------------------------------------------
# Parsing & cleaning
# ---------------------------------------------------------------------------

def html_to_clean_text(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style noise
    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    # Replace block-level tags with newlines to preserve structure
    for tag in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr"]):
        tag.insert_before("\n")

    text = soup.get_text(separator=" ")

    # Normalize whitespace: collapse multiple spaces, keep single newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split text into (heading, body) tuples by Markdown headings.
    Falls back to a single section if there are no headings.
    """
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_body: list[str] = []

    for line in lines:
        if re.match(r"^#{1,6}\s+", line):
            if current_body or current_heading:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line.lstrip("#").strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body or current_heading:
        sections.append((current_heading, "\n".join(current_body).strip()))

    return sections if sections else [("", text)]


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split a long text block into smaller chunks at paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current:
        chunks.append(current.strip())

    return chunks or [text]


# ---------------------------------------------------------------------------
# Main transform
# ---------------------------------------------------------------------------

def page_to_chunks(record: dict) -> list[dict]:
    """
    Convert a single crawler record into one or more RAG-ready chunks.
    Uses Markdown if available (richer structure), falls back to HTML.
    """
    path: str = record.get("path", "/")
    url: str = record.get("remote_url", "")
    crawled_at: str = record.get("crawled_at", "")
    md: str = record.get("markdown", "")
    html: str = record.get("html", "")

    # Prefer markdown (already structured); fall back to cleaned HTML
    base_text = md.strip() if md.strip() else html_to_clean_text(html)

    if not base_text:
        return []

    sections = split_into_sections(base_text)
    chunks: list[dict] = []

    for section_idx, (heading, body) in enumerate(sections):
        if not body.strip():
            continue

        sub_chunks = chunk_text(body)
        for chunk_idx, chunk_text_content in enumerate(sub_chunks):
            chunk_id = f"{path.strip('/').replace('/', '__')}__{section_idx}_{chunk_idx}"
            chunks.append({
                "id": chunk_id,
                "text": f"# {heading}\n\n{chunk_text_content}".strip() if heading else chunk_text_content,
                "metadata": {
                    "path": path,
                    "url": url,
                    "heading": heading,
                    "section_index": section_idx,
                    "chunk_index": chunk_idx,
                    "crawled_at": crawled_at,
                },
            })

    return chunks


def load_rag_documents(blob_name: str | None = None) -> list[dict]:
    """
    Main entry point for the RAG pipeline.
    Downloads the most recent blob (or a specific one) and returns
    a flat list of clean, chunked documents ready for embedding.
    """
    if blob_name is None:
        blob_name = get_latest_blob_name()

    records = download_blob(blob_name)

    all_chunks: list[dict] = []
    for record in records:
        all_chunks.extend(page_to_chunks(record))

    log.info("Produced %d RAG chunks from %d pages", len(all_chunks), len(records))
    return all_chunks


# ---------------------------------------------------------------------------
# CLI: run standalone to inspect output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    docs = load_rag_documents()

    print(f"\nTotal chunks: {len(docs)}")
    print("\n--- Sample chunk ---")
    if docs:
        sample = docs[0]
        print(f"ID:       {sample['id']}")
        print(f"Metadata: {sample['metadata']}")
        print(f"Text preview:\n{sample['text'][:400]}")
