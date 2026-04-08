"""
Azure DevOps Wiki Crawler
Crawls all pages of a wiki via the REST API and stores them in Azure Blob Storage
as a single JSONL file (one JSON object per line).

Setup:
  1. Copy .env.example to .env and fill in your credentials.
  2. pip install -r requirements.txt
  3. python crawler.py
"""

import json
import base64
import logging
import os
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import markdown
import requests
from bs4 import BeautifulSoup
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PAT = os.environ["AZURE_DEVOPS_PAT"]
ORG = os.environ.get("DEVOPS_ORG", "ulpsi")
PROJECT = os.environ.get("DEVOPS_PROJECT", "NetProjects10")
WIKI = os.environ.get("DEVOPS_WIKI", "NetProjects10.wiki")
STORAGE_CONN = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER", "wiki-crawl")

API_BASE = f"https://dev.azure.com/{ORG}/{PROJECT}/_apis/wiki/wikis/{WIKI}"
API_VERSION = "7.1"

# Basic-auth header: PAT is used as the password, username can be anything
_encoded = base64.b64encode(f":{PAT}".encode()).decode()
HEADERS = {
    "Authorization": f"Basic {_encoded}",
    "Content-Type": "application/json",
}

# ---------------------------------------------------------------------------
# Azure DevOps REST API helpers
# ---------------------------------------------------------------------------

def list_all_pages() -> list[dict]:
    """Return a flat list of all wiki page metadata (path, url, etc.)."""
    url = f"{API_BASE}/pages"
    params = {
        "path": "/",
        "recursionLevel": "full",
        "includeContent": "false",
        "api-version": API_VERSION,
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    data = resp.json()
    pages = []
    _collect_pages(data, pages)
    return pages


def _collect_pages(node: dict, acc: list[dict]) -> None:
    """Recursively walk the page tree returned by the API."""
    acc.append({
        "path": node.get("path", "/"),
        "remote_url": node.get("remoteUrl", ""),
        "order": node.get("order", 0),
    })
    for child in node.get("subPages", []):
        _collect_pages(child, acc)


def fetch_page_content(path: str) -> dict:
    """
    Fetch a single wiki page and return a structured record.
    The API returns Markdown; we convert it to HTML and extract links.
    """
    url = f"{API_BASE}/pages"
    params = {
        "path": path,
        "includeContent": "true",
        "api-version": API_VERSION,
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    data = resp.json()

    md_content: str = data.get("content", "")
    html_content: str = markdown.markdown(
        md_content,
        extensions=["extra", "toc", "tables", "fenced_code"],
    )

    links = _extract_links(html_content, base_path=path)

    return {
        "path": path,
        "remote_url": data.get("remoteUrl", ""),
        "html": html_content,
        "markdown": md_content,
        "links": links,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }


def _extract_links(html: str, base_path: str) -> list[str]:
    """Extract all hrefs from the HTML, resolving relative wiki paths."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    wiki_base = f"https://dev.azure.com/{ORG}/{PROJECT}/_wiki/wikis/{WIKI}"
    for tag in soup.find_all("a", href=True):
        href: str = tag["href"]
        if href.startswith("http"):
            links.append(href)
        elif href.startswith("/"):
            links.append(f"{wiki_base}{href}")
        else:
            # relative to current page path
            parent = "/".join(base_path.rstrip("/").split("/")[:-1]) or "/"
            links.append(f"{wiki_base}{parent}/{href}")
    return list(dict.fromkeys(links))  # deduplicate, preserve order


# ---------------------------------------------------------------------------
# Azure Blob Storage helper
# ---------------------------------------------------------------------------

def upload_jsonl(records: list[dict], blob_name: str) -> str:
    """Upload a list of dicts as a JSONL blob. Returns the blob URL."""
    client = BlobServiceClient.from_connection_string(STORAGE_CONN)
    container_client = client.get_container_client(CONTAINER)

    # Create container if it doesn't exist
    try:
        container_client.create_container()
        log.info("Created container: %s", CONTAINER)
    except Exception:
        pass  # already exists

    jsonl_bytes = "\n".join(json.dumps(r, ensure_ascii=False) for r in records).encode("utf-8")

    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(jsonl_bytes, overwrite=True)

    return blob_client.url


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Listing all wiki pages for %s/%s/%s ...", ORG, PROJECT, WIKI)
    pages = list_all_pages()
    log.info("Found %d pages", len(pages))

    records: list[dict] = []
    failed: list[str] = []

    for i, page in enumerate(pages, 1):
        path = page["path"]
        log.info("[%d/%d] Fetching: %s", i, len(pages), path)
        try:
            record = fetch_page_content(path)
            records.append(record)
        except Exception as exc:
            log.warning("Failed to fetch %s: %s", path, exc)
            failed.append(path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    blob_name = f"wiki_{ORG}_{PROJECT}_{timestamp}.jsonl"

    log.info("Uploading %d records to blob: %s/%s", len(records), CONTAINER, blob_name)
    url = upload_jsonl(records, blob_name)
    log.info("Done. Blob URL: %s", url)

    if failed:
        log.warning("Failed pages (%d): %s", len(failed), failed)


if __name__ == "__main__":
    main()
