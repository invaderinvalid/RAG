"""
Fetch and extract text from academic papers via the ArXiv API.

Uses the ArXiv Atom API for search/metadata and downloads PDFs for text
extraction.  Only ArXiv papers are supported.
"""

import io
import re
import xml.etree.ElementTree as ET
import requests
from pathlib import Path
from typing import Optional

from PyPDF2 import PdfReader

from src.core.config import PAPERS_DIR

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_PDF_BASE = "https://arxiv.org/pdf/"

# Atom / OpenSearch namespace
ATOM_NS = "{http://www.w3.org/2005/Atom}"


class ArXivPaper:
    """Parsed metadata for a single ArXiv paper."""

    def __init__(
        self,
        arxiv_id: str,
        title: str,
        authors: list[str],
        summary: str,
        pdf_url: str,
        published: str,
        categories: list[str],
    ):
        self.arxiv_id = arxiv_id
        self.title = title
        self.authors = authors
        self.summary = summary
        self.pdf_url = pdf_url
        self.published = published
        self.categories = categories

    def __repr__(self) -> str:
        return f"ArXivPaper({self.arxiv_id!r}, {self.title!r})"


class PaperFetcher:
    """Search, download, and cache ArXiv papers via the public API."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or PAPERS_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ──────────────────────────────────────────────────────── #

    def search(
        self,
        query: str,
        max_results: int = 5,
        sort_by: str = "relevance",   # relevance | lastUpdatedDate | submittedDate
        sort_order: str = "descending",
    ) -> list[ArXivPaper]:
        """Search ArXiv and return parsed paper metadata."""
        # If the query already contains ArXiv field prefixes, use as-is
        has_prefix = any(f"{p}:" in query for p in ("all", "ti", "au", "abs", "cat", "co"))
        search_q = query if has_prefix else f"all:{query}"
        params = {
            "search_query": search_q,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        return self._parse_feed(resp.text)

    def fetch_by_id(self, arxiv_id: str, force: bool = False) -> str:
        """
        Fetch a paper by ArXiv ID (e.g. '2511.14362'), cache locally,
        and return extracted text.
        """
        arxiv_id = self._normalise_id(arxiv_id)
        text_path = self.cache_dir / f"{arxiv_id.replace('/', '_')}.txt"

        if text_path.exists() and not force:
            return text_path.read_text(encoding="utf-8")

        # Get metadata first for structured header
        meta = self._fetch_meta(arxiv_id)

        # Download PDF
        pdf_url = f"{ARXIV_PDF_BASE}{arxiv_id}"
        pdf_bytes = self._download_pdf(pdf_url)

        pdf_path = self.cache_dir / f"{arxiv_id.replace('/', '_')}.pdf"
        pdf_path.write_bytes(pdf_bytes)

        # Extract and prepend metadata header
        body = self._extract_text(pdf_bytes)
        header = self._format_header(meta) if meta else ""
        text = f"{header}\n{body}"

        text_path.write_text(text, encoding="utf-8")
        return text

    def fetch_all(self, arxiv_ids: list[str], force: bool = False) -> dict[str, str]:
        """Fetch multiple papers by ArXiv ID, return {id: text} mapping."""
        return {aid: self.fetch_by_id(aid, force=force) for aid in arxiv_ids}

    # ── ArXiv API helpers ───────────────────────────────────────────────── #

    def _fetch_meta(self, arxiv_id: str) -> Optional[ArXivPaper]:
        """Fetch metadata for a single paper via id_list."""
        try:
            resp = requests.get(
                ARXIV_API_URL,
                params={"id_list": arxiv_id, "max_results": 1},
                timeout=30,
            )
            resp.raise_for_status()
            papers = self._parse_feed(resp.text)
            return papers[0] if papers else None
        except Exception:
            return None

    @staticmethod
    def _parse_feed(xml_text: str) -> list[ArXivPaper]:
        """Parse ArXiv Atom XML feed into ArXivPaper objects."""
        root = ET.fromstring(xml_text)
        papers: list[ArXivPaper] = []

        for entry in root.findall(f"{ATOM_NS}entry"):
            # Extract ArXiv ID from the <id> URL
            raw_id = entry.findtext(f"{ATOM_NS}id", "")
            arxiv_id = raw_id.replace("http://arxiv.org/abs/", "").strip()
            # Strip version suffix (e.g. v1)
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

            title = entry.findtext(f"{ATOM_NS}title", "").strip()
            title = re.sub(r"\s+", " ", title)  # collapse whitespace

            summary = entry.findtext(f"{ATOM_NS}summary", "").strip()

            authors = [
                a.findtext(f"{ATOM_NS}name", "").strip()
                for a in entry.findall(f"{ATOM_NS}author")
            ]

            published = entry.findtext(f"{ATOM_NS}published", "")

            # PDF link
            pdf_url = ""
            for link in entry.findall(f"{ATOM_NS}link"):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break
            if not pdf_url:
                pdf_url = f"{ARXIV_PDF_BASE}{arxiv_id}"

            # Categories
            categories = [
                c.get("term", "")
                for c in entry.findall(
                    "{http://arxiv.org/schemas/atom}primary_category"
                )
            ]

            papers.append(ArXivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                summary=summary,
                pdf_url=pdf_url,
                published=published,
                categories=categories,
            ))

        return papers

    # ── PDF helpers ─────────────────────────────────────────────────────── #

    @staticmethod
    def _download_pdf(url: str) -> bytes:
        resp = requests.get(url, timeout=60, headers={
            "User-Agent": "ScientificLitExplorer/1.0",
        })
        resp.raise_for_status()
        return resp.content

    @staticmethod
    def _extract_text(pdf_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(f"--- Page {i + 1} ---\n{text}")
        return "\n\n".join(pages)

    @staticmethod
    def _format_header(paper: ArXivPaper) -> str:
        authors = ", ".join(paper.authors[:5])
        if len(paper.authors) > 5:
            authors += " et al."
        return (
            f"# {paper.title}\n\n"
            f"**ArXiv ID:** {paper.arxiv_id}  \n"
            f"**Authors:** {authors}  \n"
            f"**Published:** {paper.published[:10]}  \n"
            f"**Categories:** {', '.join(paper.categories)}  \n"
            f"**Summary:** {paper.summary[:300]}…\n\n"
            f"---\n"
        )

    @staticmethod
    def _normalise_id(raw: str) -> str:
        """
        Accept various ArXiv ID formats and return the bare ID.
        e.g. 'https://arxiv.org/pdf/2511.14362' → '2511.14362'
             'arxiv:2511.14362v2'                → '2511.14362'
        """
        raw = raw.strip()
        for prefix in (
            "https://arxiv.org/abs/",
            "https://arxiv.org/pdf/",
            "http://arxiv.org/abs/",
            "http://arxiv.org/pdf/",
        ):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break
        raw = re.sub(r"^arxiv:", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"(\.pdf)?(v\d+)?$", "", raw)
        return raw.strip("/")
