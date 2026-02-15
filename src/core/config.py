"""Central configuration for the Scientific Literature Explorer."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────── #
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PAPERS_DIR = PROJECT_ROOT / "papers"

# ── ScaleDown ──────────────────────────────────────────────────────────────── #
SCALEDOWN_URL = "https://api.scaledown.xyz/compress/raw/"
SCALEDOWN_API_KEY = os.getenv("SCALEDOWN_API_KEY", "")
SCALEDOWN_MODEL = os.getenv("SCALEDOWN_MODEL", "gemini-2.5-flash")
SCALEDOWN_TIMEOUT = int(os.getenv("SCALEDOWN_TIMEOUT", "15"))

# ── ArXiv ──────────────────────────────────────────────────────────────────── #
ARXIV_API_URL = "https://export.arxiv.org/api/query"

# ── Gemini (answer generation) ─────────────────────────────────────────────── #
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── RAG ────────────────────────────────────────────────────────────────────── #
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))        # chars per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))   # overlap between chunks
TOP_K = int(os.getenv("TOP_K", "5"))                     # chunks to retrieve
