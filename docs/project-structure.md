# Project Structure

Complete overview of the codebase organization and file descriptions.

---

## Directory Tree

```
RAG/
├── .env                          # API keys (not in git)
├── .env.example                  # Template for API keys
├── .gitignore
├── Project.md                    # Original project specification
├── README.md                     # Main documentation
├── requirements.txt              # Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── main.py                   # CLI entry point — all commands
│   │
│   ├── core/
│   │   ├── config.py             # Central configuration (env vars, paths)
│   │   ├── gemini.py             # Gemini API client + rate-limit retry + fallback
│   │   ├── llm.py                # LLM-agnostic handler factories (COT, verify, critique)
│   │   ├── research_agent.py     # Smart discovery — triage, keywords, ArXiv, web fallback
│   │   ├── scaledown.py          # ScaleDown compression client + generation fallback
│   │   └── session.py            # Session persistence (JSON-based conversation history)
│   │
│   ├── papers/
│   │   └── fetcher.py            # ArXiv Atom API search + PDF download/extraction
│   │
│   ├── rag/
│   │   └── pipeline.py           # TF-IDF chunking, retrieval, and ScaleDown compression
│   │
│   ├── storage/
│   │   └── artifact_store.py     # Markdown artifact storage with compression
│   │
│   └── workflow/
│       └── engine.py             # Configurable reasoning pipeline (stages, toggle, reorder)
│
├── artifacts/                    # Stored COT/verify/critique markdown files
│   ├── cot/
│   ├── self_verify/
│   └── self_critique/
│
├── papers/                       # Cached PDFs and extracted text
│   ├── *.pdf
│   └── *.txt
│
├── sessions/                     # Conversation history (JSON per session)
│   └── *.json
│
└── docs/                         # MkDocs documentation
    ├── index.md
    ├── architecture.md
    ├── how-it-works.md
    ├── setup.md
    ├── configuration.md
    ├── usage.md
    ├── methodology.md
    ├── anti-hallucination.md
    ├── api-reference.md
    ├── project-structure.md
    ├── limitations.md
    └── improvements.md
```

---

## File Descriptions

### Root Files

| File | Description |
|------|-------------|
| `.env` | Contains API keys (not version controlled) |
| `.env.example` | Template for API keys with all configurable parameters |
| `.gitignore` | Excludes `.env`, `papers/`, `artifacts/`, `sessions/`, etc. |
| `Project.md` | Original project specification and requirements |
| `README.md` | Main documentation with architecture, usage, methodology |
| `requirements.txt` | Python package dependencies |
| `mkdocs.yml` | MkDocs configuration for documentation site |

---

### `src/main.py`

**CLI entry point** — defines all user-facing commands using Click:

- `ask` — Research questions with auto-discovery
- `papers` — Interactive paper explorer
- `paper` — Deep-dive into specific paper
- `search` — Quick ArXiv search
- `sessions` — List conversation history
- `workflow` — Configure reasoning pipeline
- `artifacts` — List stored outputs

---

### `src/core/` — Core Logic

#### `config.py`

Central configuration manager:
- Loads environment variables from `.env`
- Defines default values (chunk size, top-k, timeouts)
- Exports paths for `papers/`, `artifacts/`, `sessions/`
- Validates required API keys

#### `gemini.py`

Gemini API client with resilience:
- `GeminiClient`: Wraps Google Generative AI SDK
- `generate()`: Main generation with rate limit retry
- Exponential backoff: 5s, 10s, 20s, 40s, 60s
- `GeminiRateLimitError`: Raised after 5 failed retries
- Supports `thinkingConfig` for thinking budget
- Configurable `temperature` and `max_tokens`

#### `llm.py`

LLM-agnostic handler factories:
- `create_cot_handler()`: Chain-of-thought with strict citations
- `create_verify_handler()`: Citation verification
- `create_critique_handler()`: Quality evaluation
- `create_direct_handler()`: General question answering
- Each handler encapsulates system prompts and generation config

#### `research_agent.py`

Smart paper discovery:
- `discover()`: Main entry point
- Question classification: general/conceptual/research
- Keyword extraction (Gemini or heuristic fallback)
- ArXiv search integration
- Parallel PDF download
- Web search fallback (placeholder)

#### `scaledown.py`

ScaleDown compression client:
- `compress_context()`: Main compression endpoint
- `compress_artifact()`: Compress reasoning outputs
- `generate_compressed()`: Fallback generation when Gemini is rate-limited
- Automatic retry on transient errors
- Configurable timeout

#### `session.py`

Session persistence manager:
- `Session`: Class representing a conversation
- `save()`: Write session to JSON
- `load()`: Read session from JSON
- `add_turn()`: Append Q&A pair
- Tracks papers, artifacts, timestamps
- Auto-creates `sessions/` directory

---

### `src/papers/` — Paper Retrieval

#### `fetcher.py`

ArXiv integration:
- `PaperFetcher`: Main class
- `search()`: Query ArXiv Atom API
- `download_pdf()`: Fetch PDF from ArXiv
- `extract_text()`: PyPDF2 extraction
- Parallel downloads via `ThreadPoolExecutor`
- Caching (checks `papers/` before downloading)
- Auto-creates `papers/` directory

---

### `src/rag/` — Retrieval & Compression

#### `pipeline.py`

RAG pipeline implementation:
- `RAGPipeline`: Main class
- `chunk_text()`: Split text with overlap
- `build_index()`: TF-IDF vectorization
- `retrieve()`: Cosine similarity search
- `compress_chunks()`: ScaleDown compression
- Source tracking (every chunk labeled with `arxiv:ID`)

---

### `src/storage/` — Artifact Management

#### `artifact_store.py`

Persistent storage for reasoning outputs:
- `ArtifactStore`: Main class
- `save()`: Write markdown artifact
- `load()`: Read artifact
- `compress()`: ScaleDown compression before storage
- Metadata tracking (timestamps, token counts, compression stats)
- Organized by stage: `cot/`, `self_verify/`, `self_critique/`
- Auto-creates `artifacts/` directory

---

### `src/workflow/` — Pipeline Orchestration

#### `engine.py`

Configurable reasoning pipeline:
- `WorkflowEngine`: Main class
- `execute()`: Run all enabled stages in order
- `toggle_stage()`: Enable/disable stages
- `reorder()`: Change execution order
- Stages: COT, self_verify, self_critique
- Config persistence (saved to `workflow.json`)
- Rich progress display

---

## Runtime Directories

### `papers/`

Cached paper files:
- `{arxiv_id}.pdf` — Downloaded PDFs
- `{arxiv_id}.txt` — Extracted text

**Benefits:**
- No redundant downloads
- Instant follow-up questions
- Works offline (once papers are cached)

### `artifacts/`

Stored reasoning outputs organized by stage:
- `cot/{id}.md` — Chain-of-thought analysis
- `self_verify/{id}.md` — Citation verification tables
- `self_critique/{id}.md` — Quality critiques

**Benefits:**
- Audit trail of reasoning
- Debug verification failures
- Reuse previous analyses

### `sessions/`

Conversation history:
- `{session_id}.json` — All Q&A turns, papers, metadata

**Benefits:**
- Multi-turn conversations
- Context across questions
- Resume previous sessions

---

## Next: Limitations

See **[Limitations](limitations.md)** for known constraints and trade-offs.
