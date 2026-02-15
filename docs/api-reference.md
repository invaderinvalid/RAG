# API Reference

Complete reference for environment variables, CLI commands, and dependencies.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SCALEDOWN_API_KEY` | ✅ Yes | — | Your ScaleDown API key from [ScaleDown](https://blog.scaledown.ai/blog/getting-started) |
| `GEMINI_API_KEY` | ✅ Yes | — | Your Google Gemini API key from [AI Studio](https://aistudio.google.com/apikey) |
| `SCALEDOWN_MODEL` | No | `gemini-2.5-flash` | Target model for compression optimization |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model for generation |
| `SCALEDOWN_TIMEOUT` | No | `15` | Timeout (seconds) for ScaleDown API calls |
| `CHUNK_SIZE` | No | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | No | `200` | Overlap between adjacent chunks |
| `TOP_K` | No | `5` | Number of chunks to retrieve per query |

---

## CLI Commands

### `ask` — Research Questions

```bash
python -m src.main ask "question" [--session SESSION_ID]
```

**Arguments:**
- `question` (required): The research question to answer
- `--session SESSION_ID` (optional): Continue an existing session

**What it does:**
1. Triage question complexity
2. Discover and download relevant papers (if research question)
3. Chunk, index, and retrieve relevant content
4. Compress context via ScaleDown
5. Run full reasoning pipeline (COT → Verify → Critique)
6. Save session for follow-ups

**Example:**
```bash
python -m src.main ask "What are the latest advances in neural architecture search?"
python -m src.main ask "How does this compare to random search?" --session abc123
```

---

### `papers` — Interactive Paper Explorer

```bash
python -m src.main papers "search_query"
```

**Arguments:**
- `search_query` (required): ArXiv search query

**Interactive commands:**
- Type text: Ask a question about the selected paper
- Type number: Switch to a different paper
- `back`: Return to paper list
- `list`: Show paper list again
- `s`: New search
- `q`: Quit

**What it does:**
- Search ArXiv and display results
- Let you select a paper
- Answer questions using full pipeline
- Maintain session across all interactions
- Cache papers (no refetching)

**Example:**
```bash
python -m src.main papers "attention mechanism transformers"
```

---

### `paper` — Specific Paper Analysis

```bash
python -m src.main paper <arxiv_id> "question"
```

**Arguments:**
- `arxiv_id` (required): ArXiv ID (e.g., `1706.03762`)
- `question` (required): Question about the paper

**What it does:**
1. Download paper (if not cached)
2. Extract and index text
3. Run full pipeline with paper-specific grounding
4. Answer using ONLY information from this paper

**Example:**
```bash
python -m src.main paper 1706.03762 "What is the multi-head attention mechanism?"
```

---

### `search` — Quick ArXiv Search

```bash
python -m src.main search "query"
```

**Arguments:**
- `query` (required): ArXiv search query

**What it does:**
- Search ArXiv and display top 10 results
- Show titles, authors, abstracts
- No download or analysis

**Example:**
```bash
python -m src.main search "graph neural networks"
```

---

### `sessions` — List Conversations

```bash
python -m src.main sessions
```

**What it does:**
- List all saved session IDs
- Show creation timestamps
- Display number of Q&A turns
- List papers in each session

**Example output:**
```
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Session ID ┃ Created            ┃ Turns ┃ Papers             ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ abc123     │ 2024-01-15 10:30   │ 3     │ arxiv:1706.03762   │
│ def456     │ 2024-01-15 14:22   │ 1     │ arxiv:2103.14030   │
└────────────┴────────────────────┴───────┴────────────────────┘
```

---

### `workflow show` — Display Pipeline Config

```bash
python -m src.main workflow show
```

**What it does:**
- Show all pipeline stages
- Display enabled/disabled status
- Show execution order

**Example output:**
```
Current Workflow:
1. cot (enabled) — Chain-of-thought reasoning
2. self_verify (enabled) — Citation verification
3. self_critique (enabled) — Quality evaluation
```

---

### `workflow toggle` — Enable/Disable Stages

```bash
python -m src.main workflow toggle <stage> <on|off>
```

**Arguments:**
- `stage` (required): `cot`, `self_verify`, or `self_critique`
- `on|off` (required): Enable or disable

**Example:**
```bash
python -m src.main workflow toggle self_critique off
python -m src.main workflow toggle self_verify on
```

---

### `workflow reorder` — Change Stage Order

```bash
python -m src.main workflow reorder <stage1,stage2,...>
```

**Arguments:**
- Comma-separated list of stages in desired order

**Example:**
```bash
python -m src.main workflow reorder cot,self_verify,self_critique
python -m src.main workflow reorder cot,self_critique,self_verify
```

---

### `artifacts list` — View Stored Outputs

```bash
python -m src.main artifacts list
```

**What it does:**
- List all stored reasoning artifacts
- Show artifact IDs, stages, timestamps, sizes

**Example output:**
```
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Artifact ID  ┃ Stage        ┃ Timestamp          ┃ Size  ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ abc123       │ cot          │ 2024-01-15 10:30   │ 2.3KB │
│ abc123       │ self_verify  │ 2024-01-15 10:31   │ 1.1KB │
│ abc123       │ self_critique│ 2024-01-15 10:32   │ 0.8KB │
└──────────────┴──────────────┴────────────────────┴───────┘
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `requests` | Latest | HTTP client for ScaleDown, Gemini, ArXiv APIs |
| `python-dotenv` | Latest | Load `.env` environment variables |
| `numpy` | Latest | Array operations for TF-IDF calculations |
| `scikit-learn` | Latest | TF-IDF vectorizer, cosine similarity |
| `PyPDF2` | Latest | PDF text extraction from ArXiv papers |
| `rich` | Latest | Terminal UI (tables, panels, markdown, spinners) |

**Install all:**
```bash
pip install -r requirements.txt
```

---

## Next: Project Structure

See **[Project Structure](project-structure.md)** for codebase organization.
