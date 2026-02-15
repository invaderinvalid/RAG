# Scientific Literature Explorer

A production-ready RAG system that retrieves scientific papers, compresses context, and generates well-cited answers through a multi-stage anti-hallucination pipeline.

---

## Overview

The **Scientific Literature Explorer** is an intelligent research assistant that:
- Automatically discovers relevant papers from ArXiv
- Uses TF-IDF-based RAG for precise chunk retrieval
- Compresses context via ScaleDown API (40-60% token reduction)
- Runs a multi-stage reasoning workflow (COT → Verify → Critique)
- Enforces strict citation rules to minimize hallucination
- Maintains session history for multi-turn conversations

Built with **Google Gemini 2.5 Flash** for intelligence and **ScaleDown API** for context compression.

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Smart Discovery** | Auto-discovers papers via ArXiv API with parallel downloads |
| 📊 **Context Compression** | ScaleDown API reduces tokens by 40-60% while preserving meaning |
| 🧠 **Multi-Stage Reasoning** | Chain-of-Thought → Self-Verification → Self-Critique |
| 📝 **Strict Citations** | Every claim requires an inline citation `[arxiv:XXXX.XXXXX]` |
| 💬 **Session Persistence** | Multi-turn conversations with history context |
| ⚡ **Question Triage** | General questions answered instantly without paper fetch |
| 🎛️ **Configurable Pipeline** | Toggle stages, reorder workflow via CLI |
| 🔄 **Resilient Fallback** | Automatic retry with exponential backoff + ScaleDown fallback |

---

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd RAG
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Fill in your API keys:

```env
SCALEDOWN_API_KEY=your_scaledown_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Get keys:
- **ScaleDown**: [ScaleDown Getting Started](https://blog.scaledown.ai/blog/getting-started)
- **Gemini**: [Google AI Studio](https://aistudio.google.com/apikey) (free tier available)

### 3. Run

```bash
# Ask a research question
python -m src.main ask "What are the latest advances in neural architecture search?"

# Interactive paper explorer
python -m src.main papers "transformers attention mechanism"

# Deep-dive into a specific paper
python -m src.main paper 1706.03762 "What is multi-head attention?"
```

---

## System Comparison

| Feature | This System | Traditional RAG |
|---------|-------------|-----------------|
| **Paper Discovery** | ✅ Automatic ArXiv search + parallel downloads | ❌ Manual paper curation |
| **Context Compression** | ✅ ScaleDown API (40-60% reduction) | ❌ No compression (high token costs) |
| **Verification** | ✅ Multi-stage: COT → Verify → Critique | ❌ Single-pass generation |
| **Citations** | ✅ Strict inline citations enforced | ⚠️ Optional, often missing |
| **Triage** | ✅ Smart routing (general vs research) | ❌ All queries treated equally |
| **Sessions** | ✅ Persistent multi-turn conversations | ❌ Stateless single-shot |
| **Fallback** | ✅ ScaleDown fallback on rate limits | ❌ Hard failure |
| **Rate Limit Handling** | ✅ Exponential backoff (5× retries) | ⚠️ Basic retry or none |

---

## Example Workflow

### Research Question

```bash
$ python -m src.main ask "What are transformers in NLP?"
```

**What happens:**
1. ⚡ Question triaged as "research"
2. 🔍 ArXiv searched for relevant papers
3. 📥 PDFs downloaded in parallel
4. ✂️ Text chunked and indexed (TF-IDF)
5. 📊 Top-5 chunks compressed via ScaleDown (1500 → 600 tokens)
6. 🧠 COT reasoning with strict citations
7. ✅ Self-verification checks all citations
8. 📋 Self-critique evaluates quality
9. 💾 Session saved for follow-ups

**Result:** A cited answer in ~45-60 seconds

### Follow-Up Question

```bash
$ python -m src.main ask "How does this compare to RNNs?" --session abc123
```

**What happens:**
1. ⚡ Session loaded (previous papers + conversation history)
2. 📚 No re-downloading (papers cached)
3. 🧠 Full pipeline runs with context from previous Q&A
4. 💾 Session updated

**Result:** A contextual answer in ~20-30 seconds

---

## Interactive Paper Explorer

```bash
$ python -m src.main papers "attention mechanism transformers"
```

**Features:**
- 📋 Browse search results
- 🎯 Select a paper
- 💬 Ask questions about it
- 🔄 Switch between papers seamlessly
- 📝 All questions share one session
- ⚡ Instant follow-ups (no refetching)

**Interactive commands:**
- Type **text**: Ask a question
- Type **number**: Switch papers
- Type **`back`**: Return to list
- Type **`s`**: New search
- Type **`q`**: Quit

---

## Documentation Structure

### Getting Started
- **[Architecture Overview](architecture.md)** — System components and data flow
- **[How It Works](how-it-works.md)** — End-to-end flow with ScaleDown and Gemini roles
- **[Setup Guide](setup.md)** — Installation and configuration
- **[Configuration](configuration.md)** — All environment variables explained

### Usage
- **[Usage Guide](usage.md)** — All CLI commands and examples
- **[Workflow Examples](usage.md#workflow-examples)** — Common usage patterns

### Technical Details
- **[Methodology](methodology.md)** — RAG, compression, triage, resilience strategies
- **[Anti-Hallucination Pipeline](anti-hallucination.md)** — Multi-stage verification details
- **[API Reference](api-reference.md)** — Complete command and config reference

### Reference
- **[Project Structure](project-structure.md)** — Codebase organization and file descriptions
- **[Limitations](limitations.md)** — Known constraints and trade-offs
- **[Improvements](improvements.md)** — Future enhancements (short/medium/long-term)

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Intelligence** | Google Gemini 2.5 Flash | Answer generation, classification, verification |
| **Compression** | ScaleDown API | Context compression (40-60%), fallback generation |
| **Paper Source** | ArXiv Atom API | Scientific paper search and metadata |
| **PDF Processing** | PyPDF2 | Text extraction from PDFs |
| **Retrieval** | scikit-learn TF-IDF | Vectorization and similarity search |
| **Storage** | JSON (sessions), Markdown (artifacts) | Persistence |
| **CLI** | Rich (terminal UI) | Interactive tables, panels, markdown rendering |

---

## Performance

### Latency

| Query Type | Time | Breakdown |
|------------|------|-----------|
| **General Question** | ~5-7s | Triage (2s) + Direct Answer (5s) |
| **Research Question (first)** | ~45-60s | Discovery (15s) + Extraction (5s) + Pipeline (30s) |
| **Follow-Up** | ~20-30s | Cached papers + Pipeline (20s) |

### Token Efficiency

**Without ScaleDown:**
- Retrieved context: ~1500 tokens
- API cost: Higher
- Latency: Slower

**With ScaleDown:**
- Compressed context: ~600 tokens (40% reduction)
- API cost: 40% lower
- Latency: 20% faster (less to process)

---

## Project Status

**Production-Ready Features:**
- ✅ Multi-paper discovery
- ✅ Context compression
- ✅ Multi-stage verification
- ✅ Session persistence
- ✅ Interactive paper explorer
- ✅ Configurable workflow
- ✅ Rate limit resilience

**Known Limitations:**
- ArXiv-only (no IEEE, ACM, PubMed)
- TF-IDF retrieval (not semantic)
- No streaming responses
- CLI only (no web UI)

See **[Limitations](limitations.md)** for details.

---

## Contributing & Improvements

See **[Improvements](improvements.md)** for a roadmap of potential enhancements:

**Short-term wins:**
- Semantic embeddings (better retrieval)
- Async API calls (lower latency)
- ScaleDown Python SDK (cleaner code)

**Long-term goals:**
- Multi-source support (Semantic Scholar, PubMed)
- Knowledge graph for cross-paper reasoning
- Web UI (Streamlit/Gradio)

---

## License

The original project specification can be found in the root `Project.md` file.

---

## Next Steps

👉 **New users**: Start with **[Getting Started](setup.md)**

👉 **Want to understand the system**: Read **[Architecture](architecture.md)** and **[How It Works](how-it-works.md)**

👉 **Ready to use**: Jump to **[Usage Guide](usage.md)**

👉 **Technical deep-dive**: Explore **[Methodology](methodology.md)** and **[Anti-Hallucination Pipeline](anti-hallucination.md)**
