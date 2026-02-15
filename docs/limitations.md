# Limitations

Understanding the constraints and trade-offs of the current system.

---

## ArXiv-Only Source

### Current State

- **Only ArXiv papers are supported** as primary sources
- Cannot fetch papers from IEEE, ACM, Springer, PubMed, or other academic databases
- No access to commercial journals or paywalled content

### Why This Matters

- **Limited Coverage**: Many important papers are not on ArXiv (especially older works, industry research, medical journals)
- **Recency Bias**: ArXiv focuses on preprints, which may not be peer-reviewed
- **Domain Gaps**: Medicine, biology, and some engineering fields have less ArXiv coverage

### ArXiv API Constraints

- **Rate Limits**: No authentication → basic rate limiting
- **Keyword Search Only**: Results sorted by basic keyword matching, not semantic relevance
- **No Full-Text Search**: Can only search titles, abstracts, authors, categories
- **Metadata Only**: API returns metadata; PDFs must be downloaded separately

### PDF Extraction Quality

- **Heavily formatted papers**: Tables, graphs, and complex layouts often extracted poorly by PyPDF2
- **Mathematical notation**: Equations frequently garbled or unreadable
- **Figures**: Images and diagrams completely lost
- **Multi-column layouts**: Column order sometimes mixed up

---

## ScaleDown API Constraints

### Compression-Only Service

- ScaleDown is **not an LLM** — it cannot generate free-form answers
- The "fallback generation" is really **compressed extraction**, not true generation
- Cannot answer questions that require reasoning beyond the provided context

### Compression Quality

- **Very short texts** (<200 chars): Skipped, no compression applied
- **Highly technical content**: May lose nuance when compressed 40-60%
- **Query dependency**: Compression quality depends on how well the user question captures their intent

### Latency

- Each API call adds **1-3 seconds** of latency
- Multiple compression calls (context + artifacts) → 5-10s total
- No batching → sequential calls

### Cost

- **Requires a valid API key** — no free tier
- Usage-based pricing (per token compressed)

---

## Gemini Free Tier Limitations

### Rate Limits

- The free Gemini API has strict **requests per minute** and **tokens per day** limits
- Heavy usage triggers **429 errors**
- Each question with full pipeline = 3-4 Gemini API calls (triage, COT, verify, critique)

### Model Capability

- **Gemini 2.5 Flash**: Fast but not as capable as Pro models for complex multi-hop reasoning
- **Thinking Budget**: The `thinkingConfig` parameter caps internal reasoning, potentially reducing quality on highly complex questions
- **Citation Accuracy**: Even with strict prompts, the model sometimes hallucinates citations or misattributes sources

### Context Window

- While technically large (1M+ tokens), the effective context is limited by:
  - Cost (more tokens = higher API cost)
  - Quality degradation with very long contexts
  - Latency (longer contexts → slower responses)

---

## RAG Limitations

### TF-IDF Retrieval

- **Keyword-based**, not semantic
- Misses relevant chunks that use different terminology (synonym problem)
- Example: Query "neural nets" won't match "artificial neural networks" unless both terms appear

### Fixed Chunk Sizes

- **No respect for document structure** — chunks may cut through:
  - Sentences
  - Paragraphs
  - Tables
  - Equations
  - Section boundaries
- Context fragmentation can break semantic meaning

### No Re-Ranking

- Retrieved chunks are scored **solely by TF-IDF cosine similarity**
- No cross-encoder or LLM-based re-ranking is applied
- First-stage retrieval is final — no second-pass refinement

### Source Tracking

- Citations are at the **paper level**, not page/section level
- Example: `[arxiv:1706.03762]` — but which part of the paper?
- No automatic extraction of section/page metadata from chunks

---

## General Limitations

### No Real-Time Data

- Only papers already on ArXiv
- No preprint servers (bioRxiv, medRxiv, SSRN, etc.)
- No blogs, conference talks, or live research

### Single Language

- **English papers only**
- No multilingual support
- Papers in other languages will be extracted but likely produce poor results

### No Figure/Image Analysis

- Extracted text doesn't include figures or diagrams
- Cannot answer questions like "What does Figure 3 show?"
- No vision model integration

### Session State

- Sessions stored as **JSON files on disk**, not in a database
- No multi-user support
- No cloud synchronization
- Sessions lost if files are deleted

### No Evaluation Framework

- No automated hallucination detection
- No quantified quality metrics
- No benchmark datasets
- Manual verification required

---

## Performance Limitations

### Latency

**Full pipeline with paper discovery:**
- Triage + keyword extraction: ~2s
- ArXiv search + PDF download: ~10-20s (parallel)
- Text extraction + chunking + indexing: ~5s
- Retrieval + compression: ~3s
- COT generation: ~10-15s
- Verification: ~5-8s
- Critique: ~5-8s
- **Total: ~45-65 seconds**

**Direct answer (general question):**
- Triage: ~2s
- Direct generation: ~5s
- **Total: ~7 seconds**

### Throughput

- Single-threaded execution (no parallel Gemini calls)
- No response streaming (wait for full response)
- Rate limits restrict concurrent users

---

## Security & Privacy

### API Keys in `.env`

- Keys stored in plain text
- No encryption at rest
- Accidental git commits expose keys (mitigated by `.gitignore`)

### No Authentication

- CLI tool has no user authentication
- Anyone with file system access can:
  - View sessions
  - Read artifacts
  - Use your API keys

### Data Storage

- Papers, artifacts, sessions stored locally
- No data encryption
- No automatic cleanup of old data

---

## Next: Possible Improvements

See **[Improvements](improvements.md)** for ideas to address these limitations.
