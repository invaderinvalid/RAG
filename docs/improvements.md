# Possible Improvements

Ideas for enhancing the system, organized by implementation complexity.

---

## Short-Term 

These improvements can be implemented quickly with high impact.

| Improvement | Impact | Effort | Why It Matters |
|-------------|--------|--------|----------------|
| **Semantic embeddings** (Sentence Transformers / Qwen3-Embedding) | Much better retrieval quality | Medium | TF-IDF misses semantic similarity; embeddings capture meaning |
| **Cross-encoder re-ranking** after initial TF-IDF retrieval | Higher precision top-k | Low | Second-pass ranking eliminates noise from first-stage retrieval |
| **Async HTTP calls** (aiohttp) for parallel Gemini + ScaleDown | Lower latency | Medium | Currently sequential; parallel calls could save 50% time |
| **ScaleDown Python SDK** (`pip install scaledown`) | Cleaner code, batch support, built-in retry | Low | Raw HTTP calls are verbose; SDK handles boilerplate |
| **Response caching** — cache Gemini responses by (question, context_hash) | Eliminates repeat latency | Low | Same question on same papers = instant response |
| **Better PDF extraction** — use `pymupdf` or `pdfplumber` | Better text quality, especially tables | Low | PyPDF2 struggles with complex layouts; these libraries are more robust |

---

## Medium-Term

These require more design work but have significant benefits.

| Improvement | Impact | Effort | Why It Matters |
|-------------|--------|--------|----------------|
| **ScaleDown SemanticOptimizer** | Replace TF-IDF entirely | Medium | Their FAISS-based semantic search is faster and more accurate than TF-IDF |
| **ScaleDown Pipeline** — chain HasteOptimizer → Compressor | Structured compression pipeline | Medium | Eliminates manual orchestration; built-in observability |
| **Multi-source support** — Semantic Scholar API, PubMed, IEEE Xplore | Much wider paper coverage | High | ArXiv-only limits domains (medicine, older CS papers, etc.) |
| **Streaming responses** — stream Gemini output token-by-token | Better UX for long answers | Medium | Users see progress instead of waiting 15s for full response |
| **Web UI** (Streamlit/Gradio) | Broader accessibility | Medium | CLI limits non-technical users; web UI is more approachable |
| **Configurable thinking budget** per question complexity | Better quality/speed trade-off | Low | General questions waste thinking budget; research questions need more |
| **Section-level citations** — extract page/section from chunks | More precise citations | Medium | "arxiv:1706.03762 Section 3.2" is better than just "arxiv:1706.03762" |

---

## Long-Term

These are ambitious improvements requiring substantial effort.

| Improvement | Impact | Effort | Why It Matters |
|-------------|--------|--------|----------------|
| **ScaleDown Pareto Merging** — dynamic model merging | Potentially 30% cost reduction | High | Intelligently routes queries to optimal model in cost/quality trade-off space |
| **Knowledge graph** — build citation graph across papers | Deep cross-paper analysis | High | Multi-hop reasoning: "How does paper A's findings relate to paper B's methods?" |
| **Fine-tuned embeddings** on scientific text | Domain-specific retrieval | High | Sentence Transformers trained on arXiv abstracts would outperform general models |
| **Evaluation framework** — automated hallucination detection | Quantified quality metrics | Medium | ScaleDown's evaluation pipeline can score answer quality automatically |
| **Multi-agent architecture** — separate agents for search, analysis, verification | Better specialization | High | Single LLM does everything; specialist agents could improve quality |
| **Real-time monitoring** — track latency, costs, cache hits | System observability | Medium | Currently no metrics; hard to optimize without measurement |
| **Database backend** — replace JSON files with SQLite/Postgres | Multi-user, scalability | Medium | File-based storage doesn't scale; DB enables concurrent access |

---

## Retrieval Improvements

### Semantic Embeddings

**Current:** TF-IDF (keyword-based)

**Proposed:** Sentence Transformers or Qwen3-Embedding

**Implementation:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Then use FAISS or cosine similarity
```

**Benefits:**
- Captures semantic meaning (not just keywords)
- Handles synonyms, paraphrasing
- Better cross-domain retrieval

---

### Cross-Encoder Re-Ranking

**Current:** TF-IDF scores are final

**Proposed:** Second-pass with cross-encoder

**Implementation:**
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = reranker.predict([(query, chunk) for chunk in top_k])
reranked = [chunk for _, chunk in sorted(zip(scores, top_k), reverse=True)]
```

**Benefits:**
- Higher precision (fewer irrelevant chunks)
- Better top-5 than top-20 → top-5

---

### ScaleDown SemanticOptimizer

**Current:** Custom TF-IDF pipeline

**Proposed:** Use ScaleDown's built-in semantic search

**Implementation:**
```python
from scaledown import SemanticOptimizer

optimizer = SemanticOptimizer(api_key=SCALEDOWN_API_KEY)
results = optimizer.query(
    chunks=chunks,
    query=user_question,
    top_k=5
)
```

**Benefits:**
- FAISS-backed semantic search
- Integrated with ScaleDown compression
- No need to maintain custom retrieval code

---

## Compression Improvements

### ScaleDown Pipeline

**Current:** Manual compression calls

**Proposed:** ScaleDown `Pipeline` class

**Implementation:**
```python
from scaledown import Pipeline, HasteOptimizer, Compressor

pipeline = Pipeline([
    HasteOptimizer(rate="auto"),
    Compressor(model="gemini-2.5-flash")
])

result = pipeline.run(context=raw_text, prompt=user_question)
```

**Benefits:**
- Structured, maintainable
- Built-in retries and error handling
- Observability (latency, compression stats)

---

## Multi-Source Support

**Current:** ArXiv only

**Proposed:** Add Semantic Scholar, PubMed, IEEE Xplore

### Semantic Scholar API

```python
import requests

response = requests.get(
    "https://api.semanticscholar.org/graph/v1/paper/search",
    params={"query": user_query, "limit": 10}
)
papers = response.json()["data"]
```

**Benefits:**
- 200M+ papers across all domains
- Free API with generous rate limits
- Includes citations, references, metadata

### PubMed API

```python
from Bio import Entrez

Entrez.email = "your_email@example.com"
handle = Entrez.esearch(db="pubmed", term=query, retmax=10)
results = Entrez.read(handle)
```

**Benefits:**
- Medical and life sciences papers
- High-quality, peer-reviewed

---

## User Experience Improvements

### Streaming Responses

**Current:** Wait for full response

**Proposed:** Stream tokens as they're generated

**Implementation:**
```python
for chunk in gemini_client.generate_stream(prompt):
    print(chunk, end="", flush=True)
```

**Benefits:**
- Immediate feedback
- Users can read while generation continues
- Better perceived latency

---

### Web UI

**Current:** CLI only

**Proposed:** Streamlit or Gradio web app

**Streamlit Example:**
```python
import streamlit as st

question = st.text_input("Ask a research question")
if st.button("Ask"):
    with st.spinner("Researching..."):
        answer = ask_question(question)
    st.markdown(answer)
```

**Benefits:**
- No terminal knowledge required
- Richer UI (charts, images, interactive tables)
- Shareable URLs

---

## Cost & Performance Improvements

### Response Caching

**Current:** Every question hits API

**Proposed:** Cache by (question, context_hash)

**Implementation:**
```python
import hashlib

cache = {}
key = f"{question}:{hashlib.md5(context.encode()).hexdigest()}"

if key in cache:
    return cache[key]

response = gemini_client.generate(prompt)
cache[key] = response
return response
```

**Benefits:**
- Instant responses for repeat questions
- Lower API costs
- Reduced rate limit pressure

---

### Async Parallelization

**Current:** Sequential API calls

**Proposed:** Parallel with asyncio

**Implementation:**
```python
import asyncio

async def parallel_calls():
    tasks = [
        gemini_async.generate(prompt1),
        scaledown_async.compress(context1),
        scaledown_async.compress(context2)
    ]
    return await asyncio.gather(*tasks)
```

**Benefits:**
- 50% latency reduction
- Better resource utilization

---

### ScaleDown Batching

**Current:** One API call per compression

**Proposed:** Batch multiple compressions

**Implementation:**
```python
from scaledown import batch_compress

results = batch_compress([
    {"context": chunk1, "prompt": query},
    {"context": chunk2, "prompt": query},
    {"context": chunk3, "prompt": query}
])
```

**Benefits:**
- Lower latency (1 network round-trip vs 3)
- Potential cost savings

---

## Evaluation & Monitoring

### Automated Hallucination Detection

**Current:** Manual verification

**Proposed:** ScaleDown evaluation pipeline

**Implementation:**
```python
from scaledown import evaluate_response

score = evaluate_response(
    question=user_question,
    answer=cot_answer,
    sources=retrieved_chunks
)
```

**Benefits:**
- Quantified quality metrics
- Automatic flagging of low-confidence answers
- A/B testing of prompts and models

---

### System Metrics

**Proposed:** Track latency, costs, cache hits

**Implementation:**
```python
import time

start = time.time()
response = gemini_client.generate(prompt)
latency = time.time() - start

metrics.log({"latency": latency, "tokens": len(response), "cost": calculate_cost(response)})
```

**Benefits:**
- Identify bottlenecks
- Optimize slow stages
- Cost tracking per session

---

## Next Steps

**Priority order:**
1. **ScaleDown Python SDK** (easiest, immediate code cleanup)
2. **Better PDF extraction** (`pymupdf` instead of PyPDF2)
3. **Semantic embeddings** (biggest retrieval quality gain)
4. **Response caching** (instant repeat queries)
5. **Async parallelization** (50% latency reduction)

See **[Project Structure](project-structure.md)** for where these changes would fit in the codebase.
