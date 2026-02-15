# Methodology

This page explains the technical approaches and design patterns used in the Scientific Literature Explorer.

---

## 1. Retrieval-Augmented Generation (RAG)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'sans-serif'}}}%%
flowchart LR
    Paper([Paper Text]) --> Chunk[Chunking<br/>1000 chars<br/>200 overlap]
    
    Chunk --> Vec[TF-IDF<br/>Vectorization<br/>sklearn]
    Vec --> Index[(Chunk Index<br/>Sparse Matrix)]
    
    Query([User Query]) --> QVec[Vectorize Query<br/>Same TF-IDF]
    QVec --> Sim[Cosine<br/>Similarity]
    Index --> Sim
    
    Sim --> TopK[Top-K Chunks<br/>default: 5]
    TopK --> Compress[ScaleDown<br/>Compression<br/>40-60% reduction]
    
    Compress --> LLM[Gemini<br/>Generation]
    LLM --> Answer([Answer with<br/>Citations])
    
    style Paper fill:#424242,stroke:#212121,stroke-width:3px,color:#fff,rx:10,ry:10
    style Query fill:#424242,stroke:#212121,stroke-width:3px,color:#fff,rx:10,ry:10
    style Index fill:#1565c0,stroke:#0d47a1,stroke-width:3px,color:#fff,rx:5,ry:5
    style Compress fill:#f57c00,stroke:#e65100,stroke-width:3px,color:#fff,rx:5,ry:5
    style Answer fill:#2e7d32,stroke:#1b5e20,stroke-width:3px,color:#fff,rx:10,ry:10
    style Chunk fill:#00838f,stroke:#006064,stroke-width:2px,color:#fff,rx:5,ry:5
    style Vec fill:#00838f,stroke:#006064,stroke-width:2px,color:#fff,rx:5,ry:5
    style QVec fill:#00838f,stroke:#006064,stroke-width:2px,color:#fff,rx:5,ry:5
    style Sim fill:#6a1b9a,stroke:#4a148c,stroke-width:2px,color:#fff,rx:5,ry:5
    style TopK fill:#6a1b9a,stroke:#4a148c,stroke-width:2px,color:#fff,rx:5,ry:5
    style LLM fill:#1976d2,stroke:#0d47a1,stroke-width:2px,color:#fff,rx:5,ry:5
    
    linkStyle default stroke:#e0e0e0,stroke-width:2px
```

The RAG pattern ensures answers are grounded in actual paper content rather than relying solely on the LLM's training data:

- **Chunking**: Papers are split into overlapping segments (1000 chars, 200 overlap) to ensure no information is lost at boundaries
- **TF-IDF Vectorization**: scikit-learn's `TfidfVectorizer` creates sparse vector representations with English stop-word removal
- **Cosine Similarity**: Queries are matched against the chunk index; top-k (default 5) most similar chunks are retrieved
- **Source Tracking**: Every chunk retains its source label (e.g., `arxiv:2511.14362`) for citation tracing

---

## 2. Context Compression (ScaleDown)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'sans-serif'}}}%%
flowchart TD
    Raw[Raw Retrieved Chunks<br/>1500 tokens<br/>Redundant verbose] --> SD{ScaleDown API}
    
    Query[User Question] --> SD
    SD --> Analyze[Query-Aware<br/>Analysis]
    
    Analyze --> Remove[Remove Redundancy]
    Remove --> Preserve[Preserve Semantics]
    Preserve --> Optimize[Optimize for<br/>gemini-2.5-flash<br/>tokenizer]
    
    Optimize --> Compressed[Compressed Context<br/>600 tokens<br/>40-60% reduction]
    
    Compressed --> Benefits
    
    subgraph Benefits [Benefits]
        B1[Lower API Cost]
        B2[Faster Response]
        B3[More Context Fits<br/>in Window]
        B4[Better Focus]
    end
    
    style Raw fill:#c62828,stroke:#b71c1c,stroke-width:3px,color:#fff,rx:5,ry:5
    style Query fill:#424242,stroke:#212121,stroke-width:2px,color:#fff,rx:5,ry:5
    style Compressed fill:#2e7d32,stroke:#1b5e20,stroke-width:3px,color:#fff,rx:5,ry:5
    style SD fill:#f57c00,stroke:#e65100,stroke-width:3px,color:#fff,rx:5,ry:5
    style Analyze fill:#00838f,stroke:#006064,stroke-width:2px,color:#fff,rx:5,ry:5
    style Remove fill:#00838f,stroke:#006064,stroke-width:2px,color:#fff,rx:5,ry:5
    style Preserve fill:#00838f,stroke:#006064,stroke-width:2px,color:#fff,rx:5,ry:5
    style Optimize fill:#00838f,stroke:#006064,stroke-width:2px,color:#fff,rx:5,ry:5
    style Benefits fill:#6a1b9a,stroke:#4a148c,stroke-width:2px,color:#fff,rx:5,ry:5
    style B1 fill:#1976d2,stroke:#0d47a1,stroke-width:2px,color:#fff,rx:5,ry:5
    style B2 fill:#1976d2,stroke:#0d47a1,stroke-width:2px,color:#fff,rx:5,ry:5
    style B3 fill:#1976d2,stroke:#0d47a1,stroke-width:2px,color:#fff,rx:5,ry:5
    style B4 fill:#1976d2,stroke:#0d47a1,stroke-width:2px,color:#fff,rx:5,ry:5
    
    linkStyle default stroke:#e0e0e0,stroke-width:2px
```

Raw retrieved chunks are often redundant. ScaleDown's compression:
- Reduces token count by 40-60% while preserving semantics
- Uses the user's question as a guide (`prompt` parameter) to prioritize relevant information
- Optimizes for the target model's tokenizer (`gemini-2.5-flash`)
- The `"rate": "auto"` setting lets ScaleDown determine optimal compression

---

## 3. Multi-Stage Reasoning Workflow

Inspired by research on self-verification and chain-of-verification (CoVe):

- **Chain-of-Thought**: Forces step-by-step reasoning, reducing reasoning errors
- **Self-Verification**: A separate LLM call cross-references every claim against source documents
- **Self-Critique**: An independent evaluator checks for completeness and accuracy
- Stages are **configurable** — enable, disable, or reorder via CLI

---

## 4. Question Triage

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'sans-serif'}}}%%
flowchart LR
    Q([Question]) --> Classify{Gemini Triage<br/>+ Keyword Extract}
    
    Classify -->|GENERAL<br/>What is CNN?| Direct[Direct Answer<br/>No Papers<br/>~5s]
    Classify -->|CONCEPTUAL<br/>Explain attention| Minimal[Basic Search<br/>Skip Critique<br/>~15s]
    Classify -->|RESEARCH<br/>Latest NAS methods?| Full[Full Discovery<br/>COT+Verify+Critique<br/>~45s]
    
    Direct --> Answer1([Answer])
    Minimal --> Papers1[Light Paper Fetch]
    Papers1 --> Workflow1[Workflow<br/>critique=OFF]
    Workflow1 --> Answer2([Answer])
    
    Full --> Papers2[Deep Paper Discovery]
    Papers2 --> Workflow2[Full Workflow<br/>All Stages ON]
    Workflow2 --> Answer3([Answer])
    
    style Q fill:#424242,stroke:#212121,stroke-width:3px,color:#fff,rx:10,ry:10
    style Classify fill:#d32f2f,stroke:#b71c1c,stroke-width:3px,color:#fff,rx:5,ry:5
    style Direct fill:#2e7d32,stroke:#1b5e20,stroke-width:3px,color:#fff,rx:5,ry:5
    style Minimal fill:#f57c00,stroke:#e65100,stroke-width:3px,color:#fff,rx:5,ry:5
    style Full fill:#c62828,stroke:#b71c1c,stroke-width:3px,color:#fff,rx:5,ry:5
    style Answer1 fill:#1565c0,stroke:#0d47a1,stroke-width:2px,color:#fff,rx:10,ry:10
    style Answer2 fill:#1565c0,stroke:#0d47a1,stroke-width:2px,color:#fff,rx:10,ry:10
    style Answer3 fill:#1565c0,stroke:#0d47a1,stroke-width:2px,color:#fff,rx:10,ry:10
    style Papers1 fill:#00838f,stroke:#006064,stroke-width:2px,color:#fff,rx:5,ry:5
    style Papers2 fill:#00838f,stroke:#006064,stroke-width:2px,color:#fff,rx:5,ry:5
    style Workflow1 fill:#6a1b9a,stroke:#4a148c,stroke-width:2px,color:#fff,rx:5,ry:5
    style Workflow2 fill:#6a1b9a,stroke:#4a148c,stroke-width:2px,color:#fff,rx:5,ry:5
    
    linkStyle default stroke:#e0e0e0,stroke-width:2px
```

A single Gemini call classifies questions into three tiers:
- **General**: Simple factual questions → answered directly (no paper fetch, ~5s)
- **Conceptual**: Needs depth but not specific papers → uses workflow but may skip critique
- **Research**: Needs actual papers → full discovery + workflow pipeline

This saves 60-90 seconds for simple questions by skipping paper discovery entirely.

---

## 5. Resilient LLM Strategy

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'sans-serif'}}}%%
flowchart TD
    Start([API Call]) --> Gemini{Gemini API}
    
    Gemini -->|Success 200| Success([Return Response])
    Gemini -->|Rate Limit 429| Retry{Retry Count<br/>< 5?}
    
    Retry -->|Yes| Wait[Exponential Backoff<br/>5s, 10s, 20s, 40s, 60s]
    Wait --> Gemini
    
    Retry -->|No, All Failed| Fallback[ScaleDown<br/>Compression-as-Generation]
    Fallback --> FallbackSuccess([Return Compressed<br/>Extraction])
    
    Gemini -->|Other Error| Fail([Raise Exception])
    
    style Start fill:#424242,stroke:#212121,stroke-width:3px,color:#fff,rx:10,ry:10
    style Gemini fill:#1976d2,stroke:#0d47a1,stroke-width:3px,color:#fff,rx:5,ry:5
    style Success fill:#2e7d32,stroke:#1b5e20,stroke-width:3px,color:#fff,rx:10,ry:10
    style FallbackSuccess fill:#f57c00,stroke:#e65100,stroke-width:3px,color:#fff,rx:10,ry:10
    style Fail fill:#c62828,stroke:#b71c1c,stroke-width:3px,color:#fff,rx:10,ry:10
    style Retry fill:#6a1b9a,stroke:#4a148c,stroke-width:2px,color:#fff,rx:5,ry:5
    style Wait fill:#00838f,stroke:#006064,stroke-width:2px,color:#fff,rx:5,ry:5
    style Fallback fill:#ef6c00,stroke:#e65100,stroke-width:2px,color:#fff,rx:5,ry:5
    
    linkStyle default stroke:#e0e0e0,stroke-width:2px
```

**Implementation:**
```python
Primary: Gemini 2.5 Flash (full generation)
    │
    ├── Rate limited (429)?
    │   └── Retry with exponential backoff (5× up to 60s)
    │       └── Still limited?
    │           └── Fallback: ScaleDown compression-as-generation
    │
    └── Research Agent rate-limited?
        └── Heuristic keyword extraction (regex-based, no API call)
```

---

## Next: Anti-Hallucination Details

See **[Anti-Hallucination Pipeline](anti-hallucination.md)** for the full verification workflow.
