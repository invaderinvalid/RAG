# Architecture Overview

The Scientific Literature Explorer uses a modular pipeline architecture that combines retrieval-augmented generation, context compression, and multi-stage verification to produce accurate, well-cited answers from scientific literature.

---

## System Architecture

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'sans-serif'}}}%%
flowchart TD
    Start([User Question]) --> Triage[Research Agent<br/>Triage + Keywords<br/>Gemini]
    
    Triage -->|General Question| DirectAnswer[Direct Answer<br/>Gemini COT]
    Triage -->|Research Question| ArXiv[ArXiv API Search]
    
    ArXiv --> PDFs[Parallel PDF Download<br/>& Text Extraction<br/>PyPDF2]
    PDFs --> RAG[RAG Pipeline<br/>Chunk → TF-IDF → Retrieve]
    
    RAG --> Compress[ScaleDown API<br/>Context Compression<br/>40-60% reduction]
    
    Compress --> Workflow[Anti-Hallucination Workflow]
    DirectAnswer --> Workflow
    
    subgraph Workflow [Reasoning Pipeline]
        COT[Chain-of-Thought<br/>Gemini 2048 tokens<br/>Strict Citations]
        Verify[Self-Verification<br/>Gemini Fast 1024 tokens<br/>Citation Check]
        Critique[Self-Critique<br/>Gemini Fast 1024 tokens<br/>Quality Review]
        
        COT --> Verify
        Verify --> Critique
    end
    
    Workflow --> Store[Artifact Storage<br/>Compressed Markdown]
    Store --> Session[(Session JSON<br/>Conversation History)]
    
    Session --> Output([Answer + Citations])
    
    style Start fill:#1976d2,stroke:#0d47a1,stroke-width:3px,color:#fff,rx:10,ry:10
    style Output fill:#2e7d32,stroke:#1b5e20,stroke-width:3px,color:#fff,rx:10,ry:10
    style Compress fill:#f57c00,stroke:#e65100,stroke-width:3px,color:#fff,rx:5,ry:5
    style Workflow fill:#7b1fa2,stroke:#4a148c,stroke-width:2px,color:#fff,rx:5,ry:5
    style Triage fill:#d32f2f,stroke:#b71c1c,stroke-width:3px,color:#fff,rx:5,ry:5
    style DirectAnswer fill:#0288d1,stroke:#01579b,stroke-width:2px,color:#fff,rx:5,ry:5
    style ArXiv fill:#0288d1,stroke:#01579b,stroke-width:2px,color:#fff,rx:5,ry:5
    style PDFs fill:#0288d1,stroke:#01579b,stroke-width:2px,color:#fff,rx:5,ry:5
    style RAG fill:#0288d1,stroke:#01579b,stroke-width:2px,color:#fff,rx:5,ry:5
    style Store fill:#388e3c,stroke:#1b5e20,stroke-width:2px,color:#fff,rx:5,ry:5
    style Session fill:#5d4037,stroke:#3e2723,stroke-width:2px,color:#fff,rx:5,ry:5
    
    linkStyle default stroke:#e0e0e0,stroke-width:2px
```

---

## Core Components

### 1. Research Agent (Triage + Discovery)
**Purpose:** Intelligently routes questions and discovers relevant papers

- **Question Classification:** Uses Gemini to classify as `general`, `conceptual`, or `research`
- **Keyword Extraction:** Extracts search terms and ArXiv query in the same API call
- **Paper Discovery:** Searches ArXiv Atom API with parallel PDF downloads
- **Fallback Handling:** Uses heuristic keyword extraction if Gemini is rate-limited

### 2. RAG Pipeline
**Purpose:** Retrieves and prepares relevant paper content

- **Chunking:** Splits papers into overlapping segments (1000 chars, 200 overlap)
- **TF-IDF Indexing:** scikit-learn vectorization with stop-word removal
- **Retrieval:** Cosine similarity matching, returns top-k chunks (default: 5)
- **Source Tracking:** Every chunk labeled with `arxiv:XXXX.XXXXX` for citations

### 3. ScaleDown Compression
**Purpose:** Reduces token count while preserving semantic meaning

- **Context Compression:** 40-60% token reduction before sending to LLM
- **Query-Aware:** Uses the user's question to guide what information to preserve
- **Tokenizer Optimization:** Optimized for `gemini-2.5-flash` tokenizer
- **Artifact Compression:** Also compresses COT traces before storage

### 4. Anti-Hallucination Workflow
**Purpose:** Multi-stage verification ensures factual accuracy

- **Chain-of-Thought (COT):** Requires inline citations for every claim
- **Self-Verification:** Separate Gemini call checks each citation
- **Self-Critique:** Optional quality evaluation
- **Configurable:** Stages can be toggled or reordered via CLI

### 5. Artifact Storage
**Purpose:** Persistent storage of reasoning traces

- **Markdown Format:** Each stage output saved as `.md` file
- **Compression:** Artifacts compressed via ScaleDown before storage
- **Metadata:** Timestamps, token counts, compression stats tracked
- **Organization:** Separate folders for `cot/`, `self_verify/`, `self_critique/`

### 6. Session Management
**Purpose:** Maintains conversation history across interactions

- **JSON Persistence:** Each session stored as `{session_id}.json`
- **Multi-Turn Support:** Previous Q&A included in context
- **Paper Tracking:** Tracks which papers were ingested per session
- **Automatic Loading:** Latest session auto-loaded if applicable

---

## Data Flow

### Question Processing Flow

1. **User submits question** → Research Agent
2. **Triage classification** → Routes to appropriate handler
3. **Paper discovery** (if needed) → ArXiv API + parallel PDF downloads
4. **Text extraction** → PyPDF2 processes PDFs
5. **Chunking + Indexing** → TF-IDF vectorization
6. **Retrieval** → Top-k chunks by cosine similarity
7. **Compression** → ScaleDown reduces token count
8. **Reasoning** → Multi-stage workflow generates answer
9. **Storage** → Artifacts saved, session updated
10. **Response** → Final answer with citations returned

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Intelligence** | Google Gemini 2.5 Flash | Answer generation, classification, verification |
| **Compression** | ScaleDown API | Context compression, fallback generation |
| **Paper Source** | ArXiv Atom API | Scientific paper search and metadata |
| **PDF Processing** | PyPDF2 | Text extraction from PDFs |
| **Retrieval** | scikit-learn TF-IDF | Vectorization and similarity search |
| **Storage** | JSON (sessions), Markdown (artifacts) | Persistence |
| **CLI** | Rich (terminal UI) | Interactive tables, panels, markdown rendering |

---

## Next: How It Works

See **[How It Works](how-it-works.md)** for the detailed end-to-end flow and step-by-step breakdown.
