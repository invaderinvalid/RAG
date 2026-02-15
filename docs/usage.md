# Usage Guide

This page covers all CLI commands and common workflows.

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `ask "question"` | Research question → auto-discover papers → full pipeline |
| `ask "question" --session ID` | Continue an existing session |
| `papers "query"` | Interactive paper explorer — search, select, ask |
| `paper <arxiv_id> "question"` | Deep-dive into a specific paper |
| `search "query"` | Search ArXiv (listing only, no analysis) |
| `sessions` | List all conversation sessions |
| `workflow show` | Display current pipeline configuration |
| `workflow toggle <stage> <on\|off>` | Enable/disable a workflow stage |
| `workflow reorder <s1,s2,...>` | Reorder workflow stages |
| `artifacts list` | List stored reasoning artifacts |

---

## Command Details

### `ask` — Research Questions

Ask a research question and let the system discover relevant papers:

```bash
# Basic usage
python -m src.main ask "What are the latest advances in neural architecture search?"

# Continue a previous conversation
python -m src.main ask "How does this compare to random search?" --session abc123
```

**What happens:**
1. Question is triaged (general/conceptual/research)
2. If research question: ArXiv papers discovered and downloaded
3. Papers chunked, indexed, and retrieved
4. Context compressed via ScaleDown
5. Full anti-hallucination pipeline runs (COT → Verify → Critique)
6. Answer displayed with citations
7. Session saved for follow-up questions

**Expected time:**
- General questions: ~5s (direct answer)
- Research questions: ~30-60s (paper discovery + full pipeline)

---

### `papers` — Interactive Paper Explorer

Search, browse, and analyze papers interactively:

```bash
python -m src.main papers "attention mechanism transformers"
```

**Interactive Commands:**
- **Type text**: Ask a question about the currently selected paper
- **Type a number**: Switch to a different paper from the list
- **`back`**: Return to the paper list
- **`list`**: Show the paper list again
- **`s`**: Start a new search
- **`q`**: Quit the explorer

**Features:**
- **Persistent session**: All interactions share the same session
- **No refetching**: Papers are cached — follow-up questions are instant
- **Conversation history**: LLM sees previous Q&A for better context
- **Full pipeline**: Each answer goes through COT → Verify

**Example session:**

```
> papers "attention mechanism transformers"
[Paper list displayed]

Select a paper (1-10), or type 's' to search again, 'q' to quit: 1
[Paper #1 selected: "Attention Is All You Need"]

Ask a question (or 'back' to list, 'q' to quit): What is the multi-head attention mechanism?
[Full pipeline runs → Answer displayed with citations]

Ask a question (or 'back' to list, 'q' to quit): How does it differ from single-head attention?
[Follow-up answered using session history]

Ask a question (or 'back' to list, 'q' to quit): back
[Returns to paper list]

Select a paper (1-10), or type 's' to search again, 'q' to quit: q
```

---

### `paper` — Deep-Dive into a Specific Paper

Analyze a known ArXiv paper:

```bash
python -m src.main paper 1706.03762 "What is the multi-head attention mechanism?"
```

**What happens:**
1. Downloads ArXiv paper `1706.03762` (if not cached)
2. Extracts text and indexes it
3. Runs full pipeline with **paper-specific grounding**
4. Answer focused ONLY on content from this paper

**Paper-specific grounding** means:
- System prompt explicitly says "analyze THIS specific paper"
- LLM instructed NOT to use training data
- Must cite specific sections, equations, figures, or tables
- If paper doesn't mention something, it must say so

---

### `search` — Quick ArXiv Search

Search ArXiv without analysis:

```bash
python -m src.main search "graph neural networks"
```

**Output:**
- Top 10 ArXiv results with titles, authors, and abstracts
- No paper download or analysis
- Useful for quickly finding relevant papers

---

### `sessions` — View Conversation History

List all saved sessions:

```bash
python -m src.main sessions
```

**Output:**
- Session IDs
- Creation timestamps
- Number of Q&A turns
- Papers involved

Use a session ID with `ask --session` to continue a conversation.

---

### `workflow` — Configure Pipeline

View and modify the reasoning pipeline:

#### Show Current Configuration

```bash
python -m src.main workflow show
```

**Output:**
- List of stages in order
- Enabled/disabled status for each

#### Toggle Stages

```bash
# Enable self-critique
python -m src.main workflow toggle self_critique on

# Disable self-verification (not recommended!)
python -m src.main workflow toggle self_verify off
```

#### Reorder Stages

```bash
# Standard order
python -m src.main workflow reorder cot,self_verify,self_critique

# Custom order (e.g., critique before verify)
python -m src.main workflow reorder cot,self_critique,self_verify
```

---

### `artifacts` — View Stored Outputs

List all stored reasoning artifacts:

```bash
python -m src.main artifacts list
```

**Output:**
- Artifact IDs
- Stage (cot, self_verify, self_critique)
- Timestamps
- File sizes

Artifacts are stored in `artifacts/` as compressed markdown files.

---

## Workflow Examples

### Example 1: Quick Research

```bash
# Single command gets you a cited answer
python -m src.main ask "What are transformers in NLP?"
```

**Result:**
- ArXiv papers discovered
- Relevant chunks retrieved and compressed
- COT reasoning with citations
- Verification of all citations
- Critique of answer quality
- Final answer displayed in terminal

---

### Example 2: Multi-Turn Conversation

```bash
# First question
python -m src.main ask "What is neural architecture search?"
# Note the session ID in the output: abc123

# Follow-up
python -m src.main ask "What are the main challenges?" --session abc123

# Another follow-up
python -m src.main ask "How does DARTS address these?" --session abc123
```

**Result:**
- Each follow-up has context from previous turns
- Papers are cached (no re-downloading)
- Session history grows with each interaction

---

### Example 3: Interactive Paper Exploration

```bash
python -m src.main papers "transformers"

# Select paper #1
1

# Ask questions
What is the encoder structure?
How many layers does it have?
What is positional encoding?

# Switch to paper #3
3

# Compare
How does this differ from the original transformer?

# Exit
q
```

**Result:**
- Seamless switching between papers
- All questions share one session
- Full pipeline runs for each answer

---

### Example 4: Custom Workflow

```bash
# Disable critique for faster responses
python -m src.main workflow toggle self_critique off

# Ask question (skips critique stage)
python -m src.main ask "What is gradient descent?"

# Re-enable critique
python -m src.main workflow toggle self_critique on
```

**Result:**
- Workflow configuration persists across runs
- Faster responses when critique is disabled
- Flexibility to trade quality for speed

---

## Next: Methodology

See **[Methodology](methodology.md)** to understand the technical approach.
