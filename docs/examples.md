# Usage Examples

Practical examples demonstrating common workflows and use cases.

---

## Example 1: Quick Research

### Scenario
You need to quickly understand a topic from recent research papers.

### Command
```bash
python -m src.main ask "What is neural architecture search?"
```

### What Happens
1. ⚡ Question triaged as "research"
2. 🔍 ArXiv searched for relevant papers (top 5 results)
3. 📥 PDFs downloaded in parallel (~10-15s)
4. ✂️ Text extracted and chunked (1000 chars, 200 overlap)
5. 📊 Top-5 chunks retrieved via TF-IDF
6. 🗜️ Context compressed via ScaleDown (1500 → 600 tokens)
7. 🧠 COT reasoning with inline citations
8. ✅ Self-verification checks all citations
9. 📋 Self-critique evaluates quality
10. 💾 Session saved with ID `abc123`

### Expected Result
```
# Scientific Literature Explorer

## Answer

Neural architecture search (NAS) is an automated method for discovering
optimal neural network architectures [arxiv:1808.05377]. Unlike manual
design, NAS uses algorithms such as reinforcement learning or evolutionary
strategies to explore the architecture search space [arxiv:1808.05377].

DARTS (Differentiable Architecture Search) improved efficiency by making
the search space continuous and differentiable [arxiv:1806.09055]. This
allows gradient-based optimization instead of discrete search methods.

## References

- [arxiv:1808.05377] Elsken et al., "Neural Architecture Search: A Survey"
- [arxiv:1806.09055] Liu et al., "DARTS: Differentiable Architecture Search"

## Verification Summary
✅ 2 claims verified
✅ All citations supported
📄 Sources: arxiv:1808.05377, arxiv:1806.09055
```

**Time:** ~45-60 seconds

---

## Example 2: Multi-Turn Conversation

### Scenario
You want to ask follow-up questions using the same papers.

### Commands
```bash
# First question
python -m src.main ask "What is neural architecture search?"
# Note the session ID: abc123

# Follow-up #1
python -m src.main ask "What are the main challenges?" --session abc123

# Follow-up #2
python -m src.main ask "How does DARTS address these?" --session abc123
```

### What Happens
- First question: Full pipeline (~45-60s)
- Follow-ups: Papers cached, faster pipeline (~20-30s)
- Each answer has context from previous Q&A

### Expected Flow
```
Q1: "What is neural architecture search?"
→ Papers downloaded: arxiv:1808.05377, arxiv:1806.09055, arxiv:1802.03268
→ Session abc123 created

Q2: "What are the main challenges?" --session abc123
→ Papers already cached (no download)
→ LLM sees Q1 + A1 as context
→ Answer focuses on challenges (computational cost, search space size)

Q3: "How does DARTS address these?" --session abc123
→ Papers still cached
→ LLM sees Q1+A1, Q2+A2 as context
→ Answer connects DARTS from Q1 to challenges from Q2
```

**Total Time:** Q1 (60s) + Q2 (25s) + Q3 (25s) = ~110s

---

## Example 3: Interactive Paper Explorer

### Scenario
You want to browse multiple papers and ask questions about specific ones.

### Command
```bash
python -m src.main papers "transformers attention mechanism"
```

### Interactive Session
```
> papers "transformers attention mechanism"

Papers found:
1. [arxiv:1706.03762] Attention Is All You Need (Vaswani et al., 2017)
2. [arxiv:2002.04745] Reformer: The Efficient Transformer (Kitaev et al., 2020)
3. [arxiv:2006.04768] Longformer (Beltagy et al., 2020)
...

Select paper (1-10), or 's' to search, 'q' to quit: 1

[Paper #1 selected: Attention Is All You Need]

Ask question (or 'back', 'q'): What is the multi-head attention mechanism?

[Full pipeline runs with paper-specific grounding]

## Answer
The multi-head attention mechanism splits the input into h=8 parallel
attention heads [arxiv:1706.03762, Section 3.2.2]. Each head learns
different representation subspaces...

Ask question (or 'back', 'q'): How many parameters does the base model have?

[Follow-up answered instantly using cached paper]

## Answer
The base Transformer model has 65M parameters [arxiv:1706.03762, Table 3]...

Ask question (or 'back', 'q'): back

[Returns to paper list]

Select paper (1-10), or 's' to search, 'q' to quit: 2

[Switches to Paper #2: Reformer]

Ask question (or 'back', 'q'): How does this compare to the original Transformer?

[Pipeline runs comparing Reformer to Transformer]

Ask question (or 'back', 'q'): q
```

**Features:**
- Seamless paper switching
- All interactions share one session
- Context-aware answers (remembers previous questions)
- No refetching on follow-ups

---

## Example 4: Specific Paper Analysis

### Scenario
You know the exact ArXiv ID and want to analyze that paper only.

### Command
```bash
python -m src.main paper 1706.03762 "Explain the positional encoding mechanism"
```

### What Happens
1. Downloads paper `1706.03762` (if not cached)
2. Extracts text
3. Runs pipeline with **paper-specific grounding**
4. Answer uses ONLY information from this paper

### Paper-Specific Grounding
System prompt includes:
```
IMPORTANT: You are analyzing a SPECIFIC research paper.
ONLY use information from the paper excerpts below.
Do NOT add information from your training data.
If the paper does not mention something, say so.
Cite specific sections, equations, figures, or tables.
```

### Expected Result
```
## Answer

According to Section 3.5, the positional encoding uses sinusoidal functions:

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

where pos is the position and i is the dimension. This allows the model
to learn relative positions [arxiv:1706.03762, Section 3.5].

The paper states that this method was chosen because it allows the model
to extrapolate to sequence lengths longer than those seen during training
[arxiv:1706.03762, Section 3.5].

The paper does NOT provide ablation studies comparing sinusoidal encoding
to learned positional embeddings.

## References
- [arxiv:1706.03762] Section 3.5 (Positional Encoding)
```

**Time:** ~30-40 seconds

---

## Example 5: Configure Workflow

### Scenario
You want faster responses and don't need critique.

### Commands
```bash
# Check current workflow
python -m src.main workflow show

# Disable critique
python -m src.main workflow toggle self_critique off

# Ask question (skips critique stage)
python -m src.main ask "What is gradient descent?"

# Re-enable critique later
python -m src.main workflow toggle self_critique on
```

### Expected Output
```
$ python -m src.main workflow show

Current Workflow:
1. cot (enabled) — Chain-of-thought reasoning
2. self_verify (enabled) — Citation verification
3. self_critique (enabled) — Quality evaluation

$ python -m src.main workflow toggle self_critique off

✅ Disabled stage: self_critique

$ python -m src.main workflow show

Current Workflow:
1. cot (enabled) — Chain-of-thought reasoning
2. self_verify (enabled) — Citation verification
3. self_critique (disabled) — Quality evaluation
```

**Speed Improvement:** ~5-8 seconds saved per question

---

## Example 6: Artifact Management

### Scenario
You want to review the stored reasoning artifacts.

### Command
```bash
python -m src.main artifacts list
```

### Expected Output
```
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Artifact ID  ┃ Stage        ┃ Timestamp          ┃ Size  ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ abc123       │ cot          │ 2024-01-15 10:30   │ 2.3KB │
│ abc123       │ self_verify  │ 2024-01-15 10:31   │ 1.1KB │
│ abc123       │ self_critique│ 2024-01-15 10:32   │ 0.8KB │
│ def456       │ cot          │ 2024-01-15 14:45   │ 3.1KB │
│ def456       │ self_verify  │ 2024-01-15 14:46   │ 1.4KB │
└──────────────┴──────────────┴────────────────────┴───────┘
```

Files are stored in `artifacts/`:
- `artifacts/cot/abc123.md` — Chain-of-thought reasoning
- `artifacts/self_verify/abc123.md` — Verification table
- `artifacts/self_critique/abc123.md` — Critique report

You can open these files in any text editor to review the full reasoning trace.

---

## Example 7: Session Management

### Scenario
You want to see all your previous conversations.

### Command
```bash
python -m src.main sessions
```

### Expected Output
```
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Session ID ┃ Created            ┃ Turns ┃ Papers             ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ abc123     │ 2024-01-15 10:30   │ 3     │ arxiv:1706.03762   │
│            │                    │       │ arxiv:1808.05377   │
│ def456     │ 2024-01-15 14:22   │ 1     │ arxiv:2103.14030   │
│ ghi789     │ 2024-01-16 09:15   │ 5     │ arxiv:1806.09055   │
│            │                    │       │ arxiv:1802.03268   │
└────────────┴────────────────────┴───────┴────────────────────┘
```

Use any session ID to continue a conversation:
```bash
python -m src.main ask "Tell me more about the attention mechanism" --session abc123
```

---

## Next: Methodology

See **[Methodology](methodology.md)** to understand the technical approach behind these workflows.
