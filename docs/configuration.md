# Configuration Reference

All configuration is managed through environment variables loaded from the `.env` file.

---

## Environment Variables

### Required Variables

| Variable | Description |
|----------|-------------|
| `SCALEDOWN_API_KEY` | **Required.** Your ScaleDown API key from [ScaleDown](https://blog.scaledown.ai/blog/getting-started). Used for context compression and fallback generation. |
| `GEMINI_API_KEY` | **Required.** Your Google Gemini API key from [AI Studio](https://aistudio.google.com/apikey). Used for question triage, keyword extraction, COT reasoning, verification, and critique. |

---

### Optional Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SCALEDOWN_MODEL` | `gemini-2.5-flash` | Target model for ScaleDown compression optimization. ScaleDown will optimize the tokenization for this specific model's tokenizer. Valid values: `gpt-4o`, `claude-3-5-sonnet`, `gemini-2.5-flash`, etc. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | The Gemini model to use for generation. Options: `gemini-2.5-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`, etc. Flash models are faster/cheaper but less capable than Pro models. |
| `SCALEDOWN_TIMEOUT` | `15` | Timeout in seconds for ScaleDown API calls. Increase if you see timeout errors on slow networks. |
| `CHUNK_SIZE` | `1000` | Number of characters per text chunk when splitting papers. Larger chunks = more context per chunk but fewer chunks retrieved. |
| `CHUNK_OVERLAP` | `200` | Number of overlapping characters between adjacent chunks. Prevents information loss at chunk boundaries. |
| `TOP_K` | `5` | Number of chunks to retrieve per RAG query. Higher = more context but also more tokens and potential noise. |

---

## Model Selection

### ScaleDown Model

The `SCALEDOWN_MODEL` variable tells ScaleDown which tokenizer to optimize for. Use:
- `gemini-2.5-flash` if you're using Gemini 2.5 Flash (default)
- `gpt-4o` if you're using OpenAI's GPT-4o
- `claude-3-5-sonnet` if you're using Claude 3.5 Sonnet

**This does NOT change which model ScaleDown uses internally** — it only optimizes the compression output for your target model's tokenizer.

### Gemini Model

The `GEMINI_MODEL` variable selects which Gemini model to use for generation:
- **`gemini-2.5-flash`** (default): Fastest, cheapest, good quality
- **`gemini-1.5-flash`**: Previous generation, slower than 2.5
- **`gemini-1.5-pro`**: Much smarter but slower and more expensive

---

## RAG Configuration

### Chunk Size

The `CHUNK_SIZE` controls how large each text chunk is:
- **Too small** (e.g., 200): Chunks lose semantic meaning, context fragmentation
- **Too large** (e.g., 5000): Fewer chunks retrieved, may miss relevant details
- **Default 1000**: Good balance for most scientific papers

### Chunk Overlap

The `CHUNK_OVERLAP` ensures no information is lost at boundaries:
- **No overlap** (0): Risk of splitting sentences/paragraphs
- **Too much overlap** (500+): Redundant content, wasted tokens
- **Default 200**: Usually captures 1-2 sentences of overlap

### Top-K

The `TOP_K` controls how many chunks are retrieved:
- **Too few** (<3): May miss important information
- **Too many** (>10): More noise, higher token costs
- **Default 5**: Works well for most questions

---

## Example `.env`

```env
# Required
SCALEDOWN_API_KEY=sk_sd_abc123...
GEMINI_API_KEY=AIza...

# Optional - Uncomment to override defaults
# SCALEDOWN_MODEL=gemini-2.5-flash
# GEMINI_MODEL=gemini-2.5-flash
# SCALEDOWN_TIMEOUT=15
# CHUNK_SIZE=1000
# CHUNK_OVERLAP=200
# TOP_K=5
```

---

## Next: Usage Guide

See **[Usage Guide](usage.md)** for all available commands and examples.
