# Getting Started

This guide walks you through setting up the Scientific Literature Explorer on your machine.

---

## Prerequisites

Before installing, ensure you have:

- **Python ≥ 3.10** (3.11+ recommended)
- **A ScaleDown API key** — Get yours at [ScaleDown Getting Started](https://blog.scaledown.ai/blog/getting-started)
- **A Google Gemini API key** — Free tier available at [Google AI Studio](https://aistudio.google.com/apikey)

---

## Installation

### 1. Clone the Repository

```bash
git clone <repo-url>
cd RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `requests` — HTTP client for ScaleDown, Gemini, and ArXiv APIs
- `python-dotenv` — Load `.env` configuration
- `numpy` — Array operations for TF-IDF
- `scikit-learn` — TF-IDF vectorizer and cosine similarity
- `PyPDF2` — PDF text extraction from ArXiv papers
- `rich` — Terminal UI (tables, panels, markdown, progress spinners)

---

## Configuration

### 1. Create `.env` File

Copy the example environment file:

```bash
cp .env.example .env
```

### 2. Add Your API Keys

Open `.env` in your favorite editor and fill in your keys:

```env
# Required
SCALEDOWN_API_KEY=your_scaledown_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional Configuration Overrides
SCALEDOWN_MODEL=gemini-2.5-flash    # Target model for compression optimization
GEMINI_MODEL=gemini-2.5-flash       # Gemini model to use
CHUNK_SIZE=1000                      # Characters per chunk
CHUNK_OVERLAP=200                    # Overlap between chunks
TOP_K=5                              # Number of chunks to retrieve
SCALEDOWN_TIMEOUT=15                 # Timeout in seconds for ScaleDown API
```

See **[Configuration Reference](configuration.md)** for detailed explanations of each variable.

---

## Directory Structure

The system will automatically create these folders on first use:

```
RAG/
├── papers/          # Downloaded PDFs and extracted text
├── artifacts/       # Stored reasoning outputs (COT, verify, critique)
│   ├── cot/
│   ├── self_verify/
│   └── self_critique/
└── sessions/        # Conversation history (JSON per session)
```

---

## Verify Installation

Test that everything is working:

```bash
# Ask a simple question
python -m src.main ask "What is a convolutional neural network?"
```

If you see a response, you're ready to go! 🎉

---

## Next: Usage Examples

See **[Usage Guide](usage.md)** for all available commands and examples.
