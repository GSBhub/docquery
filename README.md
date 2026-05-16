# docquery

PDF → embeddings → LangGraph RAG pipeline with structured Pydantic extraction and interactive chat.

## Overview

Two modes share a single SQLite vector database:

- **Ingest** — load a PDF, chunk it, embed it, store in SQLite (`sqlite-vec` + FTS5)
- **Chat** — interactive ReAct agent that answers free-form questions using three built-in tools
- **Query** — structured extraction: provide a Pydantic model and get back validated instances

## Setup

```bash
# Install dependencies (requires uv)
uv sync

# Copy and edit the environment config
cp .env.example .env   # or create .env directly (see Configuration below)
```

## Configuration

All settings are read from environment variables or a `.env` file in the project root.

### Local Ollama (recommended for testing)

```env
EMBED_PROVIDER=ollama
EMBED_BASE_URL=http://localhost:11434
EMBED_MODEL=embeddinggemma:latest

LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=gemma4:e2b
```

### OpenAI (or any OpenAI-compatible endpoint)

```env
EMBED_PROVIDER=openai
EMBED_BASE_URL=https://api.openai.com/v1
EMBED_API_KEY=sk-...
EMBED_MODEL=text-embedding-3-small

LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

### All settings

| Variable | Default | Description |
|---|---|---|
| `EMBED_PROVIDER` | `openai` | `openai` or `ollama` |
| `EMBED_BASE_URL` | `https://api.openai.com/v1` | Embedding endpoint |
| `EMBED_API_KEY` | — | API key (`ollama` for local) |
| `EMBED_MODEL` | `text-embedding-3-small` | Embedding model name |
| `EMBED_BATCH_SIZE` | `32` | Chunks per embedding API call |
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | LLM endpoint |
| `LLM_API_KEY` | — | API key |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `DB_PATH` | `rag.db` | SQLite database file |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Similarity results returned per search |
| `MAX_RETRIES` | `3` | Max extraction retries on validation error |

---

## Usage

All commands accept `--log-level DEBUG|INFO|WARNING|ERROR` (default `INFO`) and `--db <path>` to override `DB_PATH`.

### 1. Ingest a PDF

```bash
docquery ingest manual.pdf
# or with a custom database path
docquery ingest manual.pdf --db my_manual.db
```

Progress bars show pages loaded, embedding batches, and chunks written to the database. Re-running on the same PDF is safe — duplicate chunks are skipped via content hash.

### 2. Chat mode

```bash
docquery chat --db my_manual.db
```

Opens an interactive REPL. The agent has three built-in tools it calls automatically:

| Tool | When the agent uses it |
|---|---|
| `similarity_search` | General questions about document content |
| `page_lookup` | Questions referencing a specific page number |
| `keyword_search` | Exact-match lookups: opcodes, hex bytes, mnemonics |

Example session:

```
You: What does the MOV instruction do?
Agent: The MOV instruction copies a value into a destination register...

You: What's on page 50?
Agent: Page 50 covers saturating arithmetic instructions: QADD, QADD16...

You: What is opcode 0xfe 0xed?
Agent: Searching for that byte sequence...

You: /reset       ← clears conversation history
You: exit         ← quit
```

Enable verbose output to see which tools are called and why:

```bash
docquery --log-level DEBUG chat --db my_manual.db
```

### 3. Query mode — one-shot answer or structured extraction

**Free-form answer** (no schema required):

```bash
docquery query "What does the WFI instruction do?" --db my_manual.db
```

The agent retrieves relevant context and answers in plain text. Good for scripting or quick lookups without an interactive session.

**Structured extraction** — provide a Pydantic model class via `--schema`:

```bash
docquery query "Extract the LDR instruction encoding and operands" \
  --schema examples.arm_isa.ISAInstruction \
  --db my_manual.db
```

Output is printed as JSON. The pipeline automatically retries (up to `MAX_RETRIES`) if the LLM returns output that doesn't validate against the schema.

> **Note:** `--schema` takes a dotted Python class path (`module.ClassName`), not an LLM model name. The LLM model is set via `LLM_MODEL` in `.env`.

See `examples/arm_isa.py` for a complete `ISAInstruction` example.

---

## Adding custom tools (skills)

Register any LangChain `Tool` with the `ToolRegistry` to extend the chat agent:

```python
from langchain_core.tools import Tool
from docquery.config import Settings
from docquery.embeddings.provider import get_embeddings
from docquery.storage.vector_store import VectorStore
from docquery.tools.registry import ToolRegistry
from docquery.pipeline.chat import ChatAgent

def my_tool_fn(query: str) -> str:
    return f"custom result for: {query}"

settings = Settings()
vs = VectorStore(settings.db_path, embedding_dim=768)
registry = ToolRegistry(vs, get_embeddings(settings), settings)
registry.register(Tool(name="my_tool", description="...", func=my_tool_fn))

agent = ChatAgent(registry, settings)
print(agent.chat("your question here"))
```

---

## Building your own extraction pipeline

Subclass or instantiate `ExtractionPipeline` with your own Pydantic model and system prompt:

```python
from pydantic import BaseModel
from docquery.pipeline.extractor import ExtractionPipeline

class MySchema(BaseModel):
    field_a: str
    field_b: list[str]

pipeline = ExtractionPipeline(
    db_path="my_manual.db",
    output_model=MySchema,
    system_prompt="Extract ... from the document. Return only JSON.",
)

result = pipeline.run("your query here")
print(result.model_dump_json(indent=2))
```

---

## Running tests

```bash
# Run all tests
uv run pytest

# Verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_vector_store.py -v

# Run a specific test
uv run pytest tests/test_extractor.py::test_valid_extraction -v

# Show stdout/stderr (useful for seeing tqdm progress in tests)
uv run pytest -s
```

Tests use mock embeddings (no API calls required) and a temporary SQLite database. No `.env` or running Ollama instance is needed to run the test suite.

---

## Project layout

```
src/docquery/
├── cli.py                  # ingest / chat / query subcommands
├── config.py               # Settings from env / .env
├── logging_config.py       # setup_logging()
├── ingestion/
│   ├── pdf_loader.py       # PyMuPDF + pytesseract OCR fallback
│   └── chunker.py          # RecursiveCharacterTextSplitter
├── storage/
│   └── vector_store.py     # SQLite + sqlite-vec (KNN) + FTS5 (keyword)
├── embeddings/
│   ├── provider.py         # get_embeddings() — OpenAI or Ollama
│   └── llm.py              # get_llm() — OpenAI or Ollama
├── tools/
│   ├── retrieval_tools.py  # similarity_search tool
│   ├── page_tools.py       # page_lookup tool
│   ├── keyword_tools.py    # keyword_search tool
│   └── registry.py         # ToolRegistry with .register()
└── pipeline/
    ├── state.py            # ExtractionState, ChatState TypedDicts
    ├── nodes.py            # retrieve / extract / validate / retry nodes
    ├── extractor.py        # ExtractionPipeline
    └── chat.py             # ChatAgent (LangGraph ReAct loop)
```
