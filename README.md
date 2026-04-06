# 📚 RAG From Scratch

> **Retrieval-Augmented Generation built from primitives — no LangChain, no abstractions, full transparency.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Embeddings: BGE](https://img.shields.io/badge/Embeddings-BAAI%2Fbge--small--en-orange)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![Vector DB: FAISS](https://img.shields.io/badge/VectorDB-FAISS-red)](https://github.com/facebookresearch/faiss)
[![LLM: Claude / GPT](https://img.shields.io/badge/LLM-Claude%20%7C%20GPT-purple)](https://anthropic.com)

---

Most RAG tutorials wrap everything in LangChain. This project builds every layer from scratch so you can **understand and control** each component:

| Layer | This project | LangChain equivalent |
|-------|-------------|----------------------|
| PDF loading | `fitz` (PyMuPDF) | `PyPDFLoader` |
| Text chunking | Custom `RecursiveCharacterSplitter` | `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` (BAAI/bge) | `HuggingFaceEmbeddings` |
| Vector store | Raw `faiss-cpu` with save/load | `FAISS` wrapper |
| Retrieval | Top-K + MMR re-ranking | `VectorStoreRetriever` |
| Generation | Direct `anthropic` / `openai` SDK | `ChatAnthropic` / `ChatOpenAI` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INDEXING PATH                           │
│                                                                 │
│  PDF / TXT  ──►  DocumentLoader  ──►  RecursiveCharacterSplitter│
│                       │                        │               │
│                  (PyMuPDF)             chunk_size=512           │
│                                      chunk_overlap=64           │
│                                              │                  │
│                                    EmbeddingModel               │
│                              (BAAI/bge-small-en-v1.5)           │
│                                              │                  │
│                                     VectorStore (FAISS)         │
│                                   IndexFlatIP  ◄── saved        │
│                                    to disk                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         QUERY PATH                              │
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ▼                                                          │
│  EmbeddingModel.encode_query()                                  │
│      │                                                          │
│      ▼                                                          │
│  VectorStore.search()  ──► Top-K chunks  ──► (optional MMR)     │
│      │                                                          │
│      ▼                                                          │
│  Prompt Builder                                                 │
│   [SYSTEM: grounding instructions]                              │
│   [CONTEXT: chunk_1 | chunk_2 | ... | chunk_k]                  │
│   [USER: original question]                                     │
│      │                                                          │
│      ▼                                                          │
│  RAGGenerator  ──► Claude / GPT-4o-mini / Ollama               │
│      │                                                          │
│      ▼                                                          │
│  GenerationResult { answer, sources, token_usage }              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

- **Zero LangChain** — every component is readable, debuggable Python
- **BAAI/bge embeddings** — outperforms OpenAI ada-002 on most BEIR benchmarks at zero API cost
- **MMR retrieval** — Maximal Marginal Relevance for diverse, non-redundant context
- **Multi-provider** — Claude, GPT, or any OpenAI-compatible endpoint (Together AI, Groq, Ollama)
- **Streaming** — token-by-token streaming for Gradio UI
- **Persistent index** — FAISS index saved to disk; subsequent runs skip re-embedding
- **Gradio UI** — upload PDFs, ask questions, inspect retrieved chunks

---

## Project Structure

```
rag-from-scratch/
├── src/
│   ├── ingestion.py      # PDF loading + text chunking
│   ├── embeddings.py     # sentence-transformers wrapper
│   ├── vector_store.py   # FAISS index (add / search / save / load / MMR)
│   ├── generator.py      # LLM generation (Anthropic + OpenAI)
│   └── pipeline.py       # Orchestrates all components
├── app.py                # Gradio demo UI
├── quickstart.py         # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/SwattikChakra/rag-from-scratch.git
cd rag-from-scratch
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY or OPENAI_API_KEY
```

### 3. Run the CLI quickstart

```bash
python quickstart.py \
  --pdf data/sample_docs/your_doc.pdf \
  --question "What are the key findings?" \
  --provider anthropic \
  --debug     # print retrieved chunks before the answer
```

### 4. Run the Gradio UI

```bash
python app.py
# Open http://localhost:7860
```

---

## Python API

```python
from src.pipeline import RAGPipeline

# Build once — saves FAISS index to .cache/index/ for fast restarts
rag = RAGPipeline.from_documents(
    paths=["docs/annual_report.pdf", "docs/press_release.pdf"],
    provider="anthropic",       # or "openai"
    chunk_size=512,
    chunk_overlap=64,
    top_k=5,
    use_mmr=True,               # diverse retrieval
    index_dir=".cache/index",   # skip re-embedding on second run
)

# Query
result = rag.query("What was the revenue in Q3?")
print(result.answer)
print(result.sources)          # [("annual_report.pdf", 12), ...]
print(result.total_tokens)     # prompt + completion

# Streaming (for UI)
for token in rag.stream_query("Summarise the key risks"):
    print(token, end="", flush=True)

# Inspect retrieval without generation (debugging)
chunks = rag.retrieve_only("What is the gross margin?", top_k=3)
for doc, score in chunks:
    print(f"{doc.source} p.{doc.page} | {score:.4f} | {doc.text[:100]}")

# Hot-add documents without rebuilding
rag.add_documents(["docs/q4_supplement.pdf"])
```

---

## Design Decisions

### Why BGE embeddings over OpenAI ada-002?

`BAAI/bge-small-en-v1.5` scores competitively on BEIR retrieval benchmarks while running **locally at zero cost**. The `-small` variant fits in ~130 MB RAM — ideal for development and CI. Swap to `bge-large-en-v1.5` for ~5% quality improvement in production.

### Why `IndexFlatIP` over `IndexFlatL2`?

With L2-normalised vectors, inner product equals cosine similarity. `IndexFlatIP` is slightly faster and its scores are more interpretable (0–1 range). For corpora > 100k chunks, switch `use_ivf=True` for approximate search.

### Why MMR?

Top-K can return near-duplicate chunks from the same paragraph. MMR trades a small relevance penalty for diversity — the final context window contains genuinely different information, which consistently reduces hallucination.

---

## Benchmarks

Tested on a 200-page financial report (PDF), Apple M2, 16 GB RAM:

| Stage | Time | Notes |
|-------|------|-------|
| Ingestion + chunking | ~1.5s | ~1,400 chunks at 512 chars |
| BGE embedding (CPU) | ~18s | bge-small, batch=32 |
| FAISS index build | < 0.1s | IndexFlatIP, exact search |
| Query embedding | ~0.05s | Single vector |
| FAISS retrieval (top-5) | < 1ms | Exact, 1,400 vectors |
| LLM generation | ~2–4s | Claude Haiku, streaming |

Second query (index loaded from disk): **< 0.1s** for retrieval.

---

## Extending This Project

| Extension | Where to add |
|-----------|-------------|
| Re-ranker (cross-encoder) | New `reranker.py`, call after `vector_store.search()` |
| Hybrid search (BM25 + dense) | Add `bm25.py`, merge scores in `retriever.py` |
| Multi-vector (ColBERT) | Replace `EmbeddingModel.encode_documents()` |
| Evaluation (RAGAS) | Add `eval/` directory, call `rag.retrieve_only()` |
| Metadata filtering | Extend `VectorStore.search()` with pre-filter on `doc.metadata` |

---

## Related Projects in This Series

| Repo | Description |
|------|-------------|
| `llm-eval-suite` | Hallucination + faithfulness evaluation harness |
| `multi-agent-research-assistant` | LangGraph agent with this RAG pipeline as a tool |
| `llm-finetuning-playbook` | SFT + DPO on domain-specific data |

---

## License

MIT — use freely, attribution appreciated.
