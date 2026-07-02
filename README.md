# AI Pokedex 🔴

A locally-run, RAG-powered AI Pokedex that answers natural-language Pokemon questions using a local LLM on Apple Silicon. No cloud APIs, no fine-tuning — just accurate, source-grounded answers with persistent user memory.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3-orange)
![MLX](https://img.shields.io/badge/MLX-Local%20LLM-green)

## Features

- **🤖 Local LLM** — Qwen3.5-9B (4-bit quantized) via MLX, runs entirely on your Mac
- **📚 RAG Pipeline** — Retrieval-Augmented Generation from PokeAPI, Smogon, and Bulbapedia
- **🧠 Persistent Memory** — Remembers your name, team, playstyle across sessions
- **🎯 Deterministic Tools** — Type matchups and stat calcs use hardcoded logic, never hallucinated
- **🎮 Pokedex UI** — Authentic device-styled Streamlit interface with sprites, cries, and retro fonts
- **📊 Modes** — General Q&A, VGC Team Builder, Trivia Quiz, Type Matchup Calculator

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4) — 16GB RAM minimum
- **Python 3.11+**
- **~6 GB disk space** for the quantized LLM model

## Quick Start

### 1. Clone & Install

```bash
cd pkdx
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the LLM Model (one-time, ~5.6 GB)

```bash
huggingface-cli download mlx-community/Qwen3.5-9B-MLX-4bit
```

### 3. Build the Knowledge Base (one-time, ~10 min for Gen 1)

```bash
python -m data_pipeline.etl
```

This fetches Pokemon data from PokeAPI, Smogon, and Bulbapedia, then chunks and embeds it into the local ChromaDB vector store.

### 4. Launch

```bash
chmod +x start.sh
./start.sh
```

This starts both the MLX LLM server and the Streamlit app. Open `http://localhost:8501` in your browser.

## Project Structure

```
pkdx/
├── data_pipeline/          # ETL: PokeAPI, Smogon, Bulbapedia fetchers + SQLite cache
│   ├── pokeapi_fetcher.py  # PokeAPI v2 client with rate limiting
│   ├── smogon_fetcher.py   # Smogon usage stats parser
│   ├── bulbapedia_fetcher.py # Bulbapedia MediaWiki API client
│   ├── etl.py              # Normalize all sources → document chunks
│   └── db.py               # SQLite cache manager
├── rag/                    # Retrieval-Augmented Generation
│   ├── chunker.py          # Semantic-aware document chunking
│   ├── embedder.py         # all-MiniLM-L6-v2 embedding wrapper
│   ├── vector_store.py     # ChromaDB persistent collection
│   └── retriever.py        # Hybrid search: keyword + semantic + rerank
├── memory/                 # Persistent user memory
│   ├── store.py            # Separate ChromaDB collection for user facts
│   ├── extractor.py        # LLM-based fact extraction from conversations
│   └── conflict.py         # Conflict resolution (overwrite stale facts)
├── llm/                    # LLM serving & interaction
│   ├── server_config.py    # MLX server launch configuration
│   ├── client.py           # OpenAI-compatible client wrapper
│   ├── prompts.py          # System prompt templates
│   ├── tools.py            # Deterministic tools (type calc, stat compare)
│   └── thinking.py         # Thinking-mode toggle logic
├── app/                    # Streamlit frontend
│   ├── main.py             # Entry point + boot animation
│   ├── chat.py             # Chat orchestration
│   ├── sidebar.py          # Mode selector + memory panel
│   ├── components.py       # Sprite cards, cry player, type badges
│   └── styles/
│       └── pokedex.css     # Authentic Pokedex device CSS
├── eval/                   # Evaluation suite
│   ├── test_questions.json # Fixed test Q&A with ground truth
│   ├── eval_rag.py         # RAG accuracy scoring
│   └── eval_memory.py      # Memory system checks
├── chroma_pokemon_kb/      # Persistent vector DB — Pokemon knowledge (auto-created)
├── chroma_user_memory/     # Persistent vector DB — User memory (auto-created)
├── cache/                  # SQLite cache files (auto-created)
├── requirements.txt
├── start.sh                # One-command launcher
└── README.md
```

## Data Persistence

Both ChromaDB directories persist to disk automatically:

- **`chroma_pokemon_kb/`** — Pokemon knowledge base. Rebuilt by running `python -m data_pipeline.etl`.
- **`chroma_user_memory/`** — User memory (your name, team, preferences). Survives app restarts. Manageable via the sidebar Memory panel.

## Memory Budget

| Component | RAM Usage |
|---|---|
| Qwen3.5-9B-MLX-4bit | ~6–7 GB |
| all-MiniLM-L6-v2 | ~80 MB |
| ChromaDB (2 collections) | ~100 MB |
| Streamlit + Python | ~300 MB |
| **Total** | **~7.5 GB** (leaves ~4 GB for macOS + other apps) |

## Evaluation

```bash
# Run RAG accuracy tests
python -m eval.eval_rag

# Run memory system tests
python -m eval.eval_memory
```

## License

For personal, local use only. Pokemon data sourced from PokeAPI (fair use), Smogon (community data), and Bulbapedia (CC BY-NC-SA).
