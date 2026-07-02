# RotoDex v2 — Improvement Plan

## Summary

Five major changes: rename to RotoDex, fix UI bugs, revamp memory to keyword-triggered, expand to all Pokemon, and structure LLM responses.

---

## 1. Rename to "RotoDex" + Pixel Font Everywhere

> [!IMPORTANT]
> Every reference to "Dexter" or "Pokedex OS" becomes "RotoDex". The `Press Start 2P` pixel font currently only applies to `.retro-text` / `.retro-title` classes. It needs to be the **global default** for all UI elements.

### Files to change:

#### [MODIFY] [pokedex.css](file:///Users/as-mac-1242/pkdx/app/styles/pokedex.css)
- Change `.stApp` font-family from `Share Tech Mono` → `'Press Start 2P', monospace` as the global default
- Add `VT323` as a secondary readable font for longer body text (more legible at small sizes than Press Start 2P)
- Update Google Fonts import to include `VT323`

#### [MODIFY] [main.py](file:///Users/as-mac-1242/pkdx/app/main.py)
- Boot screen: "POKEDEX OS v1.0" → "ROTODEX OS v2.0", "(c) SILPH CO." → "(c) ROTOM LABS"
- Page title: "AI Pokédex" → "RotoDex"
- All "Dexter" references → "RotoDex"
- Sprite Deck title → "RotoDex Sprite Deck"

#### [MODIFY] [sidebar.py](file:///Users/as-mac-1242/pkdx/app/sidebar.py)
- "DEXTER OS v1.0" → "ROTODEX v2.0"
- "Dexter remembers..." → "RotoDex remembers..."

#### [MODIFY] [prompts.py](file:///Users/as-mac-1242/pkdx/llm/prompts.py)
- System prompt: "You are Dexter" → "You are RotoDex"
- Memory extraction examples: "Hey Dexter" → "Hey RotoDex"

---

## 2. Fix Duplicate Chat Message Display Bug

> [!WARNING]
> The `st.rerun()` on line 181 of `main.py` causes messages to render twice: once during the streaming loop, then again after rerun re-renders the full chat history. This is the root cause of the duplication.

#### [MODIFY] [main.py](file:///Users/as-mac-1242/pkdx/app/main.py)
- **Remove `st.rerun()` after streaming completes** — the chat history loop at lines 120-134 already renders all prior messages on each Streamlit cycle. The `st.rerun()` was intended to force sprite rendering, but it causes the just-written message to appear a second time.
- Instead, render the sprite/cry/source footer **inline** right after streaming finishes (inside the same `st.chat_message("assistant")` block), so no rerun is needed.

---

## 3. Revamp User Memory — Keyword-Triggered Only

> [!IMPORTANT]
> Current behavior: extracts facts from **every** user message via an LLM call, which is slow, wasteful, and saves irrelevant info. New behavior: memory extraction only fires when the user explicitly triggers it with a keyphrase.

### Trigger keyphrases (case-insensitive):
- `"remember this"`, `"remember that"`, `"save this"`, `"my name is"`, `"i am"`, `"my favorite"`, `"my team is"`, `"i prefer"`, `"note this"`

#### [MODIFY] [chat.py](file:///Users/as-mac-1242/pkdx/app/chat.py)
- In `finalize_turn()`: check if user message contains any trigger keyphrase before calling `self.extractor.extract_and_store_facts()`
- If no keyphrase detected → skip extraction entirely (no LLM call)
- If keyphrase detected → run extraction and show a toast "RotoDex committed X facts to memory"

#### [MODIFY] [prompts.py](file:///Users/as-mac-1242/pkdx/llm/prompts.py)
- Update `MEMORY_EXTRACTION_PROMPT` examples to reflect the new trigger-based approach
- Add instruction: "The user has explicitly asked you to remember something. Extract ONLY the specific facts they want remembered."

#### [MODIFY] [sidebar.py](file:///Users/as-mac-1242/pkdx/app/sidebar.py)
- Update help text: "Say 'remember this' or 'my name is...' in chat to save facts"

---

## 4. Expand Knowledge Base — All Pokemon (Hybrid Approach)

> [!IMPORTANT]
> Decision: **Hybrid local + live fetch**. Pre-building a vector store for 1025+ Pokemon would take hours for the ETL and produce a massive ChromaDB. Instead:
> - **Local KB**: Keep the pre-built vector store for whatever was ETL'd (default Gen 1). Used for RAG semantic search.
> - **Live PokeAPI fallback**: If a Pokemon is NOT in the local KB, fetch its data live from PokeAPI (cached in SQLite on first hit). This covers all 1025 Pokemon instantly with zero pre-build time.

#### [MODIFY] [chat.py](file:///Users/as-mac-1242/pkdx/app/chat.py)
- In `process_chat_turn()`, after RAG retrieval: if matched Pokemon exist but retrieval chunks are empty/low-quality (distance > 0.8), call `poke_fetcher.fetch_full_pokemon()` as a live fallback
- Format the live-fetched data into the same context string format and inject it into the LLM prompt
- Mark source as "PokeAPI (Live)" in rendering metadata

#### [MODIFY] [retriever.py](file:///Users/as-mac-1242/pkdx/rag/retriever.py)
- Add a method `has_pokemon_in_store(name)` that checks if the vector DB has indexed data for a given Pokemon name
- Used by the chat orchestrator to decide whether to fall back to live API

#### [MODIFY] [etl.py](file:///Users/as-mac-1242/pkdx/data_pipeline/etl.py)
- Change `DEFAULT_POKEMON_LIMIT` from `151` → `1025` (all Pokemon through Gen 9)
- This only affects **future** ETL runs — existing users keep their current KB and get live fallback for missing Pokemon

---

## 5. Structured Response Formatting

> [!IMPORTANT]
> Currently the system prompt gives the LLM no guidance on **how** to structure answers. Pokemon-specific queries should get organized sections, while general questions get a normal conversational answer.

#### [MODIFY] [prompts.py](file:///Users/as-mac-1242/pkdx/llm/prompts.py)
- Add response formatting instructions to `SYSTEM_PROMPT_TEMPLATE`:

```
Response Formatting Rules:
- For Pokemon-specific factual queries (stats, abilities, evolution, typing):
  Structure your response with clear sections using markdown headers:
  ## [Pokemon Name] — Quick Profile
  **Type:** ...  |  **Abilities:** ...
  **Base Stats:** HP/Atk/Def/SpA/SpD/Spe
  Then answer the specific question below the profile.

- For strategy/team-building queries:
  Use bullet points, highlight key moves/items/EVs.

- For general or conversational queries:
  Answer naturally without forced structure.

- Always end factual answers with a [Source] tag.
```

---

## Verification Plan

### Manual Verification
1. Launch the app and confirm all UI text says "RotoDex" with pixel font
2. Send 3 messages — confirm no duplicate message display
3. Send "What is Garchomp?" (Gen 4 Pokemon, not in Gen 1 KB) — confirm live fallback works
4. Send "remember this: my name is Ash" — confirm fact is saved
5. Send "What is Pikachu's base speed?" — confirm NO memory extraction runs
6. Confirm response has structured sections with stats/type/abilities

