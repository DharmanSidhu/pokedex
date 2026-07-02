"""
Main entry point for the Streamlit RotoDex Web App.

Loads custom CSS, renders the authentic Pokedex frame, manages
session state, and coordinates the sidebar and screen components.
"""

import time
import os
import streamlit as st

# Set page config FIRST before any imports or styling
st.set_page_config(
    page_title="RotoDex",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.sidebar import render_sidebar
from app.chat import ChatOrchestrator
from app.components import (
    render_pokemon_card,
    render_cry_player,
    render_source_footer,
    render_type_badges,
    render_stat_bar
)
from data_pipeline.pokeapi_fetcher import PokeAPIFetcher
from llm.tools import type_matchup_summary

# Load CSS stylesheet
CSS_PATH = os.path.join(os.path.dirname(__file__), "styles", "pokedex.css")
with open(CSS_PATH, "r") as f:
    custom_css = f.read()
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# Initialize singletons in session state
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = ChatOrchestrator()
if "poke_fetcher" not in st.session_state:
    st.session_state.poke_fetcher = PokeAPIFetcher()
if "booted" not in st.session_state:
    st.session_state.booted = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

orchestrator = st.session_state.orchestrator
poke_fetcher = st.session_state.poke_fetcher

# --- 1. Boot-up Animation ---
if not st.session_state.booted:
    boot_container = st.empty()
    with boot_container.container():
        st.markdown("""
        <div class="pokedex-container boot-anim">
            <div class="top-lights-panel">
                <div class="big-blue-lens"></div>
                <div class="small-light light-red"></div>
                <div class="small-light light-yellow"></div>
                <div class="small-light light-green"></div>
            </div>
            <div class="pokedex-screen-frame">
                <div class="pokedex-screen" style="height: 350px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                    <div class="scanlines"></div>
                    <div class="retro-text" style="font-size: 16px; margin-bottom: 20px; text-align: center; color: var(--pokedex-yellow);">
                        ROTODEX OS v2.0
                    </div>
                    <div class="retro-text" style="font-size: 12px; margin-bottom: 10px; color: var(--pokedex-cyan-glow);">
                        INITIALIZING LOCAL MEMORY DB...
                    </div>
                    <div class="retro-text" style="font-size: 12px; margin-bottom: 30px; color: var(--pokedex-cyan-glow);">
                        ESTABLISHING LLM INSTANCE CONNECTION...
                    </div>
                    <div class="retro-text" style="font-size: 9px; color: #888;">
                        ⚡ ROTOM LABS 2026
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(2.5)
    boot_container.empty()
    st.session_state.booted = True

# --- 2. Sidebar Integration ---
selected_mode = render_sidebar(orchestrator.memory_store)

# --- 3. Main Device Layout ---
st.markdown("""
<div class="top-lights-panel" style="margin-bottom: 10px;">
    <div class="big-blue-lens"></div>
    <div class="small-light light-red"></div>
    <div class="small-light light-yellow"></div>
    <div class="small-light light-green"></div>
</div>
""", unsafe_allow_html=True)

screen_col, stats_col = st.columns([2, 1])

with screen_col:
    st.markdown(f'<div class="retro-title">{selected_mode.upper()} SCREEN</div>', unsafe_allow_html=True)

    if selected_mode in ("General Q&A", "VGC Team Builder", "Trivia Quiz"):

        # Display chat history
        chat_container = st.container(height=420)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    # Render extras for past assistant messages
                    if msg["role"] == "assistant" and "meta" in msg:
                        meta = msg["meta"]
                        if meta.get("matched_pokemon"):
                            pokemon_name = meta["matched_pokemon"][0]
                            poke_data = poke_fetcher.fetch_pokemon(pokemon_name)
                            if poke_data:
                                render_pokemon_card(poke_data["name"], poke_data["sprites"]["official_artwork"], poke_data["types"])
                                render_cry_player(poke_data["cries"]["latest"])
                        render_source_footer(meta.get("sources", set()), meta.get("user_memory_applied", False))

        # Chat input
        prompt_placeholder = "Ask RotoDex about stats, matchups, strategy..."
        if selected_mode == "VGC Team Builder":
            prompt_placeholder = "Ask RotoDex to build a team or analyze VGC strategy..."
        elif selected_mode == "Trivia Quiz":
            prompt_placeholder = "Ask RotoDex for trivia or quiz questions..."

        if user_prompt := st.chat_input(placeholder=prompt_placeholder):
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})

            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.write(user_prompt)

            # Generate and stream assistant response
            with chat_container:
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()

                    try:
                        stream_response, meta = orchestrator.process_chat_turn(
                            user_prompt,
                            st.session_state.chat_history
                        )

                        # Stream chunks
                        full_resp = ""
                        for chunk in stream_response:
                            full_resp += chunk
                            response_placeholder.write(full_resp)

                        # Save to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": full_resp,
                            "meta": meta
                        })

                        # Render sprite/cry/sources INLINE (no rerun needed)
                        if meta.get("matched_pokemon"):
                            pokemon_name = meta["matched_pokemon"][0]
                            poke_data = poke_fetcher.fetch_pokemon(pokemon_name)
                            if poke_data:
                                render_pokemon_card(poke_data["name"], poke_data["sprites"]["official_artwork"], poke_data["types"])
                                render_cry_player(poke_data["cries"]["latest"])
                        render_source_footer(meta.get("sources", set()), meta.get("user_memory_applied", False))

                        # Memory extraction (only if keyphrase detected)
                        new_facts = orchestrator.finalize_turn(user_prompt, full_resp)
                        if new_facts:
                            st.toast(f"⚡ RotoDex committed {len(new_facts)} fact(s) to memory!")

                    except ConnectionError as ce:
                        st.error(str(ce))
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

    elif selected_mode == "Type Matchup Calculator":
        st.caption("Bypasses LLM completely. Uses mathematical type charts for guaranteed accuracy.")

        calc_types = st.multiselect(
            "Select Pokemon Defending Type(s):",
            options=[
                "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground",
                "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"
            ],
            max_selections=2,
            default=["Fire", "Flying"]
        )

        if calc_types:
            types_lower = [t.lower() for t in calc_types]
            summary = type_matchup_summary(types_lower)

            st.markdown("### Defensive Profile:")
            render_type_badges(calc_types)

            col1, col2 = st.columns(2)
            with col1:
                if summary["4x"]:
                    st.error(f"4x Weak: {', '.join(summary['4x'])}")
                if summary["2x"]:
                    st.warning(f"2x Weak: {', '.join(summary['2x'])}")
            with col2:
                if summary["0.5x"]:
                    st.success(f"Resists (0.5x): {', '.join(summary['0.5x'])}")
                if summary["0.25x"]:
                    st.success(f"Double Resists (0.25x): {', '.join(summary['0.25x'])}")
                if summary["0x"]:
                    st.info(f"Immune (0x): {', '.join(summary['0x'])}")

with stats_col:
    st.markdown('<div class="retro-title">RotoDex Sprite Deck</div>', unsafe_allow_html=True)

    # Show last matched Pokemon's sprite + stats
    last_pokemon = None
    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "assistant" and "meta" in msg:
            matched = msg["meta"].get("matched_pokemon")
            if matched:
                last_pokemon = matched[0]
                break

    if last_pokemon:
        poke_data = poke_fetcher.fetch_pokemon(last_pokemon)
        if poke_data:
            st.markdown(f"**Species Found:** {poke_data['name'].title()}")
            render_pokemon_card(poke_data["name"], poke_data["sprites"]["official_artwork"], poke_data["types"])
            render_cry_player(poke_data["cries"]["latest"])

            st.markdown("**Base Stats Profile:**")
            for stat_name, val in poke_data["base_stats"].items():
                render_stat_bar(stat_name.replace("-", " ").title(), val)
    else:
        st.markdown("""
        <div style="background-color: rgba(255,255,255,0.02); border: 2px dashed #444; border-radius: 8px; padding: 25px; text-align: center; color: #888;">
            <div class="retro-text" style="font-size: 9px; line-height: 1.8;">
                NO ACTIVE POKEMON IN CONTEXT
            </div>
            <div style="font-size: 14px; margin-top: 10px; font-family: var(--font-pixel-body);">
                Mention a Pokemon name to load its sprite and stats here.
            </div>
        </div>
        """, unsafe_allow_html=True)
