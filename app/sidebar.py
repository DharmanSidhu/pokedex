"""
Sidebar panel for the Streamlit RotoDex UI.

Contains the application mode selector and the user memory viewer/editor.
"""

import streamlit as st
from memory.store import UserMemoryStore


def render_sidebar(store: UserMemoryStore) -> str:
    """
    Render RotoDex sidebar containing settings, mode select, and memory manager.

    Returns:
        Selected mode string.
    """
    st.sidebar.markdown(
        '<div class="retro-title" style="font-size: 14px; text-align: center; color: var(--pokedex-yellow);">⚡ ROTODEX v1.5</div>',
        unsafe_allow_html=True
    )

    # 1. Mode Selector
    st.sidebar.markdown('<span class="retro-text" style="font-size: 10px;">⚙️ SELECT MODE:</span>', unsafe_allow_html=True)
    mode = st.sidebar.radio(
        label="Mode Selector",
        options=["General Q&A", "VGC Team Builder", "Trivia Quiz", "Type Matchup Calculator"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # 2. User Memory Panel
    st.sidebar.markdown('<span class="retro-text" style="font-size: 10px;">🧠 TRAINER MEMORY:</span>', unsafe_allow_html=True)
    st.sidebar.caption('Say "remember this" or "my name is..." in chat to save facts about yourself.')

    # Render all stored facts
    facts = store.list_all_facts(include_deprecated=False)

    if not facts:
        st.sidebar.info("Memory is empty! Use phrases like 'remember this: ...' or 'my name is ...' in the chat to teach RotoDex about yourself.")
    else:
        # Group facts by category
        categories = {}
        for f in facts:
            cat = f.get("metadata", {}).get("category", "general")
            categories.setdefault(cat, []).append(f)

        for cat, cat_facts in categories.items():
            with st.sidebar.expander(f"{cat.upper()} ({len(cat_facts)})", expanded=True):
                for f in cat_facts:
                    fact_id = f["id"]
                    fact_text = f["fact"]
                    st.write(f"• {fact_text}")
                    if st.button("🗑️ Delete", key=f"del_{fact_id}", help="Remove this fact from memory"):
                        store.delete_fact(fact_id)
                        st.rerun()

    # Manually insert a fact
    with st.sidebar.expander("➕ Add Manual Fact"):
        with st.form("add_fact_form", clear_on_submit=True):
            new_fact = st.text_input("Fact description", placeholder="e.g. My favorite Pokemon is Arcanine")
            new_cat = st.selectbox("Category", ["identity", "team", "preference", "history", "skill_level"])
            submitted = st.form_submit_button("Add Fact")
            if submitted and new_fact:
                store.add_fact(new_fact, new_cat)
                st.toast(f"⚡ Fact committed to {new_cat}!")
                st.rerun()

    st.sidebar.markdown("---")

    # 3. Connection Status
    st.sidebar.markdown('<span class="retro-text" style="font-size: 9px;">📡 STATUS:</span>', unsafe_allow_html=True)
    from llm.server_config import check_server_health
    if check_server_health():
        st.sidebar.success("LLM Server: ONLINE")
    else:
        st.sidebar.error("LLM Server: OFFLINE")
        st.sidebar.caption("Start your LLM server (LM Studio or mlx_lm.server).")

    return mode
