"""
Visual components for the Streamlit Pokedex UI.

Handles rendering sprites, playing cries, displaying stat bars,
colored type badges, and source attributions.
"""

import streamlit as st
from typing import List, Set, Dict, Optional

# Type colors mapping for Pokemon type pills
TYPE_COLORS: Dict[str, str] = {
    "normal": "#A8A77A",
    "fire": "#EE8130",
    "water": "#6390F0",
    "electric": "#F7D02C",
    "grass": "#7AC74C",
    "ice": "#96D9D6",
    "fighting": "#C22E28",
    "poison": "#A33EA1",
    "ground": "#E2BF65",
    "flying": "#A98FF3",
    "psychic": "#F95587",
    "bug": "#A6B91A",
    "rock": "#B6A136",
    "ghost": "#735797",
    "dragon": "#6F35FC",
    "dark": "#705746",
    "steel": "#B7B7D4",
    "fairy": "#D685AD"
}

def render_type_badge(type_name: str):
    """Render a colored type pill."""
    t_name = type_name.lower().strip()
    color = TYPE_COLORS.get(t_name, "#777777")
    st.markdown(
        f'<span class="type-pill" style="background-color: {color};">{t_name}</span>',
        unsafe_allow_url=True,
        unsafe_allow_html=True
    )

def render_type_badges(types: List[str]):
    """Render multiple type badges in a row."""
    badges_html = ""
    for t in types:
        t_name = t.lower().strip()
        color = TYPE_COLORS.get(t_name, "#777777")
        badges_html += f'<span class="type-pill" style="background-color: {color};">{t_name}</span>'
    st.markdown(
        f'<div style="display: flex; flex-wrap: wrap;">{badges_html}</div>',
        unsafe_allow_html=True
    )

def render_pokemon_card(name: str, sprite_url: Optional[str], types: List[str] = None):
    """Render a retro-styled Pokemon display card with sprite and types."""
    title_name = name.replace("-", " ").title()
    
    card_html = f"""
    <div class="pokemon-card">
        <div class="retro-text" style="font-size: 11px; margin-bottom: 5px;">{title_name}</div>
    """
    
    if sprite_url:
        card_html += f'<img src="{sprite_url}" width="120" style="margin: 0 auto; display: block;"/>'
    else:
        # Placeholder or empty space if no sprite
        card_html += '<div style="height: 120px; display: flex; align-items: center; justify-content: center; color: #888;">No Sprite</div>'
        
    card_html += "</div>"
    
    st.markdown(card_html, unsafe_allow_html=True)
    if types:
        render_type_badges(types)

def render_cry_player(cry_url: Optional[str]):
    """Embed an HTML5 audio player to play the Pokemon cry."""
    if not cry_url:
        return
        
    audio_html = f"""
    <div style="margin-top: 5px; margin-bottom: 10px;">
        <span class="retro-text" style="font-size: 9px; vertical-align: middle; margin-right: 8px;">🔊 CRY:</span>
        <audio controls src="{cry_url}" style="height: 25px; vertical-align: middle; max-width: 150px; filter: sepia(1) saturate(5) hue-rotate(180deg) invert(1);">
            Your browser does not support the audio element.
        </audio>
    </div>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def render_stat_bar(stat_name: str, value: int, max_val: int = 255):
    """Render a progress-bar chart for base stats."""
    # Scale width percentage
    pct = min(100.0, (value / max_val) * 100.0)
    
    # Color coding stats (Green for high, Orange for mid, Red for low)
    if value >= 120:
        bar_color = "var(--pokedex-green)"
    elif value >= 75:
        bar_color = "var(--pokedex-yellow)"
    else:
        bar_color = "var(--pokedex-red)"
        
    stat_html = f"""
    <div style="margin-bottom: 6px;">
        <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 2px;">
            <span>{stat_name}</span>
            <span style="font-weight: bold; color: {bar_color};">{value}</span>
        </div>
        <div class="stat-bar-container">
            <div class="stat-bar-fill" style="width: {pct}%; background-color: {bar_color};"></div>
        </div>
    </div>
    """
    st.markdown(stat_html, unsafe_allow_html=True)

def render_source_footer(sources: Set[str], user_memory_applied: bool = False):
    """Display the expandable 'Sources used' footer for RAG query grounding."""
    if not sources and not user_memory_applied:
        return
        
    source_labels = []
    for s in sorted(list(sources)):
        if s.lower() == "pokeapi":
            source_labels.append("PokeAPI Database")
        elif s.lower() == "smogon":
            source_labels.append("Smogon Strategy Guides")
        elif s.lower() == "bulbapedia":
            source_labels.append("Bulbapedia Lore Archives")
        else:
            source_labels.append(s.title())
            
    if user_memory_applied:
        source_labels.append("User Memory Collection (Personalized)")

    footer_content = " • ".join(source_labels)
    
    st.markdown(
        f'<div class="source-footer">💾 SOURCE LOGS: {footer_content}</div>',
        unsafe_allow_html=True
    )


