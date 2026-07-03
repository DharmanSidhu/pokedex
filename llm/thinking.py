"""
Thinking mode classifier and coordinator.

Classifies user queries to decide whether to activate Qwen's thinking mode
(for complex strategic tasks) or keep it disabled (for factual lookups).
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Keywords that suggest complex strategy/team-building queries requiring deep reasoning
STRATEGY_KEYWORDS = [
    "build", "team", "counter", "strategy", "best set", "vgc", "competitive",
    "synergy", "matchup against", "how to beat", "teammate", "combo", "core",
    "threat", "meta", "gameplan", "playstyle", "win condition"
]

def classify_query_thinking_mode(query: str) -> Tuple[bool, int]:
    """
    Analyze user query and return whether thinking mode is required and the token cap.

    Args:
        query: User message.

    Returns:
        Tuple[bool, int]: (enable_thinking, max_tokens)
    """
    query_lower = query.lower()
    
    # Check if query matches any competitive/strategy keywords
    enable_thinking = False
    for kw in STRATEGY_KEYWORDS:
        if kw in query_lower:
            enable_thinking = True
            break
            
    # Also look at length: very long or complex strategic queries might need thinking
    if len(query_lower.split()) > 20 and ("best" in query_lower or "why" in query_lower or "should" in query_lower):
        enable_thinking = True

    # Token cap defaults
    # Factual lookup: 1024 tokens max (non-thinking, fast response)
    # Strategic reasoning: 8192 tokens max (to avoid runaway loops and allow thinking output space)
    max_tokens = 8192 if enable_thinking else 1024
    
    logger.info(f"Query classification: thinking_mode={enable_thinking}, max_tokens={max_tokens}")
    
    return enable_thinking, max_tokens
