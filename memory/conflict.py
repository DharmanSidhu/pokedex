"""
Conflict resolution rules for user memory.

Ensures that new facts overwrite or deprecate old contradictory facts in the
same category (e.g., changing favorite Pokemon or changing user name).
"""

import logging
from typing import List, Dict, Any, Tuple
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Heuristics for detecting conflicts based on category and word similarities.
# If a new fact conflicts with an old fact, we mark the old fact as deprecated.
def resolve_memory_conflicts(
    new_fact: str,
    new_category: str,
    existing_facts: List[Dict[str, Any]]
) -> List[str]:
    """
    Compare a new fact against existing facts to find conflicts.

    Args:
        new_fact: The new fact text
        new_category: The category of the new fact
        existing_facts: List of existing facts (from list_all_facts)

    Returns:
        List of fact IDs that are in conflict and should be deprecated/deleted.
    """
    conflicting_ids = []
    
    new_fact_lower = new_fact.lower()
    
    for old in existing_facts:
        old_id = old["id"]
        old_fact = old["fact"]
        old_meta = old["metadata"]
        old_category = old_meta.get("category", "")
        
        # Only resolve conflicts within the same category
        if old_category != new_category:
            continue
            
        # Rules based on categories
        conflict_detected = False
        
        # 1. Identity category usually has single value conflicts (e.g. "My name is X" vs "My name is Y")
        if new_category == "identity":
            # If both mention name or both mention age/etc.
            if "name is" in new_fact_lower and "name is" in old_fact.lower():
                conflict_detected = True
            elif "called" in new_fact_lower and "called" in old_fact.lower():
                conflict_detected = True
            elif "trainer level" in new_fact_lower and "trainer level" in old_fact.lower():
                conflict_detected = True
                
        # 2. Favorite Pokemon / Single preference conflicts
        elif new_category == "preference":
            if "favorite pokemon" in new_fact_lower and "favorite pokemon" in old_fact.lower():
                conflict_detected = True
            elif "playstyle is" in new_fact_lower and "playstyle is" in old_fact.lower():
                conflict_detected = True
            elif "prefers" in new_fact_lower and "prefers" in old_fact.lower():
                # check semantic overlap
                ratio = fuzz.token_sort_ratio(new_fact_lower, old_fact.lower())
                if ratio > 65:
                    conflict_detected = True

        # 3. Team conflicts (e.g. if the user says "My team has Charizard" and they had "My team has Charizard" before, or general team overrides)
        elif new_category == "team":
            # If the facts are extremely similar or mention the same pokemon in a different way,
            # or if the user is redefining their whole team (e.g. "My current VGC team is [X, Y, Z]")
            if "current vgc team is" in new_fact_lower and "current vgc team is" in old_fact.lower():
                conflict_detected = True
            else:
                # Check for high keyword overlap
                # e.g., "My team has Charizard" vs "My team has Venusaur" -> if they are too similar
                # we don't want to overwrite if they are different team slots, but if it's the exact same Pokemon, or too similar:
                ratio = fuzz.token_sort_ratio(new_fact_lower, old_fact.lower())
                if ratio > 80:
                    conflict_detected = True

        # General backup rule: high string similarity in same category
        if not conflict_detected:
            ratio = fuzz.token_sort_ratio(new_fact_lower, old_fact.lower())
            # Over 75% similarity in same category indicates a likely update to the same fact
            if ratio > 75:
                conflict_detected = True
                
        if conflict_detected:
            logger.info(
                f"Conflict detected between new fact: '{new_fact}' and old fact: '{old_fact}'. "
                f"Deprecating old fact (ID: {old_id})."
            )
            conflicting_ids.append(old_id)
            
    return conflicting_ids
