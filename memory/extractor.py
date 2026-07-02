"""
User memory extraction orchestrator.

Takes the user's input, calls the local LLM in non-thinking mode to extract
structured facts, resolves conflicts with existing facts, and saves them to ChromaDB.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional
from memory.store import UserMemoryStore
from memory.conflict import resolve_memory_conflicts
from llm.client import LLMClient
from llm.prompts import MEMORY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

class MemoryExtractor:
    """Orchestrates fact extraction, conflict check, and persistence to user memory store."""

    def __init__(self, store: Optional[UserMemoryStore] = None, client: Optional[LLMClient] = None):
        self.store = store or UserMemoryStore()
        self.client = client or LLMClient()

    def extract_and_store_facts(self, user_message: str) -> List[Dict[str, Any]]:
        """
        Extract personal facts from a user message and store them, resolving conflicts.

        Args:
            user_message: Raw text input from user.

        Returns:
            List of newly added facts as dictionaries.
        """
        # 1. Ask the LLM to extract facts in non-thinking mode
        messages = [
            {"role": "system", "content": MEMORY_EXTRACTION_PROMPT},
            {"role": "user", "content": f"User message: \"{user_message}\""}
        ]
        
        try:
            # We use enable_thinking=False for extraction speed and accuracy
            response_text = self.client.chat_completion(
                messages=messages,
                temperature=0.0,  # Deterministic output
                enable_thinking=False,
                bypass_cache=True, # Do not cache extraction output as user input is highly dynamic
            )
            
            # Clean response text from any markdown tags if model outputs them
            response_text = response_text.strip()
            if response_text.startswith("```"):
                # strip code block wrapper
                response_text = re.sub(r'^```(?:json)?\n', '', response_text)
                response_text = re.sub(r'\n```$', '', response_text)
                response_text = response_text.strip()
                
            if not response_text or response_text == "[]":
                return []
                
            extracted_facts = json.loads(response_text)
            if not isinstance(extracted_facts, list):
                logger.warning("Extracted facts is not a list. Skipping.")
                return []
                
        except json.JSONDecodeError as je:
            logger.warning(f"Could not parse memory extraction JSON: {response_text}. Error: {je}")
            return []
        except Exception as e:
            logger.error(f"Error during memory extraction: {e}")
            return []

        # 2. Process each fact, resolve conflicts, and save
        stored_facts = []
        existing_facts = self.store.list_all_facts(include_deprecated=False)

        for item in extracted_facts:
            fact_text = item.get("fact", "").strip()
            category = item.get("category", "preference").strip()
            
            if not fact_text or len(fact_text) < 3:
                continue
                
            # Run conflict resolution logic
            conflicting_ids = resolve_memory_conflicts(fact_text, category, existing_facts)
            
            # Deprecate or delete conflicting facts
            for old_id in conflicting_ids:
                # Mark as deprecated in ChromaDB or delete it.
                # The spec says: "overwrite/deprecate the old fact rather than stacking both."
                # Deprecating is safer as it keeps history but filters it. Let's delete/overwrite it:
                self.store.delete_fact(old_id)
                # Remove from local list so we don't conflict against it again in the same loop
                existing_facts = [f for f in existing_facts if f["id"] != old_id]

            # Add new fact to store
            fact_id = self.store.add_fact(fact_text, category)
            stored_facts.append({
                "id": fact_id,
                "fact": fact_text,
                "category": category
            })
            
        return stored_facts
