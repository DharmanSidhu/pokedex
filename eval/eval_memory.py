"""
Evaluation checks for the User Memory System.

Programmatically tests:
1. Adding facts to the user memory ChromaDB.
2. Retrieval of relevant facts.
3. Conflict resolution (overwriting/deprecating stale facts in same categories).
"""

import logging
import time
from memory.store import UserMemoryStore
from memory.conflict import resolve_memory_conflicts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_memory_evaluation():
    print("\n" + "="*50)
    print("         USER MEMORY SYSTEM EVALUATION")
    print("="*50)
    
    # Initialize store pointing to a test-specific directory to avoid polluting actual user memory
    test_store = UserMemoryStore(path="./chroma_test_user_memory")
    test_store.clear()
    
    try:
        # Test 1: Basic insertion
        print("[TEST 1] Testing basic fact insertion...")
        fid_1 = test_store.add_fact("My name is Ash", "identity")
        fid_2 = test_store.add_fact("My favorite Pokemon is Charizard", "preference")
        
        all_facts = test_store.list_all_facts(include_deprecated=False)
        assert len(all_facts) == 2, f"Expected 2 facts, got {len(all_facts)}"
        print("  ✓ Basic insertion successful.")
        
        # Test 2: Retrieval matching
        print("[TEST 2] Testing semantic/relevant fact retrieval...")
        relevant = test_store.get_relevant_facts("What's my name again?")
        assert len(relevant) >= 1, "Expected to retrieve at least one fact about identity."
        assert "Ash" in relevant[0]["fact"], f"Expected to find 'Ash', got: {relevant[0]['fact']}"
        print("  ✓ Semantic retrieval successful.")

        # Test 3: Conflict detection and resolution (overwrite old fact)
        print("[TEST 3] Testing conflict resolution (name update)...")
        # Simulate new name fact
        new_fact = "My name is now Red"
        category = "identity"
        
        conflicting = resolve_memory_conflicts(new_fact, category, all_facts)
        assert fid_1 in conflicting, "Expected original name fact to be flagged as conflicting."
        
        # Apply override
        for old_id in conflicting:
            test_store.delete_fact(old_id)
            
        test_store.add_fact(new_fact, category)
        
        updated_facts = test_store.list_all_facts(include_deprecated=False)
        assert len(updated_facts) == 2, f"Expected exactly 2 facts, got {len(updated_facts)}"
        
        names = [f["fact"] for f in updated_facts if f["metadata"]["category"] == "identity"]
        assert len(names) == 1, "Expected only 1 name fact to be active."
        assert "Red" in names[0], f"Expected name to be updated to Red, got {names[0]}"
        print("  ✓ Conflict resolution and database override successful.")
        
        # Test 4: Delete logic
        print("[TEST 4] Testing fact deletion...")
        remaining = test_store.list_all_facts()
        for f in remaining:
            test_store.delete_fact(f["id"])
            
        final_facts = test_store.list_all_facts()
        assert len(final_facts) == 0, "Expected empty memory store after deletion."
        print("  ✓ Deletion successful.")
        
        print("="*50)
        print("Trainer Memory System checks: ALL PASSED")
        print("="*50 + "\n")
        
    finally:
        # Clean up test database files
        test_store.clear()

if __name__ == "__main__":
    run_memory_evaluation()
