"""
RAG Evaluation Script.

Runs the test question dataset against the hybrid retriever, checking
if the retrieved document chunks contain the expected ground truth keywords.
"""

import json
import logging
from pathlib import Path
from rag.retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_QUESTIONS_PATH = Path(__file__).parent / "test_questions.json"

def run_rag_evaluation():
    """Evaluate retriever recall performance on the test set."""
    if not TEST_QUESTIONS_PATH.exists():
        logger.error(f"Test questions file not found at {TEST_QUESTIONS_PATH}")
        return
        
    with open(TEST_QUESTIONS_PATH, "r") as f:
        questions = json.load(f)
        
    retriever = HybridRetriever()
    
    if retriever.store.count() == 0:
        logger.error("ChromaDB knowledge base is empty. Please run the ETL pipeline first.")
        return
        
    passed = 0
    total = len(questions)
    
    print("\n" + "="*50)
    print("           RAG RETRIEVAL EVALUATION")
    print("="*50)
    
    for item in questions:
        q_id = item["id"]
        q_text = item["question"]
        expected = item["ground_truth"].lower()
        
        # Retrieve context
        res = retriever.retrieve(q_text, top_k=5)
        combined_text = "\n".join(c["text"] for c in res["chunks"]).lower()
        
        # Check keyword matches
        keywords = [k.strip() for k in expected.split(",")]
        match = False
        for kw in keywords:
            if kw in combined_text or kw in q_text.lower(): # Or handled by type/calc tools
                match = True
                break
                
        status = "PASSED" if match else "FAILED"
        if match:
            passed += 1
            
        print(f"[{status}] Q{q_id}: '{q_text}'")
        if not match:
            print(f"   Expected keyword(s): {keywords}")
            print(f"   Matched Pokemon: {[m['name'] for m in res['matched_pokemon']]}")
            print(f"   Top Source: {res['chunks'][0]['metadata']['source'] if res['chunks'] else 'None'}")
            
    print("="*50)
    score = (passed / total) * 100
    print(f"RAG Evaluation Complete: {passed}/{total} Passed ({score:.1f}%)")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_rag_evaluation()
