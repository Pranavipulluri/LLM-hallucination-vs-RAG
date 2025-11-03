"""Simple script to view results without Streamlit."""
import json
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

def view_results():
    """Display generated answers."""
    
    # Load baseline answers
    baseline_file = PROCESSED_DIR / "baseline_answers_truthfulqa.jsonl"
    if baseline_file.exists():
        print("=" * 70)
        print("BASELINE ANSWERS (First 5)")
        print("=" * 70)
        
        with open(baseline_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                data = json.loads(line)
                print(f"\nQuestion {i+1}: {data['question']}")
                print(f"Answer: {data['answer']}")
                print(f"Ground Truth: {data.get('ground_truth', 'N/A')}")
                print("-" * 70)
    
    # Load RAG answers
    rag_file = PROCESSED_DIR / "rag_answers_truthfulqa.jsonl"
    if rag_file.exists():
        print("\n" + "=" * 70)
        print("RAG ANSWERS (First 5)")
        print("=" * 70)
        
        with open(rag_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                data = json.loads(line)
                print(f"\nQuestion {i+1}: {data['question']}")
                print(f"RAG Answer: {data['rag_answer']}")
                print(f"Retrieved Docs: {len(data.get('retrieved_docs', []))}")
                if data.get('retrieved_docs'):
                    print(f"Top Doc: {data['retrieved_docs'][0]['text'][:100]}...")
                print("-" * 70)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    baseline_count = sum(1 for _ in open(baseline_file, encoding="utf-8")) if baseline_file.exists() else 0
    rag_count = sum(1 for _ in open(rag_file, encoding="utf-8")) if rag_file.exists() else 0
    
    print(f"Baseline answers: {baseline_count}")
    print(f"RAG answers: {rag_count}")
    print(f"\nFiles location: {PROCESSED_DIR}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Install Streamlit: pip install streamlit")
    print("2. Launch dashboard: streamlit run app/streamlit_app.py")
    print("3. Or start annotation using docs/annotation_guidelines.md")

if __name__ == "__main__":
    view_results()
