"""RAG pipeline with REAL answer generation."""
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Import from other scripts
sys.path.append(str(Path(__file__).parent))
from retrieve import Retriever
from generate_real_answers import load_model, generate_answer

PROCESSED_DIR = Path("data/processed")

RAG_PROMPT_TEMPLATE = """Answer the question using ONLY the information from these documents. If the documents don't contain the answer, say "I don't have enough information."

Documents:
{evidence}

Question: {question}

Answer:"""

def format_evidence(retrieved_docs):
    """Format retrieved documents for prompt."""
    evidence_text = ""
    for i, doc in enumerate(retrieved_docs, 1):
        text = doc['passage'].get('text', '')
        evidence_text += f"[{i}] {text}\n\n"
    return evidence_text.strip()

def generate_rag_answer(question, evidence):
    """Generate answer using RAG (with evidence)."""
    # Format prompt with evidence
    prompt = RAG_PROMPT_TEMPLATE.format(
        evidence=evidence,
        question=question
    )
    
    # Generate using same model
    model, tokenizer = load_model()
    
    import torch
    
    # Tokenize (longer max length for evidence)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

def run_rag_pipeline(baseline_file, output_file):
    """Run RAG pipeline on baseline answers."""
    # Load retriever
    print("\n📚 Loading retrieval system...")
    retriever = Retriever(
        index_path=PROCESSED_DIR / "wiki.index",
        passages_path=PROCESSED_DIR / "passages.jsonl"
    )
    
    # Load baseline answers
    baseline_answers = []
    with open(baseline_file, encoding="utf-8") as f:
        for line in f:
            baseline_answers.append(json.loads(line))
    
    print(f"\n{'='*70}")
    print(f"Generating {len(baseline_answers)} RAG answers")
    print(f"{'='*70}\n")
    
    results = []
    
    for item in tqdm(baseline_answers, desc="RAG generation"):
        question = item['question']
        
        # Retrieve documents
        retrieved = retriever.retrieve(question, k=3)
        
        # Format evidence
        evidence = format_evidence(retrieved)
        
        # Generate RAG answer
        try:
            rag_answer = generate_rag_answer(question, evidence)
        except Exception as e:
            print(f"\nError: {e}")
            rag_answer = "[Error generating RAG answer]"
        
        result = {
            "id": item["id"],
            "question": question,
            "baseline_answer": item["answer"],
            "rag_answer": rag_answer,
            "retrieved_docs": [
                {
                    "text": d["passage"]["text"],
                    "score": d["score"],
                    "title": d["passage"].get("title", "")
                }
                for d in retrieved
            ],
            "ground_truth": item.get("ground_truth")
        }
        results.append(result)
    
    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*70}")
    print(f"✓ Generated {len(results)} RAG answers")
    print(f"✓ Saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Show comparison
    print("Sample comparisons:")
    for i, r in enumerate(results[:3], 1):
        print(f"\n{i}. Q: {r['question'][:60]}...")
        print(f"   Baseline: {r['baseline_answer']}")
        print(f"   RAG: {r['rag_answer']}")
        print(f"   Retrieved: {len(r['retrieved_docs'])} documents")

def main():
    """Generate RAG answers."""
    print("\n" + "="*70)
    print("RAG ANSWER GENERATION")
    print("="*70)
    print("\nThis will generate RAG answers using retrieved evidence.")
    print("Make sure you've run generate_real_answers.py first!\n")
    
    # Check if baseline exists
    baseline_file = PROCESSED_DIR / "baseline_answers_truthfulqa.jsonl"
    if not baseline_file.exists():
        print("❌ Baseline answers not found!")
        print("Run first: python scripts/generate_real_answers.py")
        return
    
    cont = input("Continue? (y/n): ").strip().lower()
    if cont != 'y':
        return
    
    # Generate RAG answers
    run_rag_pipeline(
        baseline_file,
        PROCESSED_DIR / "rag_answers_truthfulqa.jsonl"
    )
    
    # SQuAD if exists
    squad_baseline = PROCESSED_DIR / "baseline_answers_squad.jsonl"
    if squad_baseline.exists():
        print("\n" + "="*70)
        print("SQUAD DATASET")
        print("="*70)
        run_rag_pipeline(
            squad_baseline,
            PROCESSED_DIR / "rag_answers_squad.jsonl"
        )
    
    print("\n" + "="*70)
    print("✓ ALL DONE!")
    print("="*70)
    print("\nYou now have:")
    print("✓ Baseline answers (without retrieval)")
    print("✓ RAG answers (with retrieval)")
    print("\nNext steps:")
    print("1. Refresh Streamlit dashboard")
    print("2. Compare baseline vs RAG")
    print("3. Start annotation for evaluation")

if __name__ == "__main__":
    main()
