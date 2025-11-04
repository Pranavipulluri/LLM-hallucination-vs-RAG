"""Generate COMPLETE real comparison: Baseline vs RAG with actual model predictions."""
import json
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent))
from generate_real_answers import load_model, generate_answer
from retrieve import Retriever

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

def load_questions(limit=50):
    """Load questions from TruthfulQA."""
    questions_file = RAW_DIR / "truthfulqa.jsonl"
    
    if not questions_file.exists():
        print("❌ TruthfulQA data not found!")
        print("Run: python scripts/download_data_fast.py")
        return None
    
    questions = []
    with open(questions_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            data = json.loads(line)
            questions.append({
                "id": data["id"],
                "question": data["question"],
                "ground_truth": data.get("best_answer", "")
            })
    
    return questions

def generate_baseline_predictions(questions):
    """Generate REAL baseline predictions using FLAN-T5."""
    print("\n" + "="*70)
    print("GENERATING BASELINE PREDICTIONS (Real FLAN-T5)")
    print("="*70 + "\n")
    
    # Load model
    print("Loading FLAN-T5 model...")
    model, tokenizer = load_model()
    print("✓ Model loaded\n")
    
    results = []
    
    for q in tqdm(questions, desc="Baseline generation"):
        answer = generate_answer(q["question"])
        
        results.append({
            "id": q["id"],
            "question": q["question"],
            "answer": answer,
            "ground_truth": q["ground_truth"],
            "model": "flan-t5-base"
        })
    
    # Save
    output_file = PROCESSED_DIR / "baseline_predictions_real.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Generated {len(results)} baseline predictions")
    print(f"✓ Saved to: {output_file}")
    
    return results

def generate_rag_predictions(questions):
    """Generate REAL RAG predictions using FLAN-T5 + Retrieval."""
    print("\n" + "="*70)
    print("GENERATING RAG PREDICTIONS (Real FLAN-T5 + Retrieval)")
    print("="*70 + "\n")
    
    # Load model
    print("Loading FLAN-T5 model...")
    model, tokenizer = load_model()
    print("✓ Model loaded")
    
    # Load retriever
    print("Loading retrieval system...")
    try:
        retriever = Retriever(
            index_path=PROCESSED_DIR / "wiki.index",
            passages_path=PROCESSED_DIR / "passages.jsonl"
        )
        print("✓ Retriever loaded\n")
    except Exception as e:
        print(f"❌ Error loading retriever: {e}")
        print("Using baseline answers as RAG (no retrieval available)")
        retriever = None
    
    results = []
    
    for q in tqdm(questions, desc="RAG generation"):
        # Retrieve documents
        if retriever:
            retrieved = retriever.retrieve(q["question"], k=3)
            evidence = "\n".join([f"[{i+1}] {d['passage']['text']}" 
                                 for i, d in enumerate(retrieved)])
        else:
            retrieved = []
            evidence = "No evidence available."
        
        # Generate with evidence
        if evidence and evidence != "No evidence available.":
            prompt = f"""Answer this question using the provided evidence. If the evidence doesn't help, answer based on your knowledge.

Evidence:
{evidence}

Question: {q["question"]}

Answer:"""
        else:
            prompt = f"Answer this question: {q['question']}"
        
        # Generate
        import torch
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=4)
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        results.append({
            "id": q["id"],
            "question": q["question"],
            "answer": answer,
            "retrieved_docs": [
                {
                    "text": d["passage"]["text"],
                    "score": d["score"],
                    "title": d["passage"].get("title", "")
                }
                for d in retrieved
            ] if retrieved else [],
            "ground_truth": q["ground_truth"],
            "model": "flan-t5-base-rag"
        })
    
    # Save
    output_file = PROCESSED_DIR / "rag_predictions_real.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Generated {len(results)} RAG predictions")
    print(f"✓ Saved to: {output_file}")
    
    return results

def compare_predictions(baseline_results, rag_results):
    """Compare baseline vs RAG predictions."""
    print("\n" + "="*70)
    print("COMPARISON: BASELINE vs RAG")
    print("="*70 + "\n")
    
    comparisons = []
    
    for b, r in zip(baseline_results, rag_results):
        # Simple correctness check (if ground truth in answer)
        gt = b["ground_truth"].lower()
        baseline_correct = gt in b["answer"].lower() if gt else False
        rag_correct = gt in r["answer"].lower() if gt else False
        
        comparison = {
            "id": b["id"],
            "question": b["question"],
            "ground_truth": b["ground_truth"],
            "baseline_answer": b["answer"],
            "rag_answer": r["answer"],
            "baseline_correct": baseline_correct,
            "rag_correct": rag_correct,
            "retrieved_docs": r.get("retrieved_docs", [])
        }
        
        comparisons.append(comparison)
    
    # Calculate metrics
    baseline_accuracy = sum(1 for c in comparisons if c["baseline_correct"]) / len(comparisons)
    rag_accuracy = sum(1 for c in comparisons if c["rag_correct"]) / len(comparisons)
    
    print(f"Baseline Accuracy: {baseline_accuracy:.1%}")
    print(f"RAG Accuracy: {rag_accuracy:.1%}")
    print(f"Improvement: {(rag_accuracy - baseline_accuracy):.1%}")
    
    # Show examples
    print("\n" + "="*70)
    print("SAMPLE COMPARISONS")
    print("="*70)
    
    for i, c in enumerate(comparisons[:5], 1):
        print(f"\n{i}. Q: {c['question']}")
        print(f"   Ground Truth: {c['ground_truth']}")
        print(f"   Baseline: {c['baseline_answer']} {'✅' if c['baseline_correct'] else '❌'}")
        print(f"   RAG: {c['rag_answer']} {'✅' if c['rag_correct'] else '❌'}")
    
    # Save comparison
    output_file = PROCESSED_DIR / "complete_comparison.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for c in comparisons:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Saved comparison to: {output_file}")
    
    # Save metrics
    metrics = {
        "baseline_accuracy": f"{baseline_accuracy:.1%}",
        "rag_accuracy": f"{rag_accuracy:.1%}",
        "improvement": f"{(rag_accuracy - baseline_accuracy):.1%}",
        "total_samples": len(comparisons)
    }
    
    metrics_file = PROCESSED_DIR / "comparison_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved metrics to: {metrics_file}")
    
    return comparisons, metrics

def main():
    """Generate complete real comparison."""
    print("\n" + "="*70)
    print("COMPLETE REAL MODEL COMPARISON")
    print("="*70)
    print("\nThis will generate REAL predictions from FLAN-T5 for:")
    print("1. Baseline (model only)")
    print("2. RAG (model + retrieval)")
    print("3. Compare and evaluate")
    
    num = input("\nHow many questions? (default: 20): ").strip()
    limit = int(num) if num else 20
    
    print(f"\nGenerating predictions for {limit} questions...")
    print(f"Estimated time: {limit * 5 / 60:.1f} minutes\n")
    
    cont = input("Continue? (y/n): ").strip().lower()
    if cont != 'y':
        print("Cancelled")
        return
    
    # Load questions
    questions = load_questions(limit)
    if not questions:
        return
    
    # Generate baseline predictions
    baseline_results = generate_baseline_predictions(questions)
    
    # Generate RAG predictions
    rag_results = generate_rag_predictions(questions)
    
    # Compare
    comparisons, metrics = compare_predictions(baseline_results, rag_results)
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  • baseline_predictions_real.jsonl")
    print("  • rag_predictions_real.jsonl")
    print("  • complete_comparison.jsonl")
    print("  • comparison_metrics.json")
    print("\nNext steps:")
    print("  1. Refresh your Streamlit dashboard")
    print("  2. View real predictions in Browse Data tab")
    print("  3. See comparison metrics")

if __name__ == "__main__":
    main()
