"""Generate REAL LLM answers using local model - guaranteed to work!"""
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Global model cache
_model = None
_tokenizer = None

def load_model():
    """Load FLAN-T5 model (small, fast, works on CPU)."""
    global _model, _tokenizer
    
    if _model is not None:
        return _model, _tokenizer
    
    print("\n📦 Loading FLAN-T5 model (first time only, ~1GB download)...")
    print("This will be cached for future runs.\n")
    
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    
    model_name = "google/flan-t5-base"  # 250M params, good quality
    
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        _model = _model.cuda()
        print("✓ Using GPU")
    else:
        print("✓ Using CPU (slower but works)")
    
    return _model, _tokenizer

def generate_answer(question):
    """Generate a real answer using FLAN-T5."""
    model, tokenizer = load_model()
    
    import torch
    
    # Format prompt
    prompt = f"Answer this question concisely and accurately: {question}"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    # Move to same device as model
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,  # Better quality
            early_stopping=True,
            temperature=0.7
        )
    
    # Decode
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

def load_questions(file_path):
    """Load questions from JSONL file."""
    questions = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    return questions

def generate_baseline_answers(questions_file, output_file, limit=50):
    """Generate REAL baseline answers."""
    questions = load_questions(questions_file)
    
    if limit:
        questions = questions[:limit]
    
    print(f"\n{'='*70}")
    print(f"Generating {len(questions)} REAL answers using FLAN-T5")
    print(f"{'='*70}\n")
    
    results = []
    
    for i, q in enumerate(tqdm(questions, desc="Generating answers")):
        question_text = q.get("question", q.get("claim", ""))
        
        try:
            answer = generate_answer(question_text)
        except Exception as e:
            print(f"\nError on question {i+1}: {e}")
            answer = "[Error generating answer]"
        
        result = {
            "id": q["id"],
            "question": question_text,
            "model": "flan-t5-base",
            "temperature": 0.7,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "ground_truth": q.get("best_answer", q.get("answer", q.get("label")))
        }
        results.append(result)
    
    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*70}")
    print(f"✓ Generated {len(results)} REAL answers")
    print(f"✓ Saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Show sample
    print("Sample answers:")
    for i, r in enumerate(results[:3], 1):
        print(f"\n{i}. Q: {r['question'][:60]}...")
        print(f"   A: {r['answer']}")
        if r.get('ground_truth'):
            print(f"   Truth: {r['ground_truth']}")

def main():
    """Generate real answers for all datasets."""
    print("\n" + "="*70)
    print("REAL ANSWER GENERATION - FLAN-T5 (Local, Free)")
    print("="*70)
    print("\nThis will generate REAL LLM answers using FLAN-T5.")
    print("- Model: google/flan-t5-base (250M parameters)")
    print("- Quality: Good for factual QA")
    print("- Speed: ~2-3 seconds per answer on CPU")
    print("- Cost: FREE (runs locally)")
    print("- First run: Downloads ~1GB model (cached after)")
    print()
    
    num_questions = input("How many questions to generate? (default: 50): ").strip()
    limit = int(num_questions) if num_questions else 50
    
    print(f"\nGenerating {limit} answers per dataset...")
    print("Estimated time: {:.1f} minutes\n".format(limit * 2.5 / 60))
    
    cont = input("Continue? (y/n): ").strip().lower()
    if cont != 'y':
        print("Cancelled.")
        return
    
    # Install transformers if needed
    try:
        import transformers
        import torch
    except ImportError:
        print("\n📦 Installing required packages...")
        os.system("pip install transformers torch")
    
    # Generate for TruthfulQA
    truthfulqa_file = RAW_DIR / "truthfulqa.jsonl"
    if truthfulqa_file.exists():
        print("\n" + "="*70)
        print("TRUTHFULQA DATASET")
        print("="*70)
        generate_baseline_answers(
            truthfulqa_file,
            PROCESSED_DIR / "baseline_answers_truthfulqa.jsonl",
            limit=limit
        )
    
    # Generate for SQuAD
    squad_file = RAW_DIR / "squad_questions.jsonl"
    if squad_file.exists():
        print("\n" + "="*70)
        print("SQUAD DATASET")
        print("="*70)
        generate_baseline_answers(
            squad_file,
            PROCESSED_DIR / "baseline_answers_squad.jsonl",
            limit=limit
        )
    
    print("\n" + "="*70)
    print("✓ ALL DONE!")
    print("="*70)
    print("\nNext steps:")
    print("1. View results: python scripts/view_results.py")
    print("2. Generate RAG answers: python scripts/rag_pipeline_real.py")
    print("3. Refresh Streamlit dashboard")
    print("\nYour baseline answers are ready for comparison!")

if __name__ == "__main__":
    main()
