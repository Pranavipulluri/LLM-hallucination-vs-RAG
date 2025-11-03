"""Fast download - uses smaller, pre-processed datasets."""
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_truthfulqa():
    """Download TruthfulQA dataset (small, ~800 questions)."""
    print("Downloading TruthfulQA...")
    dataset = load_dataset("truthful_qa", "generation", split="validation", trust_remote_code=True)
    
    questions = []
    for item in dataset:
        questions.append({
            "id": f"tqa_{len(questions)}",
            "question": item["question"],
            "best_answer": item["best_answer"],
            "correct_answers": item["correct_answers"],
            "incorrect_answers": item["incorrect_answers"]
        })
    
    with open(RAW_DIR / "truthfulqa.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    
    print(f"✓ Saved {len(questions)} TruthfulQA questions")
    return len(questions)

def download_nq_open(limit=1000):
    """Download NQ Open (much smaller than full NQ)."""
    print(f"Downloading NQ Open (first {limit} questions)...")
    
    # Use NQ Open - pre-processed, much smaller
    dataset = load_dataset("nq_open", split=f"train[:{limit}]")
    
    questions = []
    for item in tqdm(dataset, desc="Processing NQ"):
        questions.append({
            "id": f"nq_{len(questions)}",
            "question": item["question"],
            "answer": item["answer"]
        })
    
    with open(RAW_DIR / "nq_questions.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    
    print(f"✓ Saved {len(questions)} NQ questions")
    return len(questions)

def download_squad_v2(limit=1000):
    """Download SQuAD v2 as alternative to NQ."""
    print(f"Downloading SQuAD v2 (first {limit} questions)...")
    
    dataset = load_dataset("squad_v2", split=f"train[:{limit}]")
    
    questions = []
    for item in tqdm(dataset, desc="Processing SQuAD"):
        questions.append({
            "id": item["id"],
            "question": item["question"],
            "context": item["context"],
            "answers": item["answers"]
        })
    
    with open(RAW_DIR / "squad_questions.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    
    print(f"✓ Saved {len(questions)} SQuAD questions")
    return len(questions)

def download_fever_small(limit=1000):
    """Download small subset of FEVER."""
    print(f"Downloading FEVER (first {limit} claims)...")
    
    dataset = load_dataset("fever", "v1.0", split=f"train[:{limit}]", trust_remote_code=True)
    
    claims = []
    for item in tqdm(dataset, desc="Processing FEVER"):
        claims.append({
            "id": item["id"],
            "claim": item["claim"],
            "label": item["label"]
        })
    
    with open(RAW_DIR / "fever_claims.jsonl", "w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    
    print(f"✓ Saved {len(claims)} FEVER claims")
    return len(claims)

def main():
    print("=" * 60)
    print("FAST DATASET DOWNLOAD")
    print("=" * 60)
    print("\nThis will download smaller, pre-processed datasets.")
    print("Much faster than the full datasets!\n")
    
    total = 0
    
    # TruthfulQA - small and essential
    try:
        total += download_truthfulqa()
    except Exception as e:
        print(f"✗ Error downloading TruthfulQA: {e}")
    
    print()
    
    # Choose between NQ Open or SQuAD
    print("Choose QA dataset:")
    print("1. NQ Open (Natural Questions - smaller version)")
    print("2. SQuAD v2 (Stanford QA - faster download)")
    choice = input("Enter 1 or 2 (default: 2): ").strip() or "2"
    
    try:
        if choice == "1":
            total += download_nq_open(limit=1000)
        else:
            total += download_squad_v2(limit=1000)
    except Exception as e:
        print(f"✗ Error downloading QA dataset: {e}")
    
    print()
    
    # FEVER - optional
    download_fever = input("Download FEVER dataset? (y/n, default: n): ").strip().lower()
    if download_fever == 'y':
        try:
            total += download_fever_small(limit=1000)
        except Exception as e:
            print(f"✗ Error downloading FEVER: {e}")
    
    print("\n" + "=" * 60)
    print(f"✓ Download complete! Total samples: {total}")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python scripts/download_wikipedia.py")
    print("2. Run: python scripts/embed.py")

if __name__ == "__main__":
    main()
