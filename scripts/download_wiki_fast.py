"""Download pre-processed Wikipedia passages - FAST and WORKS!"""
from datasets import load_dataset
from pathlib import Path
import json
from tqdm import tqdm

PROCESSED_DIR = Path("data/processed")

def download_wiki_passages(num_passages=10000):
    """Download pre-processed Wikipedia passages."""
    print("\n" + "="*70)
    print("DOWNLOADING WIKIPEDIA PASSAGES")
    print("="*70)
    print(f"\nDownloading {num_passages} pre-processed passages...")
    print("These are real Wikipedia sentences, ready for retrieval.\n")
    
    # Use sentence-transformers Wikipedia dataset (pre-processed!)
    dataset = load_dataset(
        "sentence-transformers/wikipedia-en-sentences",
        split=f"train[:{num_passages}]"
    )
    
    passages = []
    
    for i, item in enumerate(tqdm(dataset, desc="Processing passages")):
        # Handle different possible keys
        text = item.get('sentence', item.get('text', item.get('passage', '')))
        
        if text and len(text) > 50:  # Only keep substantial passages
            passages.append({
                "id": f"wiki_{i}",
                "title": item.get('title', item.get('section', 'Wikipedia')),
                "text": text
            })
    
    # Save
    output_file = PROCESSED_DIR / "passages.jsonl"
    
    # Backup old file
    if output_file.exists():
        import shutil
        shutil.copy(output_file, PROCESSED_DIR / "passages_backup.jsonl")
        print("✓ Backed up old passages")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Saved {len(passages)} real Wikipedia passages")
    print(f"✓ File: {output_file}")
    
    # Show samples
    print("\nSample passages:")
    for i, p in enumerate(passages[:3], 1):
        print(f"\n{i}. {p['text'][:100]}...")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Rebuild embeddings: python scripts/embed.py")
    print("2. Try interactive Q&A: python scripts/interactive_qa.py")
    print("\nNow you have REAL Wikipedia for RAG!")

if __name__ == "__main__":
    print("\nThis will download REAL Wikipedia passages.")
    print("Pre-processed and ready to use!\n")
    
    num = input("How many passages? (default: 10000, ~3 min): ").strip()
    num_passages = int(num) if num else 10000
    
    cont = input(f"\nDownload {num_passages} passages? (y/n): ").strip().lower()
    if cont == 'y':
        download_wiki_passages(num_passages)
    else:
        print("Cancelled")
