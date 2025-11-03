"""Create embeddings and FAISS index for retrieval."""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_passages(passages_file):
    """Load passages from JSONL file."""
    passages = []
    with open(passages_file) as f:
        for line in f:
            data = json.loads(line)
            passages.append(data['text'])
    return passages

def create_embeddings(passages, model_name='all-mpnet-base-v2', batch_size=64):
    """Create embeddings for passages."""
    print(f"Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)
    
    print(f"Encoding {len(passages)} passages...")
    embeddings = embedder.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return embeddings, embedder

def build_faiss_index(embeddings, index_path):
    """Build and save FAISS index."""
    embeddings = np.array(embeddings).astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner product after normalization = cosine
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")
    
    return index

def main():
    # For demo, create sample Wikipedia passages
    # In production, use actual Wikipedia dump
    passages_file = PROCESSED_DIR / "passages.jsonl"
    
    if not passages_file.exists():
        print("Creating sample passages (replace with actual Wikipedia data)...")
        sample_passages = [
            {"id": f"p{i}", "text": f"Sample passage {i} about various topics."}
            for i in range(100)
        ]
        with open(passages_file, "w") as f:
            for p in sample_passages:
                f.write(json.dumps(p) + "\n")
    
    passages = load_passages(passages_file)
    embeddings, embedder = create_embeddings(passages)
    
    index_path = PROCESSED_DIR / "wiki.index"
    build_faiss_index(embeddings, index_path)
    
    print(f"\nIndex created with {len(passages)} passages")
    print(f"Embedding dimension: {embeddings.shape[1]}")

if __name__ == "__main__":
    main()
