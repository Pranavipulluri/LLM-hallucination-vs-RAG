"""Retrieval utilities for RAG pipeline."""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

class Retriever:
    def __init__(self, index_path, passages_path, model_name='all-mpnet-base-v2'):
        """Initialize retriever with FAISS index and passages."""
        self.embedder = SentenceTransformer(model_name)
        self.index = faiss.read_index(str(index_path))
        
        # Load passages
        self.passages = []
        with open(passages_path) as f:
            for line in f:
                data = json.loads(line)
                self.passages.append(data)
    
    def retrieve(self, query, k=5):
        """Retrieve top-k passages for a query."""
        # Encode query
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        q_emb = np.array(q_emb).astype('float32')
        faiss.normalize_L2(q_emb)
        
        # Search
        scores, indices = self.index.search(q_emb, k)
        
        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "passage": self.passages[idx],
                "score": float(score)
            })
        
        return results

def main():
    """Test retrieval."""
    retriever = Retriever(
        index_path="data/processed/wiki.index",
        passages_path="data/processed/passages.jsonl"
    )
    
    test_query = "What is the capital of France?"
    results = retriever.retrieve(test_query, k=3)
    
    print(f"Query: {test_query}\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.4f}")
        print(f"   {result['passage']['text']}\n")

if __name__ == "__main__":
    main()
