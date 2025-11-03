"""Interactive Q&A - Ask your own questions and get real LLM answers!"""
import json
from pathlib import Path
from datetime import datetime
import sys

# Import model loading
sys.path.append(str(Path(__file__).parent))
from generate_real_answers import load_model, generate_answer
from retrieve import Retriever

PROCESSED_DIR = Path("data/processed")

def generate_rag_answer_interactive(question, retriever):
    """Generate RAG answer with evidence."""
    model, tokenizer = load_model()
    import torch
    
    # Retrieve documents
    retrieved = retriever.retrieve(question, k=3)
    
    # Format evidence
    evidence = ""
    for i, doc in enumerate(retrieved, 1):
        evidence += f"[{i}] {doc['passage']['text']}\n\n"
    
    # Create RAG prompt
    prompt = f"""Answer the question using ONLY the information from these documents. If the documents don't contain the answer, say "I don't have enough information."

Documents:
{evidence}

Question: {question}

Answer:"""
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip(), retrieved

def interactive_mode():
    """Interactive Q&A mode."""
    print("\n" + "="*70)
    print("INTERACTIVE Q&A - Real LLM Answers")
    print("="*70)
    print("\nThis uses FLAN-T5 model to answer your questions.")
    print("You'll get both baseline and RAG answers with evidence.")
    print("\nType 'quit' to exit\n")
    
    # Load model
    print("📦 Loading FLAN-T5 model...")
    load_model()
    print("✓ Model loaded\n")
    
    # Load retriever
    print("📚 Loading retrieval system...")
    retriever = Retriever(
        index_path=PROCESSED_DIR / "wiki.index",
        passages_path=PROCESSED_DIR / "passages.jsonl"
    )
    print("✓ Retriever loaded\n")
    
    print("="*70)
    print("Ready! Ask your questions:")
    print("="*70 + "\n")
    
    results = []
    
    while True:
        # Get question
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        print("\n⏳ Generating answers...\n")
        
        try:
            # Generate baseline answer
            print("1️⃣ Baseline (no retrieval):")
            baseline_answer = generate_answer(question)
            print(f"   {baseline_answer}\n")
            
            # Generate RAG answer
            print("2️⃣ RAG (with retrieval):")
            rag_answer, retrieved = generate_rag_answer_interactive(question, retriever)
            print(f"   {rag_answer}\n")
            
            # Show evidence
            print("📄 Retrieved Evidence:")
            for i, doc in enumerate(retrieved, 1):
                print(f"   [{i}] (Score: {doc['score']:.3f})")
                print(f"       {doc['passage']['text'][:100]}...\n")
            
            # Save result
            result = {
                "question": question,
                "baseline_answer": baseline_answer,
                "rag_answer": rag_answer,
                "retrieved_docs": [
                    {
                        "text": d["passage"]["text"],
                        "score": d["score"],
                        "title": d["passage"].get("title", "")
                    }
                    for d in retrieved
                ],
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error: {e}\n")
        
        print("="*70 + "\n")
    
    # Save results
    if results:
        output_file = PROCESSED_DIR / "interactive_qa_results.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        print(f"\n✓ Saved {len(results)} Q&A pairs to: {output_file}")
        print("\nYou can add these to your dashboard by updating the data files!")

def batch_mode():
    """Process multiple questions from a file."""
    print("\n" + "="*70)
    print("BATCH MODE - Process Multiple Questions")
    print("="*70)
    
    questions_file = input("\nEnter path to questions file (one per line): ").strip()
    
    if not Path(questions_file).exists():
        print("❌ File not found!")
        return
    
    # Read questions
    with open(questions_file, encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    
    print(f"\n✓ Loaded {len(questions)} questions")
    print("⏳ Processing...\n")
    
    # Load model and retriever
    load_model()
    retriever = Retriever(
        index_path=PROCESSED_DIR / "wiki.index",
        passages_path=PROCESSED_DIR / "passages.jsonl"
    )
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question}")
        
        try:
            baseline_answer = generate_answer(question)
            rag_answer, retrieved = generate_rag_answer_interactive(question, retriever)
            
            result = {
                "id": f"custom_{i}",
                "question": question,
                "baseline_answer": baseline_answer,
                "rag_answer": rag_answer,
                "retrieved_docs": [
                    {
                        "text": d["passage"]["text"],
                        "score": d["score"]
                    }
                    for d in retrieved
                ],
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Save
    output_file = PROCESSED_DIR / "custom_qa_results.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Saved {len(results)} results to: {output_file}")

def main():
    """Main menu."""
    print("\n" + "="*70)
    print("CUSTOM QUESTION ANSWERING")
    print("="*70)
    print("\nChoose mode:")
    print("1. Interactive - Ask questions one by one")
    print("2. Batch - Process questions from a file")
    print("3. Quick test - Try a few example questions")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        interactive_mode()
    elif choice == "2":
        batch_mode()
    elif choice == "3":
        # Quick test
        test_questions = [
            "What is the capital of Japan?",
            "Who wrote Harry Potter?",
            "What is 2+2?"
        ]
        
        print("\n" + "="*70)
        print("QUICK TEST")
        print("="*70)
        
        load_model()
        retriever = Retriever(
            index_path=PROCESSED_DIR / "wiki.index",
            passages_path=PROCESSED_DIR / "passages.jsonl"
        )
        
        for q in test_questions:
            print(f"\nQ: {q}")
            baseline = generate_answer(q)
            rag, retrieved = generate_rag_answer_interactive(q, retriever)
            print(f"Baseline: {baseline}")
            print(f"RAG: {rag}")
            print(f"Evidence: {retrieved[0]['passage']['text'][:80]}...")
            print("-"*70)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
