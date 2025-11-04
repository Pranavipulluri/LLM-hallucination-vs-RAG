"""Evaluate ALL models with REAL measured data - no fake numbers!"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
LORA_DIR = Path("models/lora_weights")
RESULTS_DIR.mkdir(exist_ok=True)

def load_labeled_data():
    """Load labeled evaluation data."""
    labeled_file = PROCESSED_DIR / "labeled_evaluation_data.jsonl"
    
    if not labeled_file.exists():
        print("❌ No labeled data found!")
        print("Run: python scripts/generate_evaluation_results.py")
        return None
    
    data = []
    with open(labeled_file, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    
    return data

def evaluate_lora_if_exists(data):
    """Evaluate LoRA model ONLY if it actually exists."""
    if not LORA_DIR.exists() or not list(LORA_DIR.glob("*.bin")):
        print("\n⚠️  LoRA model not trained yet")
        print("To train: python scripts/train_lora_complete.py")
        return None
    
    print("\n✓ LoRA model found, evaluating...")
    
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from peft import PeftModel
        import torch
        
        # Load LoRA model
        base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        model = PeftModel.from_pretrained(base_model, str(LORA_DIR))
        tokenizer = AutoTokenizer.from_pretrained(str(LORA_DIR))
        
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        
        # Evaluate on labeled data
        correct = 0
        hallucinations = 0
        
        print("Evaluating LoRA on test set...")
        for item in data[:20]:  # Test on subset
            question = item['question']
            evidence = item.get('retrieved_docs', [{}])[0].get('text', '')
            
            prompt = f"""Answer using the evidence:
Evidence: {evidence}
Question: {question}
Answer:"""
            
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=128)
            
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check if correct (simple check)
            ground_truth = str(item.get('ground_truth', '')).lower()
            if ground_truth in answer.lower():
                correct += 1
            else:
                hallucinations += 1
        
        lora_accuracy = correct / 20
        lora_hallucination_rate = hallucinations / 20
        
        print(f"✓ LoRA Accuracy: {lora_accuracy:.1%}")
        print(f"✓ LoRA Hallucination Rate: {lora_hallucination_rate:.1%}")
        
        return {
            'accuracy': lora_accuracy,
            'hallucination_rate': lora_hallucination_rate
        }
        
    except Exception as e:
        print(f"❌ Error evaluating LoRA: {e}")
        return None

def plot_real_comparison(baseline_acc, rag_acc, lora_results=None):
    """Plot comparison with ONLY REAL measured data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    if lora_results:
        models = ['Baseline', 'RAG', 'RAG + LoRA']
        accuracies = [baseline_acc, rag_acc, lora_results['accuracy']]
        hall_rates = [1-baseline_acc, 1-rag_acc, lora_results['hallucination_rate']]
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        title_suffix = "(All Real Measured Data)"
    else:
        models = ['Baseline', 'RAG']
        accuracies = [baseline_acc, rag_acc]
        hall_rates = [1-baseline_acc, 1-rag_acc]
        colors = ['#ff6b6b', '#4ecdc4']
        title_suffix = "(Real Measured Data - LoRA Not Trained)"
    
    # Accuracy plot
    ax1.bar(models, accuracies, color=colors, alpha=0.8)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'Accuracy Comparison {title_suffix}', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
    
    # Hallucination rate plot
    ax2.bar(models, hall_rates, color=colors, alpha=0.8)
    ax2.set_ylabel('Hallucination Rate', fontsize=12)
    ax2.set_title(f'Hallucination Rate {title_suffix}', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 0.5])
    ax2.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(hall_rates):
        ax2.text(i, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    output_file = RESULTS_DIR / "real_comparison_all_models.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved real comparison: {output_file}")
    plt.close()

def main():
    """Evaluate all models with REAL data only."""
    print("\n" + "="*70)
    print("REAL EVALUATION - NO FAKE NUMBERS")
    print("="*70)
    
    # Load labeled data
    data = load_labeled_data()
    if not data:
        return
    
    # Calculate baseline and RAG metrics from labeled data
    baseline_correct = sum(1 for item in data if item.get('baseline_correct', False))
    rag_correct = sum(1 for item in data if item.get('rag_correct', False))
    
    baseline_accuracy = baseline_correct / len(data)
    rag_accuracy = rag_correct / len(data)
    
    print(f"\n📊 REAL MEASURED RESULTS:")
    print(f"  Baseline Accuracy: {baseline_accuracy:.1%}")
    print(f"  RAG Accuracy: {rag_accuracy:.1%}")
    print(f"  Improvement: {(rag_accuracy - baseline_accuracy):.1%}")
    
    # Try to evaluate LoRA if it exists
    lora_results = evaluate_lora_if_exists(data)
    
    # Plot with real data only
    plot_real_comparison(baseline_accuracy, rag_accuracy, lora_results)
    
    # Save results
    results = {
        "baseline_accuracy": f"{baseline_accuracy:.1%}",
        "rag_accuracy": f"{rag_accuracy:.1%}",
        "improvement": f"{(rag_accuracy - baseline_accuracy):.1%}",
        "lora_trained": lora_results is not None
    }
    
    if lora_results:
        results["lora_accuracy"] = f"{lora_results['accuracy']:.1%}"
        results["lora_hallucination_rate"] = f"{lora_results['hallucination_rate']:.1%}"
    
    output_file = RESULTS_DIR / "real_evaluation_all_models.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results: {output_file}")
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE - ALL NUMBERS ARE REAL!")
    print("="*70)
    
    if not lora_results:
        print("\n💡 To get LoRA results:")
        print("   1. python scripts/train_lora_complete.py")
        print("   2. python scripts/evaluate_all_models.py")

if __name__ == "__main__":
    main()
