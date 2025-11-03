"""Complete LoRA fine-tuning implementation for hallucination reduction."""
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
LORA_DIR = MODELS_DIR / "lora_weights"
LORA_DIR.mkdir(parents=True, exist_ok=True)

def create_training_data():
    """Create training data for LoRA fine-tuning."""
    print("\n" + "="*70)
    print("CREATING LORA TRAINING DATA")
    print("="*70 + "\n")
    
    training_examples = [
        # Factual QA - Correct answers
        {
            "instruction": "Answer this question accurately using the provided evidence. If the evidence doesn't contain the answer, say 'I don't have enough information.'",
            "input": "When was the Eiffel Tower built?",
            "evidence": "The Eiffel Tower was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.",
            "output": "The Eiffel Tower was built between 1887 and 1889, completed in 1889."
        },
        {
            "instruction": "Answer this question accurately using the provided evidence. If the evidence doesn't contain the answer, say 'I don't have enough information.'",
            "input": "Who invented the telephone?",
            "evidence": "Alexander Graham Bell was awarded the first US patent for the telephone in 1876.",
            "output": "Alexander Graham Bell invented the telephone in 1876."
        },
        {
            "instruction": "Answer this question accurately using the provided evidence. If the evidence doesn't contain the answer, say 'I don't have enough information.'",
            "input": "What is the capital of France?",
            "evidence": "Paris is the capital and most populous city of France.",
            "output": "Paris is the capital of France."
        },
        {
            "instruction": "Answer this question accurately using the provided evidence. If the evidence doesn't contain the answer, say 'I don't have enough information.'",
            "input": "What is the speed of light?",
            "evidence": "The speed of light in vacuum is defined as exactly 299,792,458 metres per second.",
            "output": "The speed of light is 299,792,458 meters per second."
        },
        {
            "instruction": "Answer this question accurately using the provided evidence. If the evidence doesn't contain the answer, say 'I don't have enough information.'",
            "input": "How many continents are there?",
            "evidence": "Earth has seven continents: Africa, Antarctica, Asia, Australia, Europe, North America, and South America.",
            "output": "There are seven continents."
        },
        # Abstention examples - Learning to say "I don't know"
        {
            "instruction": "Answer this question accurately using the provided evidence. If the evidence doesn't contain the answer, say 'I don't have enough information.'",
            "input": "What is the population of Mars?",
            "evidence": "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System.",
            "output": "I don't have enough information about Mars' population in the provided evidence."
        },
        {
            "instruction": "Answer this question accurately using the provided evidence. If the evidence doesn't contain the answer, say 'I don't have enough information.'",
            "input": "Who will win the next election?",
            "evidence": "Elections are democratic processes where citizens vote for their representatives.",
            "output": "I don't have enough information to predict future election results."
        },
        # Evidence-based reasoning
        {
            "instruction": "Answer this question accurately using the provided evidence. If the evidence doesn't contain the answer, say 'I don't have enough information.'",
            "input": "When did World War 2 end?",
            "evidence": "World War II ended in 1945 with the surrender of Germany in May and Japan in September.",
            "output": "World War 2 ended in 1945."
        },
        {
            "instruction": "Answer this question accurately using the provided evidence. If the evidence doesn't contain the answer, say 'I don't have enough information.'",
            "input": "What is the largest planet?",
            "evidence": "Jupiter is the largest planet in the Solar System with a mass more than twice that of all other planets combined.",
            "output": "Jupiter is the largest planet in the Solar System."
        },
        {
            "instruction": "Answer this question accurately using the provided evidence. If the evidence doesn't contain the answer, say 'I don't have enough information.'",
            "input": "Who wrote Romeo and Juliet?",
            "evidence": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career, between 1594 and 1596.",
            "output": "William Shakespeare wrote Romeo and Juliet."
        }
    ]
    
    # Expand with variations
    expanded_data = []
    for example in training_examples:
        # Original
        expanded_data.append(example)
        
        # Add variations with different phrasings
        if "don't have enough information" not in example["output"]:
            # Variation 1: More detailed
            expanded_data.append({
                **example,
                "output": example["output"] + " According to the evidence provided."
            })
            
            # Variation 2: Cite evidence
            expanded_data.append({
                **example,
                "output": example["output"] + " This is based on the provided evidence."
            })
    
    print(f"Created {len(expanded_data)} training examples")
    
    # Save training data
    output_file = PROCESSED_DIR / "lora_training_data.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for example in expanded_data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"✓ Saved to: {output_file}\n")
    
    return expanded_data

def prepare_dataset(examples, tokenizer, max_length=512):
    """Prepare dataset for training."""
    
    def format_example(example):
        """Format example as instruction-following prompt."""
        prompt = f"""{example['instruction']}

Evidence: {example['evidence']}

Question: {example['input']}

Answer:"""
        return prompt, example['output']
    
    # Format all examples
    inputs = []
    outputs = []
    
    for example in examples:
        prompt, answer = format_example(example)
        inputs.append(prompt)
        outputs.append(answer)
    
    # Tokenize
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=False
    )
    
    labels = tokenizer(
        outputs,
        max_length=128,
        truncation=True,
        padding=False
    )
    
    # Create dataset
    dataset_dict = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels["input_ids"]
    }
    
    return Dataset.from_dict(dataset_dict)

def train_lora(model_name="google/flan-t5-base"):
    """Train LoRA adapter for hallucination reduction."""
    
    print("\n" + "="*70)
    print("LORA FINE-TUNING")
    print("="*70 + "\n")
    
    # Create training data
    training_examples = create_training_data()
    
    # Split train/val (80/20)
    split_idx = int(len(training_examples) * 0.8)
    train_examples = training_examples[:split_idx]
    val_examples = training_examples[split_idx:]
    
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}\n")
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,                          # Rank
        lora_alpha=32,                # Scaling
        target_modules=["q", "v"],    # Attention layers
        lora_dropout=0.05,            # Regularization
        bias="none"
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(train_examples, tokenizer)
    val_dataset = prepare_dataset(val_examples, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(LORA_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70 + "\n")
    
    trainer.train()
    
    # Save
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70 + "\n")
    
    trainer.save_model(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    
    print(f"✓ LoRA adapter saved to: {LORA_DIR}")
    print(f"✓ Adapter size: ~10MB (vs 1GB for full model)")
    
    # Test
    print("\n" + "="*70)
    print("TESTING")
    print("="*70 + "\n")
    
    test_questions = [
        ("When was the Eiffel Tower built?", "The Eiffel Tower was constructed from 1887 to 1889."),
        ("Who invented the telephone?", "Alexander Graham Bell invented the telephone."),
        ("What is the population of Mars?", "Mars has no permanent human population.")
    ]
    
    model.eval()
    for question, evidence in test_questions:
        prompt = f"""Answer this question accurately using the provided evidence.

Evidence: {evidence}

Question: {question}

Answer:"""
        
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Q: {question}")
        print(f"A: {answer}\n")
    
    print("="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Evaluate: python scripts/evaluate_lora.py")
    print("2. Use in RAG: python scripts/rag_with_lora.py")
    print("3. Compare in dashboard: streamlit run app/streamlit_app.py")

def main():
    """Main training function."""
    print("\n" + "="*70)
    print("LORA FINE-TUNING FOR HALLUCINATION REDUCTION")
    print("="*70)
    print("\nThis will:")
    print("1. Create training data (30 examples)")
    print("2. Configure LoRA (r=8, alpha=32)")
    print("3. Train adapter (3 epochs, ~30-60 min)")
    print("4. Save adapter (~10MB)")
    print("5. Test on sample questions")
    
    if torch.cuda.is_available():
        print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
        print("  Training will be faster!")
    else:
        print("\n⚠️  No GPU detected, using CPU")
        print("  Training will take longer (~60 min)")
    
    cont = input("\nContinue? (y/n): ").strip().lower()
    if cont != 'y':
        print("Cancelled")
        return
    
    train_lora()

if __name__ == "__main__":
    main()
