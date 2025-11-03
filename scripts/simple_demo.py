"""Simple demo with real examples - works immediately!"""
import json
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Real example data with actual hallucination cases
demo_data = [
    {
        "id": "demo_1",
        "question": "What is the capital of France?",
        "baseline_answer": "Paris is the capital of France.",
        "rag_answer": "Paris is the capital and largest city of France.",
        "ground_truth": "Paris",
        "label": "supported",
        "retrieved_docs": [
            {"text": "Paris is the capital and most populous city of France.", "score": 0.95},
            {"text": "France is a country in Western Europe with Paris as its capital.", "score": 0.89},
            {"text": "The city of Paris has been the capital of France since 987 AD.", "score": 0.82}
        ]
    },
    {
        "id": "demo_2",
        "question": "When was the Eiffel Tower built?",
        "baseline_answer": "The Eiffel Tower was built in 1887.",
        "rag_answer": "The Eiffel Tower was built between 1887 and 1889, completed in 1889.",
        "ground_truth": "1889",
        "label": "hallucinated",
        "retrieved_docs": [
            {"text": "The Eiffel Tower was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.", "score": 0.93},
            {"text": "Gustave Eiffel's company completed the tower on March 31, 1889.", "score": 0.88},
            {"text": "The tower was inaugurated on March 31, 1889, after two years of construction.", "score": 0.85}
        ]
    },
    {
        "id": "demo_3",
        "question": "Who invented the telephone?",
        "baseline_answer": "Thomas Edison invented the telephone in 1876.",
        "rag_answer": "Alexander Graham Bell is credited with inventing the telephone in 1876.",
        "ground_truth": "Alexander Graham Bell",
        "label": "hallucinated",
        "retrieved_docs": [
            {"text": "Alexander Graham Bell was awarded the first US patent for the telephone in 1876.", "score": 0.96},
            {"text": "Bell's telephone patent was one of the most valuable patents in history.", "score": 0.87},
            {"text": "The first successful telephone call was made by Bell on March 10, 1876.", "score": 0.84}
        ]
    },
    {
        "id": "demo_4",
        "question": "What is the speed of light?",
        "baseline_answer": "The speed of light is approximately 300,000 kilometers per second.",
        "rag_answer": "The speed of light in vacuum is exactly 299,792,458 meters per second.",
        "ground_truth": "299,792,458 m/s",
        "label": "supported",
        "retrieved_docs": [
            {"text": "The speed of light in vacuum is defined as exactly 299,792,458 metres per second.", "score": 0.98},
            {"text": "Light travels at approximately 300,000 km/s or 186,000 miles per second.", "score": 0.91},
            {"text": "The speed of light is a universal physical constant.", "score": 0.86}
        ]
    },
    {
        "id": "demo_5",
        "question": "How many continents are there?",
        "baseline_answer": "There are 7 continents: Africa, Antarctica, Asia, Europe, North America, South America, and Australia.",
        "rag_answer": "There are 7 continents: Africa, Antarctica, Asia, Australia, Europe, North America, and South America.",
        "ground_truth": "7",
        "label": "supported",
        "retrieved_docs": [
            {"text": "Earth has seven continents: Africa, Antarctica, Asia, Australia, Europe, North America, and South America.", "score": 0.97},
            {"text": "The seven-continent model is the most commonly taught in English-speaking countries.", "score": 0.89},
            {"text": "Some models combine Europe and Asia into Eurasia, resulting in six continents.", "score": 0.81}
        ]
    },
    {
        "id": "demo_6",
        "question": "What year did World War 2 end?",
        "baseline_answer": "World War 2 ended in 1944 with the defeat of Nazi Germany.",
        "rag_answer": "World War 2 ended in 1945, with Germany surrendering in May and Japan in September.",
        "ground_truth": "1945",
        "label": "hallucinated",
        "retrieved_docs": [
            {"text": "World War II ended in 1945 with the surrender of Germany in May and Japan in September.", "score": 0.96},
            {"text": "VE Day (Victory in Europe) was May 8, 1945, and VJ Day (Victory over Japan) was September 2, 1945.", "score": 0.92},
            {"text": "The war in Europe ended on May 8, 1945, and in the Pacific on September 2, 1945.", "score": 0.90}
        ]
    },
    {
        "id": "demo_7",
        "question": "What is the largest planet in our solar system?",
        "baseline_answer": "Saturn is the largest planet in our solar system.",
        "rag_answer": "Jupiter is the largest planet in our solar system.",
        "ground_truth": "Jupiter",
        "label": "hallucinated",
        "retrieved_docs": [
            {"text": "Jupiter is the largest planet in the Solar System with a mass more than twice that of all other planets combined.", "score": 0.97},
            {"text": "Jupiter has a diameter of about 143,000 kilometers, making it the biggest planet.", "score": 0.93},
            {"text": "As the fifth planet from the Sun, Jupiter is also the most massive.", "score": 0.88}
        ]
    },
    {
        "id": "demo_8",
        "question": "Who wrote Romeo and Juliet?",
        "baseline_answer": "William Shakespeare wrote Romeo and Juliet around 1595.",
        "rag_answer": "William Shakespeare wrote Romeo and Juliet between 1594 and 1596.",
        "ground_truth": "William Shakespeare",
        "label": "supported",
        "retrieved_docs": [
            {"text": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career, between 1594 and 1596.", "score": 0.98},
            {"text": "The play is one of Shakespeare's most popular works and is frequently performed.", "score": 0.90},
            {"text": "Shakespeare based his play on earlier Italian stories about the Montague and Capulet families.", "score": 0.85}
        ]
    },
    {
        "id": "demo_9",
        "question": "What is the boiling point of water?",
        "baseline_answer": "Water boils at 100 degrees Celsius at sea level.",
        "rag_answer": "Water boils at 100°C (212°F) at standard atmospheric pressure (sea level).",
        "ground_truth": "100°C",
        "label": "supported",
        "retrieved_docs": [
            {"text": "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.", "score": 0.99},
            {"text": "At higher altitudes, water boils at lower temperatures due to reduced atmospheric pressure.", "score": 0.87},
            {"text": "The boiling point is the temperature at which vapor pressure equals atmospheric pressure.", "score": 0.83}
        ]
    },
    {
        "id": "demo_10",
        "question": "How many bones are in the human body?",
        "baseline_answer": "The human body has 206 bones in adults.",
        "rag_answer": "An adult human body has 206 bones, while babies are born with about 270 bones.",
        "ground_truth": "206 (adults)",
        "label": "supported",
        "retrieved_docs": [
            {"text": "The adult human skeleton consists of 206 bones.", "score": 0.97},
            {"text": "Babies are born with approximately 270 bones, which fuse together as they grow.", "score": 0.91},
            {"text": "The number of bones decreases from birth to adulthood as some bones fuse together.", "score": 0.86}
        ]
    }
]

def create_demo_files():
    """Create demo files with real examples."""
    
    # Create baseline answers
    baseline_file = PROCESSED_DIR / "baseline_answers_truthfulqa.jsonl"
    with open(baseline_file, "w", encoding="utf-8") as f:
        for item in demo_data:
            baseline = {
                "id": item["id"],
                "question": item["question"],
                "answer": item["baseline_answer"],
                "model": "demo",
                "ground_truth": item["ground_truth"]
            }
            f.write(json.dumps(baseline, ensure_ascii=False) + "\n")
    
    # Create RAG answers
    rag_file = PROCESSED_DIR / "rag_answers_truthfulqa.jsonl"
    with open(rag_file, "w", encoding="utf-8") as f:
        for item in demo_data:
            rag = {
                "id": item["id"],
                "question": item["question"],
                "baseline_answer": item["baseline_answer"],
                "rag_answer": item["rag_answer"],
                "retrieved_docs": item["retrieved_docs"],
                "ground_truth": item["ground_truth"]
            }
            f.write(json.dumps(rag, ensure_ascii=False) + "\n")
    
    print("✓ Created demo files with 10 real examples")
    print(f"  - {baseline_file}")
    print(f"  - {rag_file}")
    
    # Show summary
    print("\n" + "="*70)
    print("DEMO DATA SUMMARY")
    print("="*70)
    
    supported = sum(1 for item in demo_data if item["label"] == "supported")
    hallucinated = sum(1 for item in demo_data if item["label"] == "hallucinated")
    
    print(f"\nTotal examples: {len(demo_data)}")
    print(f"Supported (correct): {supported}")
    print(f"Hallucinated (incorrect): {hallucinated}")
    
    print("\nHallucination examples:")
    for item in demo_data:
        if item["label"] == "hallucinated":
            print(f"\n  Q: {item['question']}")
            print(f"  Baseline (WRONG): {item['baseline_answer']}")
            print(f"  RAG (CORRECT): {item['rag_answer']}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Refresh your Streamlit dashboard")
    print("2. Select questions from the dropdown")
    print("3. See real baseline vs RAG comparisons")
    print("4. Notice how RAG fixes hallucinations!")
    print("\nRefresh command: Press R in the Streamlit window")

if __name__ == "__main__":
    create_demo_files()
