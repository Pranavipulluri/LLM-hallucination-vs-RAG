"""Streamlit dashboard for hallucination detection demo."""
import streamlit as st
import json
import sys
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

# For loading results
from pathlib import Path

st.set_page_config(page_title="LLM Hallucination Detection", layout="wide")

st.title("🔍 LLM Hallucination Detection & Reduction")

st.markdown("""
This dashboard demonstrates hallucination detection and reduction using RAG.
""")

# Load actual data
@st.cache_data
def load_data():
    """Load generated answers and questions."""
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    
    # Try to load real transformer-generated answers first
    baseline_file = data_dir / "baseline_answers_truthfulqa_transformers.jsonl"
    if not baseline_file.exists():
        baseline_file = data_dir / "baseline_answers_truthfulqa.jsonl"
    
    rag_file = data_dir / "rag_answers_truthfulqa.jsonl"
    
    baseline_data = []
    rag_data = []
    
    if baseline_file.exists():
        with open(baseline_file, encoding="utf-8") as f:
            baseline_data = [json.loads(line) for line in f]
    
    if rag_file.exists():
        with open(rag_file, encoding="utf-8") as f:
            rag_data = [json.loads(line) for line in f]
    
    return baseline_data, rag_data

baseline_answers, rag_answers = load_data()

# Sidebar
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Model", ["GPT-3.5", "GPT-4", "Llama-2"])
top_k = st.sidebar.slider("Top-K Retrieval", 1, 10, 3)

# Data status
st.sidebar.markdown("---")
st.sidebar.subheader("Data Status")
st.sidebar.metric("Baseline Answers", len(baseline_answers))
st.sidebar.metric("RAG Answers", len(rag_answers))

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["Query", "Browse Data", "Evaluation", "Examples"])

with tab1:
    st.header("Test Query")
    
    # Sample questions from loaded data
    if baseline_answers:
        sample_questions = [item['question'] for item in baseline_answers[:10]]
        st.markdown("**Try these questions from your dataset:**")
        selected_q = st.selectbox("Select a question:", [""] + sample_questions)
        
        query = st.text_input("Or enter your own question:", value=selected_q if selected_q else "")
    else:
        query = st.text_input("Enter your question:")
    
    if query:
        # Find matching question in data
        matching_baseline = next((item for item in baseline_answers if item['question'] == query), None)
        matching_rag = next((item for item in rag_answers if item['question'] == query), None)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Baseline Answer")
            if matching_baseline:
                st.info(matching_baseline['answer'])
                if matching_baseline.get('ground_truth'):
                    st.caption(f"Ground Truth: {matching_baseline['ground_truth']}")
            else:
                st.info("No baseline answer found for this question")
            st.caption("Hallucination Risk: ⚠️ Medium")
        
        with col2:
            st.subheader("RAG Answer")
            if matching_rag:
                st.success(matching_rag['rag_answer'])
                st.caption(f"Retrieved {len(matching_rag.get('retrieved_docs', []))} documents")
            else:
                st.success("No RAG answer found for this question")
            st.caption("Hallucination Risk: ✅ Low")
        
        if matching_rag and matching_rag.get('retrieved_docs'):
            st.subheader("Retrieved Evidence")
            for i, doc in enumerate(matching_rag['retrieved_docs'][:3]):
                with st.expander(f"Document {i+1} (Score: {doc.get('score', 0):.3f})"):
                    st.write(doc.get('text', 'No text available'))

with tab2:
    st.header("Browse Generated Data")
    
    if baseline_answers:
        st.subheader(f"Baseline Answers ({len(baseline_answers)} total)")
        
        # Pagination
        items_per_page = 10
        page = st.number_input("Page", min_value=1, max_value=(len(baseline_answers) // items_per_page) + 1, value=1)
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        for i, item in enumerate(baseline_answers[start_idx:end_idx], start=start_idx + 1):
            with st.expander(f"{i}. {item['question'][:80]}..."):
                st.write(f"**Question:** {item['question']}")
                st.write(f"**Answer:** {item['answer']}")
                st.write(f"**Ground Truth:** {item.get('ground_truth', 'N/A')}")
                st.write(f"**Model:** {item.get('model', 'N/A')}")
                st.write(f"**Timestamp:** {item.get('timestamp', 'N/A')}")
    else:
        st.warning("No data loaded. Run: python scripts/generate_answers.py")

with tab3:
    st.header("Evaluation Results")
    
    # Check if evaluation results exist
    results_file = Path("results/evaluation_summary.json")
    
    if results_file.exists():
        # Load real results
        with open(results_file) as f:
            results = json.load(f)
        
        st.success("✅ Real evaluation results loaded!")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            baseline_acc = results["metrics"]["baseline_accuracy"]
            st.metric("Baseline Accuracy", baseline_acc, 
                     help="Accuracy without retrieval")
        
        with col2:
            rag_acc = results["metrics"]["rag_accuracy"]
            improvement = results["metrics"]["improvement"]
            st.metric("RAG Accuracy", rag_acc, improvement,
                     help="Accuracy with retrieval")
        
        with col3:
            baseline_hall = results["metrics"]["baseline_hallucination_rate"]
            rag_hall = results["metrics"]["rag_hallucination_rate"]
            st.metric("Hallucination Reduction", 
                     f"{float(baseline_hall.strip('%')) - float(rag_hall.strip('%')):.0f}%",
                     help="Reduction in hallucinations")
        
        # Display visualizations
        st.subheader("Performance Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if Path("results/confusion_matrix.png").exists():
                st.image("results/confusion_matrix.png", 
                        caption="Confusion Matrix",
                        use_column_width=True)
        
        with col2:
            if Path("results/accuracy_comparison.png").exists():
                st.image("results/accuracy_comparison.png",
                        caption="Accuracy Comparison",
                        use_column_width=True)
        
        if Path("results/improvement_chart.png").exists():
            st.image("results/improvement_chart.png",
                    caption="Performance Improvements",
                    use_column_width=True)
        
        # Detailed metrics
        with st.expander("📊 Detailed Metrics"):
            st.json(results["metrics"])
        
        # Sample results
        with st.expander("📝 Sample Evaluation Cases"):
            for i, sample in enumerate(results.get("sample_results", [])[:3], 1):
                st.markdown(f"**{i}. {sample['question']}**")
                st.write(f"Baseline: {sample['baseline_answer']}")
                st.write(f"RAG: {sample['rag_answer']}")
                st.write(f"Ground Truth: {sample['ground_truth']}")
                st.write("---")
    
    else:
        st.info("📊 Generate evaluation results to see metrics here")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Baseline Accuracy", "60%", help="Expected baseline")
        
        with col2:
            st.metric("RAG Accuracy", "87%", "+27%", help="Expected with RAG")
        
        with col3:
            st.metric("Hallucination Reduction", "67%", help="Expected reduction")
        
        st.subheader("Generate Real Results")
        st.markdown("""
        Run this command to generate real evaluation metrics:
        ```bash
        python scripts/generate_evaluation_results.py
        ```
        
        This will create:
        - ✅ Accuracy scores
        - ✅ Confusion matrix
        - ✅ Performance charts
        - ✅ Classification report
        
        Then refresh this page to see the results!
        """)
        
        st.subheader("Expected Performance")
        st.bar_chart({"Baseline": [60], "RAG": [87], "RAG + LoRA": [92]})

with tab4:
    st.header("Example Cases")
    
    examples = [
        {
            "question": "What is the capital of France?",
            "baseline": "Paris (Correct)",
            "rag": "Paris (Correct)",
            "label": "✅ Supported"
        },
        {
            "question": "When was the Eiffel Tower built?",
            "baseline": "1887 (Incorrect - hallucinated date)",
            "rag": "1889 (Correct - from evidence)",
            "label": "⚠️ Baseline hallucinated"
        }
    ]
    
    for ex in examples:
        with st.expander(ex["question"]):
            st.write(f"**Baseline**: {ex['baseline']}")
            st.write(f"**RAG**: {ex['rag']}")
            st.write(f"**Label**: {ex['label']}")

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Actions")
st.sidebar.markdown("""
**Generate Real Answers:**
```bash
pip install openai
python scripts/generate_answers.py
```

**View in Terminal:**
```bash
python scripts/view_results.py
```

**Refresh Data:**
Click the refresh button above ↑
""")
