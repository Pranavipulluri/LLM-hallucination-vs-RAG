# Annotation Guidelines for Hallucination Detection

## Overview
This document provides guidelines for annotating LLM outputs to identify hallucinations.

## Label Schema

### 1. Supported (Label: 0)
The answer is grounded in the provided evidence and can be verified.

**Criteria:**
- All facts in the answer appear in the evidence
- No fabricated information
- Logical inferences are reasonable and supported

**Example:**
- Question: "What is the capital of France?"
- Answer: "Paris"
- Evidence: "Paris is the capital and largest city of France."
- Label: **Supported**

### 2. Not Supported / Hallucinated (Label: 1)
The answer contains information that cannot be verified from the evidence.

**Criteria:**
- Contains facts not present in evidence
- Fabricated details (dates, names, numbers)
- Unverifiable claims

**Example:**
- Question: "When was the Eiffel Tower built?"
- Answer: "The Eiffel Tower was built in 1887."
- Evidence: "The Eiffel Tower was completed in 1889."
- Label: **Not Supported** (incorrect date)

### 3. Contradicted (Label: 2)
The answer directly conflicts with the evidence.

**Criteria:**
- Explicitly contradicts evidence
- Opposite or conflicting information
- Negates verified facts

**Example:**
- Question: "Is the Earth flat?"
- Answer: "Yes, the Earth is flat."
- Evidence: "The Earth is an oblate spheroid."
- Label: **Contradicted**

## Annotation Process

1. **Read the question carefully**
2. **Review the model's answer**
3. **Examine all provided evidence passages**
4. **Determine if the answer is:**
   - Fully supported by evidence → Label 0
   - Contains unverifiable information → Label 1
   - Contradicts evidence → Label 2
5. **Highlight specific evidence** that supports your decision
6. **Note any ambiguous cases** for discussion

## Edge Cases

### Partial Support
If part of the answer is supported but part is not:
- Label as **Not Supported** (Label 1)
- Note which parts are supported in comments

### Reasonable Inference
If the answer makes a reasonable inference from evidence:
- Label as **Supported** (Label 0) if the inference is logical
- Label as **Not Supported** (Label 1) if the inference is a stretch

### Missing Evidence
If evidence is insufficient to verify:
- Label as **Not Supported** (Label 1)
- Note "insufficient evidence" in comments

### Ambiguous Evidence
If evidence is contradictory or unclear:
- Discuss with team lead
- Mark for review
- Use best judgment based on majority of evidence

## Quality Checks

- Aim for consistency across annotations
- Review 10% of annotations with a second annotator
- Calculate inter-annotator agreement (Cohen's kappa)
- Target kappa > 0.7 for good agreement

## Tips

- Be conservative: when in doubt, prefer "Not Supported" over "Supported"
- Focus on factual accuracy, not writing quality
- Ignore minor phrasing differences
- Consider the question context
- Document reasoning for difficult cases

## Common Hallucination Types

1. **Entity Hallucination**: Wrong names, places, organizations
2. **Temporal Errors**: Incorrect dates, time periods
3. **Numerical Errors**: Wrong statistics, measurements
4. **Fabricated Citations**: Non-existent sources
5. **Logical Leaps**: Unsupported conclusions
6. **Conflation**: Mixing facts from different contexts

## Contact

For questions or clarification, contact the annotation team lead.
