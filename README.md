# Legal Hallucination Detection Engine

A research-grade hallucination detection and evaluation framework for **legal language models**, designed to support **epistemically fair comparison** between baseline LLMs, retrieval-augmented generation (RAG), and activation steering methods.

This repository **does not generate answers**. It evaluates factual correctness of generated legal answers under a unified, claim-level verification framework.

---

## Current Research Focus (Dec 2025)

**Evaluating the interaction between Retrieval-Augmented Generation (RAG) and Activation Steering under a unified, correctness-only hallucination definition.**

Key goals:
- Compare RAG and non-RAG systems under **identical factual criteria**
- Measure hallucinations at the **claim level**, not the response level
- Avoid conflating factual correctness with grounding or prompt fidelity
- Provide reviewer-defensible, mode-agnostic metrics

---

## Final Evaluation Results (Threshold-Enforced)

All results below are computed using:
- Claim-level decomposition
- Reference-based NLI verification
- Confidence thresholding (low-confidence → UNSUPPORTED)
- Conditional hallucination rate over adjudicable claims only

| Variant        | Claims | Testable | Testability | Hallucination Rate |
|----------------|--------|----------|-------------|--------------------|
| Baseline       | 196    | 74       | 37.8%       | 95.9%              |
| RAG            | 318    | 100      | 31.4%       | 94.0%              |
| Steering       | 201    | 67       | 33.3%       | **89.6%**          |
| RAG + Steering | 334    | 110      | 32.9%       | 90.9%              |

**Key Findings**
- Activation steering produces the largest reduction in factual contradictions.
- RAG alone yields marginal improvement in this setting.
- RAG + Steering does not exhibit additive synergy, suggesting interaction effects between retrieval context and behavioral control.
- High hallucination rates reflect the difficulty of adversarial legal questions under conservative evaluation, not detector aggressiveness.

---

## What We Mean by “Hallucination”

A hallucination is defined **strictly** as a factual claim that is **contradicted by authoritative legal evidence**.

Each claim is labeled as one of:
- **SUPPORTED** — entailed by evidence
- **CONTRADICTED** — refuted by evidence (hallucination)
- **UNSUPPORTED** — insufficient evidence (epistemic uncertainty)

The primary metric is a **conditional hallucination rate**:

```
Hallucination Rate = CONTRADICTED / (SUPPORTED + CONTRADICTED)
```

Unsupported claims are **excluded** from the denominator and reported separately via **testability**.

---

## Detection Architecture (Claim-Level)

The detector operates through layered verification stages:

1. **Structural Verification**  
   Citation format validation, reporter checks, and existence verification.

2. **Claim Decomposition**  
   Legal-aware segmentation into atomic factual claims.

3. **Semantic Grounding (Bi-Encoder)**  
   Fast relevance filtering using legal-domain embeddings.

4. **NLI Entailment Verification**  
   Cross-encoder classification into SUPPORTED / CONTRADICTED / UNSUPPORTED.

5. **Calibration & Aggregation**  
   Confidence thresholding, testability tracking, and metric aggregation.

---

## Three Detection Pipelines (Epistemically Explicit)

This system provides three detectors, each tied to a specific epistemic assumption.

### 1. NonRAGHallucinationDetector
- **Use case**: Baseline / parametric LLMs
- **Assumption**: Open-world (parametric knowledge)
- **Grounding**: Not required
- **Hallucination**: Contradiction with authoritative law

### 2. RAGHallucinationDetector
- **Use case**: Retrieval-augmented systems
- **Assumption**: Closed-world (retrieved evidence)
- **Grounding**: Applicable
- **Hallucination**: Contradiction with law (correctness-only metric)

### 3. UnifiedHallucinationDetector
- **Use case**: Production or mixed pipelines
- **Assumption**: Explicitly provided or inferred
- **Warnings**: Issued when epistemic mode is ambiguous

For academically rigorous evaluation, **use the mode-specific detectors**.

---

## RAG Implementation Notes

- Hybrid retrieval: BM25 + dense embeddings
- Fixed oracle corpus (CourtListener)
- No finetuning of model weights
- Retrieval context injected only at inference time
- Citation-graph re-ranking is optional and not enabled in reported experiments

---

## Installation

```bash
git clone https://github.com/edoherty2017/LegalLies.git
cd LegalLies
pip install -e .
python -m spacy download en_core_web_sm
```

---

## Intended Use

✔ Research evaluation of legal LLM outputs  
✔ Fair comparison of RAG vs non-RAG systems  
✔ Hallucination analysis under conservative definitions  

✘ Not an answer generator  
✘ Not a legal advice system  

---

## Citation

```bibtex
@software{legal_hallucination_detector,
  title  = {Legal Hallucination Detection Engine},
  author = {Doherty et al.},
  year   = {2025}
}
```

---

