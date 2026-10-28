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

## Preliminary Evaluation Results (Threshold-Enforced)

> **Sample size: 15 questions per variant** (the same 15 adversarial legal questions
> for all four variants). These are **pilot-scale** results. No pairwise difference
> between variants is statistically significant — see *Statistical power* below.
> Do not cite the variant ranking as a finding.

All results below are computed using:
- Claim-level decomposition
- Reference-based NLI verification
- Confidence thresholding (low-confidence → UNSUPPORTED)
- Conditional hallucination rate over adjudicable claims only

| Variant        | Questions | Claims | Testable | Testability | Hallucination Rate | 95% CI (Wilson) |
|----------------|-----------|--------|----------|-------------|--------------------|-----------------|
| Baseline       | 15        | 196    | 74       | 37.8%       | 95.9%              | 88.7% – 98.6%   |
| RAG            | 15        | 318    | 100      | 31.4%       | 94.0%              | 87.5% – 97.2%   |
| Steering       | 15        | 201    | 67       | 33.3%       | 89.6%              | 80.0% – 94.8%   |
| RAG + Steering | 15        | 334    | 110      | 32.9%       | 90.9%              | 84.1% – 95.0%   |

Reproduce everything in this section — the table, the confidence intervals and the
significance tests — with no third-party dependencies:

```bash
python3 scripts/verify_reported_results.py
```

It reads only `data/evaluation_results.csv` (per-response `num_supported` /
`num_contradicted`; the hallucination rate is
`contradicted / (supported + contradicted)`).

### Statistical power

Two-sided Fisher exact tests on CONTRADICTED vs SUPPORTED claim counts, all six
variant pairs:

| Comparison                      | p-value | Significant at α=0.05 |
|---------------------------------|---------|-----------------------|
| Baseline vs RAG                 | 0.735   | no                    |
| Baseline vs Steering            | 0.193   | no                    |
| Baseline vs RAG + Steering      | 0.248   | no                    |
| RAG vs Steering                 | 0.379   | no                    |
| RAG vs RAG + Steering           | 0.445   | no                    |
| Steering vs RAG + Steering      | 0.796   | no                    |

**None reach significance.** The 95% confidence intervals above overlap
substantially. Moreover, this test treats each claim as an independent
observation, but claims are nested within only 15 questions per variant; properly
accounting for that clustering would *widen* the intervals further. The observed
ordering is therefore **not distinguishable from sampling noise**.

**What can and cannot be said**
- ✔ The detector runs end-to-end and produces claim-level verdicts with explicit
  testability accounting.
- ✔ Testability is low (31–38%) across all variants: most generated legal claims
  could not be adjudicated against the oracle corpus at all. This is itself a
  substantive result about corpus coverage.
- ✔ Hallucination rates among adjudicable claims are high (≈90–96%) for every
  variant, reflecting adversarial question difficulty under a conservative
  contradiction-only definition.
- ✘ **Not supported**: that activation steering reduces contradictions. The
  numerically lowest rate (Steering, 89.6%) is not significantly below Baseline
  (95.9%), p=0.193.
- ✘ **Not supported**: that RAG helps, or that RAG + Steering interact
  non-additively. Both comparisons are well short of significance.

Establishing any of the ✘ claims requires a substantially larger question set.

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

`FactualityDetector` runs four stages:

1. **Claim Decomposition**  
   Legal-aware segmentation into atomic factual claims
   (`preprocessing/claim_decomposer.py`).

2. **Evidence Retrieval**  
   Hybrid retrieval from the fixed oracle corpus
   (`retrieval/document_retriever.py`).

3. **NLI Entailment Verification**  
   Cross-encoder classification into SUPPORTED / CONTRADICTED / UNSUPPORTED
   (`detection/entailment_verifier.py`).

4. **Calibration & Aggregation**  
   Confidence thresholding, testability tracking, and metric aggregation.

**Not in the active pipeline.** Structural citation verification and bi-encoder
semantic matching were prototyped but are **not wired into** `FactualityDetector`;
they live in `src/detection/_archived/` and did not contribute to the results
above. Earlier revisions of this README described a five-stage pipeline including
them — that was inaccurate.

---

## One Detector, Two Convenience Wrappers (Epistemically Symmetric)

**There is a single detection method.** `FactualityDetector` verifies every claim
against the fixed oracle corpus and labels it SUPPORTED / CONTRADICTED /
UNSUPPORTED. RAG and non-RAG outputs are judged by **identical factual criteria** —
that symmetry is the design intent (see *Current Research Focus* above), not an
oversight.

The two wrappers differ **only** in where evidence comes from:

| Wrapper | Evidence passed to `FactualityDetector` | Intended input |
|---|---|---|
| `RAGHallucinationDetector` | the `retrieved_context` you supply | output of a RAG system, verified against the context it was shown |
| `NonRAGHallucinationDetector` | `None` → auto-retrieved from the oracle corpus | output of a baseline/parametric model, verified post-hoc |

`UnifiedHallucinationDetector` is a **backwards-compatibility alias** for
`FactualityDetector` (`detection/unified_detector.py`), not a third pipeline.

### Honest limitations of this design

- Non-RAG detection is **not reference-free**. It retrieves evidence post-hoc to
  approximate what the model should have known. Self-consistency and
  internal-contradiction methods (SelfCheckGPT, semantic entropy) are **not
  implemented in this repository**.
- Because non-RAG claims are verified against the same corpus RAG retrieves from,
  cross-mode comparisons inherit that corpus as a shared ceiling. The low
  testability rates (31–38%) are the visible symptom.
- No per-mode epistemic warnings are emitted. `GenerationMode` and
  `EpistemicContext` were defined in an earlier revision but never wired into the
  detector, and have been removed to avoid implying behavior that does not exist.

Earlier revisions of this README described three detectors "each tied to a specific
epistemic assumption." That was inaccurate: the three names resolve to one method.

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
✔ Mode-symmetric comparison of RAG vs non-RAG systems under identical criteria  
✔ Hallucination analysis under conservative definitions  

✘ Not an answer generator  
✘ Not a legal advice system  
✘ Not a source of statistically established claims about RAG or steering
  effectiveness — the reported results are pilot-scale (15 questions) and no
  variant difference is significant  

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

