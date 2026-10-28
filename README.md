# Legal Hallucination Detection Engine

A research-grade hallucination detection and evaluation framework for **legal language models**, designed to support **epistemically fair comparison** between baseline LLMs, retrieval-augmented generation (RAG), and activation steering methods.

This repository **does not generate answers**. It evaluates factual correctness of generated legal answers under a unified, claim-level verification framework.

---

## Research Focus

**Evaluating the interaction between Retrieval-Augmented Generation (RAG) and Activation Steering under a unified, correctness-only hallucination definition.**

Key goals:
- Compare RAG and non-RAG systems under **identical factual criteria**
- Measure hallucinations at the **claim level**, not the response level
- Avoid conflating factual correctness with grounding or prompt fidelity
- Provide reviewer-defensible, mode-agnostic metrics

---

## Evaluation Results

**Pilot scale: 15 questions per variant**, the same 15 adversarial legal
questions for all four. Generation is held fixed; the answers are stored in
`data/evaluation_results.csv` and re-scored from there.

All results are computed using:

- Claim-level decomposition
- NLI verification against the fixed oracle corpus, identically for every variant
- Confidence thresholding (low-confidence becomes UNSUPPORTED)
- Conditional hallucination rate over adjudicable claims only, pooled as
  `sum(contradicted) / sum(testable)` across answers

| Variant        | Questions | Claims | Testable | Testability | Hallucination Rate | 95% CI (Wilson) |
|----------------|-----------|--------|----------|-------------|--------------------|-----------------|
| Baseline       | 15        | 186    | 89       | 47.8%       | 83.1%              | 74.0% - 89.5%   |
| RAG            | 15        | 301    | 163      | 54.2%       | 92.6%              | 87.6% - 95.7%   |
| Steering       | 15        | 203    | 79       | 38.9%       | 91.1%              | 82.8% - 95.6%   |
| RAG + Steering | 15        | 326    | 183      | 56.1%       | 92.9%              | 88.2% - 95.8%   |

Reproduce with no third-party dependencies:

```bash
python3 scripts/verify_reported_results.py                           # tables, CIs, significance
python3 scripts/rescore_stored_answers.py --out data/rescored.csv    # re-score stored answers
```

### Does any variant beat another? No.

Two tests, because they answer different questions. Pooled Fisher exact treats
each claim as an independent observation; the paired tests respect the fact that
all four variants answered the same 15 questions, and are the appropriate test
for this design.

| Comparison                 | Fisher (pooled claims) | Sign test (paired) | Wilcoxon (paired, exact) |
|----------------------------|------------------------|--------------------|--------------------------|
| Baseline vs RAG            | 0.031                  | 0.774              | 0.197                    |
| Baseline vs Steering       | 0.169                  | 0.070              | 0.195                    |
| Baseline vs RAG + Steering | 0.019                  | 0.344              | 0.160                    |
| RAG vs Steering            | 0.799                  | 0.344              | 0.740                    |
| RAG vs RAG + Steering      | 1.000                  | 1.000              | 0.685                    |
| Steering vs RAG + Steering | 0.619                  | 0.508              | 0.883                    |

**Nothing is significant under the paired tests.** The two pooled Fisher results
below 0.05 should not be cited as findings: with a thousand-odd claims nested in
only 15 questions, treating claims as independent overstates the evidence, which
is exactly what the paired tests correct for.

**What can be said**

- The detector runs end-to-end and produces claim-level verdicts with explicit
  testability accounting.
- Fewer than 60% of generated legal claims can be adjudicated against the oracle
  corpus at all (testability 38.9-56.1%). Corpus coverage, not model quality, is
  the binding constraint on this kind of evaluation.
- Among adjudicable claims, every variant is wrong most of the time (83-93%) on
  adversarial questions under a contradiction-only definition.

**What cannot be said**

- That activation steering reduces contradictions. It does not: Steering (91.1%)
  scores numerically *worse* than Baseline (83.1%), and the difference is not
  significant (paired sign test p=0.070, Wilcoxon p=0.195). The sign is the
  opposite of what an earlier revision of this README claimed.
- That RAG helps, or that RAG and steering interact. No paired comparison comes
  close to significance.

Separating these variants at these effect sizes needs a substantially larger
question set, which means re-running generation, not re-scoring.

### A note on reproducibility

These figures come from re-scoring the stored answers with the corrected
detector (see *Scoring corrections* below). An earlier revision of this README
reported 95.9 / 94.0 / 89.6 / 90.9% from the original December 2025 run.

Those older figures are not directly comparable, for two reasons. The detector
was wrong in ways that moved the numbers. And the environment moved too: this
project pins `transformers ^4.30` but current installs resolve to 5.x, and NLI
verdicts shift between those versions - per-answer verdicts move substantially
even when the decomposed claim text is byte-identical.

So the comparison that is actually controlled is old detector versus new
detector *within one environment*. Running the pre-fix detector in this
environment reproduces every Baseline and Steering claim count exactly (30 of 30
answers) at 91.8% and 90.0%, which is what establishes that the harness is
faithful and that the differences above come from the detector changes rather
than the library upgrade. Both runs, and the full comparison, are in
[`docs/experiments/rescore-2026-10-28.md`](docs/experiments/rescore-2026-10-28.md).

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

1. **Claim Decomposition** — legal-aware segmentation into atomic factual
   claims, via spaCy with legal abbreviation exceptions so citations are not
   split mid-claim (`preprocessing/claim_decomposer.py`,
   `preprocessing/legal_segmenter.py`).

2. **Per-Claim Evidence Retrieval** — hybrid BM25 + dense retrieval from the
   fixed oracle corpus, run once per claim
   (`retrieval/document_retriever.py`).

3. **NLI Entailment Verification** — cross-encoder classification into
   SUPPORTED / CONTRADICTED / UNSUPPORTED, batched per claim
   (`detection/entailment_verifier.py`).

4. **Calibration & Aggregation** — confidence thresholding, testability
   tracking, and metric aggregation.

Structural citation verification and bi-encoder semantic matching were
prototyped but are not part of the pipeline; they remain in
`src/detection/_archived/` and contribute to no reported number.

---

## One Detector, Two Wrappers

There is a single detection method. `FactualityDetector` verifies every claim
against the fixed oracle corpus and labels it SUPPORTED / CONTRADICTED /
UNSUPPORTED. RAG and non-RAG outputs are judged by **identical factual
criteria** — that symmetry is what makes the variants comparable.

| Wrapper | Evidence used | For |
|---|---|---|
| `NonRAGHallucinationDetector` | oracle corpus, retrieved per claim | baseline / parametric output |
| `RAGHallucinationDetector` | oracle corpus by default; the supplied context with `grade_against="context"` | RAG output |

`UnifiedHallucinationDetector` is a backwards-compatibility alias for
`FactualityDetector`, not a third pipeline.

The RAG wrapper's default matters. Grading RAG answers against the passages the
generator was shown measures *faithfulness* — whether the answer follows from
those passages — which is a different question from whether the claim is true,
and is not comparable with a baseline graded against the corpus. Both are
available; only the corpus default is comparable across variants.

Non-RAG detection is **not reference-free**: it retrieves evidence post-hoc to
approximate what the model should have known. Self-consistency and
semantic-entropy methods are not implemented here.

## Scoring corrections

The detector was audited in October 2026 and several defects that moved the
reported numbers were fixed. Briefly: citations such as `42 U.S.C. § 1983` were
split into fragments that were then counted as claims; evidence was retrieved
once per answer and reused for every claim; a single contradicting passage
outvoted passages that supported the claim; RAG answers were graded against
their own retrieved context rather than the corpus; an answer with nothing
testable reported a 0% hallucination rate rather than an undefined one; and
component load failures were swallowed, so a broken environment produced a clean
0%. The citation graph built almost no edges, leaving authority re-ranking inert.

Each fix is a separate commit with its reasoning, and the resulting change in
every number is recorded in
[`docs/experiments/rescore-2026-10-28.md`](docs/experiments/rescore-2026-10-28.md).

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

Suitable for:

- Research evaluation of legal LLM outputs
- Comparison of RAG vs non-RAG systems under identical criteria
- Hallucination analysis under conservative definitions

Not suitable for:

- Generating answers; this repository evaluates them
- Legal advice
- Established claims about RAG or steering effectiveness: at 15 questions, no
  variant difference is statistically significant

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

