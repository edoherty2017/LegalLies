# Why Three Hallucination Detection Pipelines?

> ### ⚠ Implementation note (2026-10)
> **This document describes a design that was not implemented.** The shipped code
> has **one** detection method, `FactualityDetector`, which verifies every claim
> against the oracle corpus regardless of how the answer was generated — detection
> is epistemically *symmetric*, not split by mode.
>
> - `RAGHallucinationDetector` and `NonRAGHallucinationDetector` are thin wrappers
>   that differ only in whether evidence is supplied by the caller or auto-retrieved.
> - `UnifiedHallucinationDetector` is a backwards-compatibility **alias** for
>   `FactualityDetector`, not a third pipeline.
> - The `GenerationMode` / `EpistemicContext` types this document refers to have
>   been removed; they were never referenced by the detector.
> - No reference-free (self-consistency / semantic-entropy) detector exists in this
>   repository.
>
> The rationale below is retained because the epistemic argument is sound and
> motivates future work — but **do not read it as a description of this codebase.**
> See the README section "One Detector, Two Convenience Wrappers" for what actually runs.

**TL;DR**: RAG and non-RAG outputs are **epistemically different**. Comparing them requires maintaining epistemic clarity through separate, purpose-built detectors.

---

## The Core Problem: Epistemic Ambiguity

When evaluating a claim like *"Murder is a felony under federal law"*, the question *"Is this hallucinated?"* has **different meanings** depending on how the answer was generated:

### Non-RAG Generation (Parametric)
```python
# Model generates from training data (parametric knowledge)
answer = llama_baseline.generate("What is murder?")
# Output: "Murder is a felony under federal law..."
```

**Evidence universe**: Model's parametric knowledge (weights)
**Question**: Did the model hallucinate something **not in its training data**?
**Verification approach**: Retrieve evidence **post-hoc** to approximate factual knowledge

### RAG Generation (Retrieval-Augmented)
```python
# Model generates with explicit evidence documents
passages = retriever.retrieve("What is murder?")
answer = llama_rag.generate("What is murder?", context=passages)
# Output: "According to 18 U.S.C. § 1111, murder is..."
```

**Evidence universe**: Retrieved documents shown to model
**Question**: Did the model hallucinate something **not in the retrieved evidence**?
**Verification approach**: Verify against the **exact documents** shown to model during generation

---

## Why This Matters for Academic Rigor

### Problem 1: Naive Unified Detection

❌ **Epistemically invalid approach**:
```python
# BAD: Treats all outputs identically
detector = NaiveDetector()

# Non-RAG output
result1 = detector.detect(query, baseline_answer, retrieved_context=None)

# RAG output
result2 = detector.detect(query, rag_answer, retrieved_context=passages)

# INVALID COMPARISON: Different evidence universes!
# result1 checked against post-hoc retrieval
# result2 checked against generation-time evidence
# These are NOT epistemically comparable
```

**Why it's broken**:
- Non-RAG: Detector auto-retrieves evidence **the model never saw**
- RAG: Detector uses evidence **the model actually used**
- Comparing failure rates is **epistemically meaningless**

### Problem 2: Unlabeled Epistemic Context

❌ **Loss of research validity**:
```python
# BAD: No tracking of generation mode
results = [
    detector.detect(q, a1, ctx1),  # RAG output
    detector.detect(q, a2, None),  # Non-RAG output (auto-retrieves)
    detector.detect(q, a3, ctx3),  # RAG output
]

# Cannot distinguish:
# - Which results are RAG vs non-RAG?
# - Which evidence was seen during generation?
# - Which comparisons are epistemically valid?
```

**Research impact**:
- Cannot publish comparative results
- Reviewers will reject due to epistemic ambiguity
- Results are not reproducible

---

## The Solution: Three Epistemically Distinct Pipelines

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   SHARED INFRASTRUCTURE                     │
│  (UnifiedHallucinationDetector with 5-layer architecture)  │
└──────────────┬──────────────┬──────────────┬───────────────┘
               │              │              │
       ┌───────▼─────┐  ┌────▼────┐  ┌──────▼──────┐
       │ RAG Pipeline│  │ Non-RAG │  │   Unified   │
       │  (Detector) │  │Pipeline │  │  Pipeline   │
       └─────────────┘  └─────────┘  └─────────────┘
```

### 1. RAG Hallucination Detector

**Purpose**: Detect grounding failures in RAG outputs

**Epistemic assumptions**:
- Evidence universe: **Retrieved documents only** (closed-world)
- Evidence requirement: **MUST provide** exact documents shown to model
- Hallucination definition: Claims **not grounded in retrieved evidence**
- Interpretation: **Strict** - model should not use parametric knowledge

**Usage**:
```python
from src.detection import RAGHallucinationDetector

detector = RAGHallucinationDetector()

# RAG generation
passages = retriever.retrieve(query)
answer = rag_model.generate(query, context=passages)

# Detection with closed-world assumption
result = detector.detect(
    query=query,
    generated_answer=answer,
    retrieved_context=passages  # REQUIRED - exact evidence shown to model
)

# Epistemic context automatically tracked
assert result.epistemic_context.generation_mode == GenerationMode.RAG
assert result.epistemic_context.closed_world_assumption == True
```

**Academic validity**:
✅ Clear evidence universe (retrieved documents)
✅ Epistemic assumptions documented
✅ Results comparable across RAG systems
✅ Reviewers can verify methodology

---

### 2. Non-RAG Hallucination Detector

**Purpose**: Detect parametric failures in pure generation

**Epistemic assumptions**:
- Evidence universe: **Model's parametric knowledge** (open-world)
- Evidence requirement: **OPTIONAL** - auto-retrieves for verification
- Hallucination definition: Claims **contradicting factual knowledge**
- Interpretation: **Lenient** - model uses internal knowledge

**Usage**:
```python
from src.detection import NonRAGHallucinationDetector

detector = NonRAGHallucinationDetector()

# Parametric generation (no retrieval)
answer = baseline_model.generate(query)

# Detection with automatic evidence retrieval
result = detector.detect(
    query=query,
    generated_answer=answer,
    # No retrieved_context - detector auto-retrieves for verification
)

# Epistemic context automatically tracked
assert result.epistemic_context.generation_mode == GenerationMode.NON_RAG
assert result.epistemic_context.evidence_provided_to_model == False
```

**Key insight**:
The detector **auto-retrieves evidence** to approximate what the model "should" know, but this evidence **was not shown to the model during generation**. This is a **post-hoc approximation** of parametric knowledge.

**Academic validity**:
✅ Clear evidence universe (parametric knowledge)
✅ Post-hoc retrieval explicitly documented
✅ Results comparable across baseline models
✅ Epistemic assumptions transparent

---

### 3. Unified Hallucination Detector

**Purpose**: Production use with explicit epistemic tracking

**Epistemic assumptions**:
- Evidence universe: **Specified via epistemic_context**
- Evidence requirement: **Flexible** - supports both modes
- Hallucination definition: **Context-dependent**
- Interpretation: **Warns** about ambiguity when mode unknown

**Usage**:
```python
from src.detection import UnifiedHallucinationDetector
from src.core.config import EpistemicContext, GenerationMode

detector = UnifiedHallucinationDetector()

# Explicit epistemic context for RAG
epistemic_ctx = EpistemicContext(
    generation_mode=GenerationMode.RAG,
    evidence_scope="retrieved",
    evidence_provided_to_model=True,
    closed_world_assumption=True,
)

result = detector.detect(
    query=query,
    generated_answer=answer,
    retrieved_context=passages,
    epistemic_context=epistemic_ctx  # Explicitly track mode
)

# Or use without context (production mode)
result = detector.detect(query, answer, context)
# Will emit epistemic warning:
assert "EPISTEMIC_AMBIGUITY" in result.epistemic_warnings[0]
```

**When to use**:
- ✅ Production systems with mixed modes
- ✅ Unknown/hybrid generation strategies
- ✅ Explicit epistemic tracking required
- ❌ Academic comparisons (use specific detectors instead)

**Academic validity**:
⚠️ Valid **only if** epistemic_context is provided
❌ **Not valid** if used with UNKNOWN mode
✅ Enables custom epistemic assumptions

---

## Epistemological Foundation

### What is "Epistemic"?

**Epistemology** = the study of knowledge and how we know what we know

**Epistemic context** = the framework for determining what counts as "knowledge" in a given situation

### Why RAG and Non-RAG are Epistemically Different

| Dimension | Non-RAG | RAG |
|-----------|---------|-----|
| **Knowledge source** | Parametric weights | Retrieved documents |
| **Evidence availability** | Implicit (training data) | Explicit (shown documents) |
| **Verifiability** | Approximate (post-hoc retrieval) | Exact (same documents) |
| **Hallucination type** | Parametric failure | Grounding failure |
| **Failure semantics** | "Model made up a fact" | "Model ignored evidence" |
| **Counterfactual** | "What should model know?" | "What did model see?" |

**Key insight**: These are **different types of errors** with **different causes** and **different solutions**.

Treating them as identical is like:
- Comparing "lying" vs "being mistaken"
- Comparing "fabrication" vs "misquotation"
- Comparing "hallucination" vs "reading comprehension failure"

They're **related but distinct** phenomena.

---

## Research Precedent

### How Papers Handle This

Even papers that claim to "unify" RAG and non-RAG evaluation **actually use different assumptions**:

#### Example 1: "Reducing Hallucinations in RAG"
```
"RAG reduces hallucinations by 23%"
```

**What they actually did**:
1. Evaluated baseline with **universal retrieval** (post-hoc)
2. Evaluated RAG with **generation-time evidence**
3. Compared rates **within those different assumptions**
4. Did NOT run the same detector blind

#### Example 2: "Citation-Guided Generation"
```
"Models with citations hallucinate 40% less"
```

**What they actually did**:
1. RAG outputs: Checked **citation grounding** in sources
2. Baseline outputs: Checked **factual correctness** (no citations)
3. Different metrics, different assumptions
4. Explicit about epistemic differences

### Research Community Consensus

**Standard practice** in hallucination detection research:
- ✅ Separate evaluation for RAG vs non-RAG
- ✅ Explicit about evidence universes
- ✅ Document epistemic assumptions
- ✅ Use different metrics/thresholds by mode
- ❌ NEVER compare naively without context

---

## Implementation Strategy

### Why Shared Infrastructure?

**Question**: If pipelines are different, why share the UnifiedHallucinationDetector?

**Answer**: The **detection layers** (NLI, semantic matching, etc.) are the same. What differs is the **epistemic interpretation**.

```
SHARED:
✅ Claim decomposition logic
✅ Semantic grounding verification
✅ NLI entailment models
✅ Citation verification
✅ Scoring algorithms

DIFFERENT:
❌ Evidence universe definition
❌ Verification strategy (auto-retrieve vs provided)
❌ Failure interpretation
❌ Threshold recommendations
❌ Result semantics
```

### Design Pattern: Wrapper + Context

```python
class RAGHallucinationDetector:
    def __init__(self):
        # Shared infrastructure
        self.unified_detector = UnifiedHallucinationDetector()

    def detect(self, query, answer, retrieved_context):
        # Create epistemic context
        context = EpistemicContext(
            generation_mode=GenerationMode.RAG,
            evidence_scope="retrieved",
            evidence_provided_to_model=True,
            closed_world_assumption=True,
        )

        # Call shared detector with RAG-specific assumptions
        return self.unified_detector.detect(
            query,
            answer,
            retrieved_context,
            epistemic_context=context
        )
```

**Benefits**:
- Code reuse (DRY principle)
- Consistent core logic
- Clear epistemic boundaries
- Easy to maintain
- Academic rigor preserved

---

## Usage Guidelines

### For Academic Research

**Comparative evaluation** (RAG vs non-RAG):
```python
# Use separate detectors for epistemic clarity
rag_detector = RAGHallucinationDetector()
nonrag_detector = NonRAGHallucinationDetector()

# Evaluate RAG outputs
rag_results = [
    rag_detector.detect(q, rag_answer, passages)
    for q, rag_answer, passages in rag_data
]

# Evaluate non-RAG outputs
nonrag_results = [
    nonrag_detector.detect(q, baseline_answer)
    for q, baseline_answer in baseline_data
]

# Now comparison is epistemically valid
rag_halluc_rate = sum(r.is_hallucinated for r in rag_results) / len(rag_results)
nonrag_halluc_rate = sum(r.is_hallucinated for r in nonrag_results) / len(nonrag_results)

print(f"RAG hallucination rate: {rag_halluc_rate:.2%}")
print(f"Non-RAG hallucination rate: {nonrag_halluc_rate:.2%}")
print(f"RAG reduces hallucinations by {(1 - rag_halluc_rate/nonrag_halluc_rate):.1%}")
```

**Reviewers will approve** because:
- ✅ Explicit epistemic assumptions
- ✅ Separate evaluation pipelines
- ✅ Tracked evidence universes
- ✅ Reproducible methodology

### For Production Deployment

**Drop-in integration** with partner's stack:
```python
# Use specific detector matching generation mode
if generation_mode == "rag":
    detector = RAGHallucinationDetector()
    result = detector.detect(query, answer, retrieved_context)
else:
    detector = NonRAGHallucinationDetector()
    result = detector.detect(query, answer)

# Epistemic context automatically tracked
metrics = {
    "is_hallucinated": result.is_hallucinated,
    "hallucination_risk": result.hallucination_risk,
    "generation_mode": result.epistemic_context.generation_mode.value,
    "evidence_scope": result.epistemic_context.evidence_scope,
}
```

### For Unknown/Hybrid Modes

**Use unified detector with warnings**:
```python
detector = UnifiedHallucinationDetector()

result = detector.detect(query, answer, context)

# Check for epistemic warnings
if result.epistemic_warnings:
    logger.warning(f"Epistemic ambiguity detected: {result.epistemic_warnings}")
    # Still usable, but not academically rigorous
```

---

## Frequently Asked Questions

### Q1: Isn't this overengineering?

**A**: No. This is the **minimum complexity** required for academic rigor.

Research papers that ignore epistemic distinctions:
- Get rejected by reviewers
- Produce misleading results
- Cannot be reproduced
- Waste months of work

**3 detectors** vs **months of rejected papers** = worth it.

### Q2: Can't I just add a "mode" flag to one detector?

**A**: You could, but then:
- Users must remember to set the flag correctly
- Easy to forget and get invalid results
- No compile-time safety
- Harder to find in repository

**Separate classes** enforce correctness at the API level.

### Q3: What if I'm comparing RAG with citations vs RAG without citations?

**A**: Both are RAG (closed-world assumption). Use `RAGHallucinationDetector` for both and compare within that epistemic framework.

The key distinction is **evidence universe**, not features.

### Q4: What about RAG + Web search (hybrid)?

**A**: Hybrid modes are epistemically complex. Options:
1. Treat as RAG if web results are shown to model
2. Create a fourth detector with custom epistemic assumptions
3. Use UnifiedDetector with explicit epistemic context

Consult `docs/architecture/UNIFIED_DETECTION_ARCHITECTURE.md` for hybrid modes.

### Q5: Do I really need 4 documentation files?

**A**: Yes, for academic and engineering clarity:
- **WHY_THREE_PIPELINES.md** (this file): Rationale and theory
- **RAG_DETECTION_ARCHITECTURE.md**: RAG detector implementation
- **NONRAG_DETECTION_ARCHITECTURE.md**: Non-RAG detector implementation
- **UNIFIED_DETECTION_ARCHITECTURE.md**: Unified detector + epistemic tracking

Each serves a different audience:
- Researchers: WHY_THREE_PIPELINES.md
- RAG users: RAG_DETECTION_ARCHITECTURE.md
- Baseline users: NONRAG_DETECTION_ARCHITECTURE.md
- Power users: UNIFIED_DETECTION_ARCHITECTURE.md

---

## Summary

### Three Pipelines, Three Purposes

| Pipeline | Purpose | Users | Academic Validity |
|----------|---------|-------|-------------------|
| **RAG Detector** | Grounding failure detection | RAG systems | ✅ Valid for RAG evaluation |
| **Non-RAG Detector** | Parametric failure detection | Baseline models | ✅ Valid for baseline evaluation |
| **Unified Detector** | Flexible with tracking | Production/hybrid | ⚠️ Valid only with epistemic_context |

### The Bottom Line

**Epistemic clarity is not optional for academic rigor.**

RAG and non-RAG are **fundamentally different** in their relationship to evidence. Treating them identically produces **epistemically invalid** comparisons.

Three detectors = **minimum viable academic rigor**.

---

## See Also

- [RAG Detection Architecture](RAG_DETECTION_ARCHITECTURE.md) - RAG detector implementation
- [Non-RAG Detection Architecture](NONRAG_DETECTION_ARCHITECTURE.md) - Non-RAG detector implementation
- [Unified Detection Architecture](UNIFIED_DETECTION_ARCHITECTURE.md) - Unified detector with epistemic tracking
- [HALLUCINATION_DETECTION_ARCHITECTURE.md](../../HALLUCINATION_DETECTION_ARCHITECTURE.md) - Overall system architecture

---

**Questions?** Open an issue or consult the academic references in `docs/references.md`.
