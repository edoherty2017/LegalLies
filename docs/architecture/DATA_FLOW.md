# Hallucination Detection Pipeline: Complete Data Flow

**Version**: 1.1
**Last Updated**: 2026-10-28
**Status**: ⚠ Partially stale - see note below

> ### WARNING - Implementation note (2026-10)
> This document was previously marked "verified against actual code." It is not.
> The steps describing **epistemic mode routing** and **"emit epistemic ambiguity
> warning"** (around lines 221 and 407 below) describe behavior that does **not
> exist** in `FactualityDetector`; the `GenerationMode` / `EpistemicContext` types
> they depend on were never wired in and have been removed.
>
> What actually runs is a single symmetric path: claim decomposition -> oracle-corpus
> retrieval -> NLI entailment -> threshold/aggregation. The claim-decomposition,
> retrieval, NLI and aggregation steps described here are accurate.

This document traces the complete journey of a response through the hallucination detection system, from user input to final verdict.

---

## Table of Contents

- [Overview](#overview)
- [Complete Data Flow Diagram](#complete-data-flow-diagram)
- [Layer-by-Layer Breakdown](#layer-by-layer-breakdown)
- [Entry Point Selection](#entry-point-selection)
- [Example Trace](#example-trace)
- [Code References](#code-references)

---

## Overview

The hallucination detection system processes responses through a **5-layer pipeline** orchestrated by `UnifiedHallucinationDetector`. Users choose one of **three entry points** (RAG, NonRAG, or Unified detector) based on how the response was generated, which determines the **epistemic context** for evaluation.

**Key Principle**: Same detection layers, different epistemic interpretations.

---

## Complete Data Flow Diagram

```
USER INPUT (query, answer, context)
            │
            ▼
┌───────────────────────────────────────────────────────┐
│  ENTRY POINT: One of Three Detector Classes           │
│  User selects based on generation method              │
└───────┬───────────┬───────────┬───────────────────────┘
        │           │           │
   ┌────▼────┐ ┌───▼────┐ ┌────▼──────┐
   │   RAG   │ │NonRAG  │ │ Unified   │
   │Detector │ │Detector│ │ Detector  │
   └────┬────┘ └───┬────┘ └────┬──────┘
        │           │           │
        │ Creates   │ Creates   │ Optional
        │ RAG mode  │ NON_RAG   │ epistemic
        │ epistemic │ epistemic │ context
        │ context   │ context   │
        └───────────┴───────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│         UnifiedHallucinationDetector.detect()           │
│         (The Core 5-Layer Pipeline)                     │
│                                                          │
│  INPUT: query, answer, context, epistemic_context       │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │ Layer 0: Epistemic Context Initialization      │    │
│  │  - Create/validate epistemic context           │    │
│  │  - Emit warnings if UNKNOWN mode               │    │
│  │  - Track generation mode (RAG/NON_RAG/UNKNOWN) │    │
│  │                                                 │    │
│  │  Code: unified_detector.py:214-226             │    │
│  └────────────────┬───────────────────────────────┘    │
│                   │                                      │
│                   ▼                                      │
│  ┌────────────────────────────────────────────────┐    │
│  │ Auto-Retrieval (Optional)                       │    │
│  │  - If context=None and retriever enabled       │    │
│  │  - RAG mode: Use provided context only         │    │
│  │  - NonRAG mode: Auto-retrieve for verification │    │
│  │  - Unified mode: Auto-retrieve if needed       │    │
│  │                                                 │    │
│  │  Code: unified_detector.py:229-262             │    │
│  └────────────────┬───────────────────────────────┘    │
│                   │                                      │
│                   ▼                                      │
│  ┌────────────────────────────────────────────────┐    │
│  │ Layer 1: Structural Verification                │    │
│  │  - Citation extraction & normalization          │    │
│  │  - Citation context cross-check                │    │
│  │  - Format validation (blocklist, reporters)    │    │
│  │  - Date consistency check                      │    │
│  │  ⚡ FAST-FAIL: Return immediately if invalid   │    │
│  │                                                 │    │
│  │  Code: unified_detector.py:265-290             │    │
│  │  Component: StructuralVerifier                 │    │
│  └────────────────┬───────────────────────────────┘    │
│                   │                                      │
│                   ▼                                      │
│  ┌────────────────────────────────────────────────┐    │
│  │ Layer 2: Claim Decomposition                    │    │
│  │  - Legal-aware sentence splitting               │    │
│  │  - Quality gate validation (threshold=0.8)     │    │
│  │  - Minimum length enforcement                  │    │
│  │  ⚡ Early return if no claims extracted        │    │
│  │                                                 │    │
│  │  Code: unified_detector.py:293-317             │    │
│  │  Component: ClaimDecomposer                    │    │
│  └────────────────┬───────────────────────────────┘    │
│                   │                                      │
│                   ▼                                      │
│  ┌────────────────────────────────────────────────┐    │
│  │ PER-CLAIM VERIFICATION LOOP                     │    │
│  │  For each decomposed claim:                    │    │
│  │  ┌──────────────────────────────────────────┐  │    │
│  │  │ Layer 3: Semantic Grounding (Bi-Encoder) │  │    │
│  │  │  - Fast cosine similarity filter         │  │    │
│  │  │  - Compare claim vs evidence embeddings  │  │    │
│  │  │  - Top-K evidence retrieval per claim    │  │    │
│  │  │  - Threshold check (default: 0.50)       │  │    │
│  │  │                                           │  │    │
│  │  │  If score < threshold:                   │  │    │
│  │  │    ❌ Mark claim as failed (semantic)    │  │    │
│  │  │    ⏭️ Skip expensive NLI, next claim      │  │    │
│  │  │                                           │  │    │
│  │  │  Code: unified_detector.py:326-352       │  │    │
│  │  │  Component: SemanticGroundingVerifier    │  │    │
│  │  └──────────────┬───────────────────────────┘  │    │
│  │                 │                                │    │
│  │                 ▼                                │    │
│  │  ┌──────────────────────────────────────────┐  │    │
│  │  │ Layer 4: NLI Entailment (Cross-Encoder)  │  │    │
│  │  │  - Precise entailment verification       │  │    │
│  │  │  - Multi-source aggregation              │  │    │
│  │  │  - Check for CONTRADICTION               │  │    │
│  │  │  - Check for ENTAILMENT                  │  │    │
│  │  │  - Threshold check (default: 0.70)       │  │    │
│  │  │                                           │  │    │
│  │  │  If CONTRADICTION:                       │  │    │
│  │  │    ❌ Mark claim as failed (contradiction)│ │    │
│  │  │  Else if entailment < threshold:         │  │    │
│  │  │    ❌ Mark claim as failed (entailment)  │  │    │
│  │  │  Else:                                   │  │    │
│  │  │    ✅ Mark claim as grounded             │  │    │
│  │  │                                           │  │    │
│  │  │  Code: unified_detector.py:354-411       │  │    │
│  │  │  Component: EntailmentVerifier           │  │    │
│  │  └──────────────────────────────────────────┘  │    │
│  └────────────────┬───────────────────────────────┘    │
│                   │                                      │
│                   ▼                                      │
│  ┌────────────────────────────────────────────────┐    │
│  │ Layer 5: Aggregation & Result Construction     │    │
│  │  - Count failed vs grounded claims             │    │
│  │  - Calculate failure_rate (failed/total)       │    │
│  │  - Calculate hallucination_risk (0-1)          │    │
│  │  - Determine hallucination_type:               │    │
│  │    • "contradiction" if any contradictions     │    │
│  │    • "semantic" if semantic failures           │    │
│  │    • "entailment" if entailment failures       │    │
│  │  - Calculate confidence (avg entailment scores)│    │
│  │  - Attach epistemic_context to result          │    │
│  │  - Attach epistemic_warnings (if any)          │    │
│  │                                                 │    │
│  │  Code: unified_detector.py:420-468             │    │
│  └────────────────┬───────────────────────────────┘    │
│                   │                                      │
│                   ▼                                      │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│              HallucinationResult                       │
│  {                                                     │
│    # Overall verdict                                  │
│    is_hallucinated: bool                              │
│    hallucination_risk: float (0-1)                    │
│    confidence: float (0-1)                            │
│                                                        │
│    # Hallucination details                            │
│    hallucination_type: str | None                     │
│      ("structural", "semantic", "contradiction")      │
│    failure_reason: str | None                         │
│                                                        │
│    # Layer-specific results                           │
│    structural_passed: bool                            │
│    structural_details: dict                           │
│    claims: List[DecomposedClaim]                      │
│    claim_results: List[dict]                          │
│                                                        │
│    # Raw scores for analysis                          │
│    semantic_scores: List[float]                       │
│      (groundedness per claim)                         │
│    entailment_scores: List[float]                     │
│      (correctness per claim)                          │
│                                                        │
│    # Epistemic tracking (for academic rigor)          │
│    epistemic_context: EpistemicContext {              │
│      generation_mode: RAG | NON_RAG | UNKNOWN         │
│      evidence_scope: str                              │
│      evidence_provided_to_model: bool                 │
│      closed_world_assumption: bool                    │
│    }                                                   │
│    epistemic_warnings: List[str]                      │
│                                                        │
│    # Telemetry                                        │
│    latency_ms: float                                  │
│    layer_latencies: dict                              │
│  }                                                     │
└───────────────────────────────────────────────────────┘
```

---

## Layer-by-Layer Breakdown

### Layer 0: Epistemic Context Initialization

**Purpose**: Establish or validate the epistemic framework for detection.

**Process**:
1. Check if `epistemic_context` was provided
2. If **provided** (from RAG/NonRAG detector):
   - Use the explicit context
   - No warnings generated
3. If **not provided** (direct Unified detector call):
   - Create `UNKNOWN` mode context
   - Emit epistemic ambiguity warning
   - Results flagged as not academically rigorous

**Code**: [unified_detector.py:214-226](../../src/detection/unified_detector.py#L214-L226)

**Output**: `epistemic_context` and optional `epistemic_warnings`

---

### Auto-Retrieval (Optional)

**Purpose**: Retrieve evidence if not provided, based on epistemic mode.

**Process**:
1. **RAG mode**:
   - `force_retrieve=False`
   - ONLY uses provided `retrieved_context`
   - Never auto-retrieves (would violate closed-world assumption)

2. **NonRAG mode**:
   - `force_retrieve=False`
   - Auto-retrieves if `retrieved_context=None`
   - Evidence is POST-HOC approximation of factual knowledge

3. **UNKNOWN mode**:
   - Auto-retrieves if needed
   - Behavior depends on configuration

**Code**: [unified_detector.py:229-262](../../src/detection/unified_detector.py#L229-L262)

**Output**: `retrieved_context` (provided or auto-retrieved)

---

### Layer 1: Structural Verification

**Purpose**: Fast-fail check for obvious citation hallucinations.

**Checks**:
1. **Citation extraction**: Parse legal citations from answer
2. **Context cross-check**: Verify citations exist in evidence (if `require_context=True`)
3. **Format validation**: Check against blocklist patterns, validate reporter abbreviations
4. **Date consistency**: Detect anachronisms

**Fast-fail behavior**:
- If structural verification fails, return immediately with `is_hallucinated=True`
- Skips expensive layers 2-4
- Confidence: 0.95 (high confidence in structural failures)

**Code**: [unified_detector.py:265-290](../../src/detection/unified_detector.py#L265-L290)
**Component**: `StructuralVerifier` ([structural_verifier.py](../../src/detection/structural_verifier.py))

**Output**: `StructuralVerificationResult` (is_valid, failed_citations, details)

---

### Layer 2: Claim Decomposition

**Purpose**: Break down long-form answer into atomic, verifiable claims.

**Process**:
1. **Legal-aware segmentation**: Split into sentences respecting legal syntax
2. **Quality validation**: NLI-based quality gate (completeness + correctness >= 0.8)
3. **Length filtering**: Enforce minimum claim length
4. **Deduplication**: Remove redundant claims

**Early return behavior**:
- If no claims extracted, return with `is_hallucinated=False`
- Confidence: 0.50 (low confidence due to no verifiable content)

**Code**: [unified_detector.py:293-317](../../src/detection/unified_detector.py#L293-L317)
**Component**: `ClaimDecomposer` ([claim_decomposer.py](../../src/preprocessing/claim_decomposer.py))

**Output**: List of `DecomposedClaim` objects

---

### Layer 3: Semantic Grounding (Per-Claim)

**Purpose**: Fast filter using embedding-based similarity.

**Process**:
1. **Encode claim**: Generate embedding using bi-encoder (Legal-BERT)
2. **Encode evidence**: Generate embeddings for all evidence documents
3. **Cosine similarity**: Compare claim embedding to each evidence embedding
4. **Top-K selection**: Retrieve K most similar evidence chunks
5. **Threshold check**: If max similarity < threshold (default 0.50):
   - Mark claim as **failed (semantic)**
   - Skip expensive NLI layer
   - Continue to next claim

**Optimization**: Bi-encoders are fast (~10ms per claim) vs cross-encoders (~100ms per claim)

**Code**: [unified_detector.py:326-352](../../src/detection/unified_detector.py#L326-L352)
**Component**: `SemanticGroundingVerifier` ([semantic_matcher.py](../../src/detection/semantic_matcher.py))

**Output**: `GroundingResult` (similarity_score, is_grounded)

---

### Layer 4: NLI Entailment (Per-Claim)

**Purpose**: Precise verification using Natural Language Inference.

**Process** (only for claims that passed semantic filter):
1. **Multi-source verification**: Test claim against multiple evidence documents
2. **Cross-encoder inference**: Use NLI model (DeBERTa-v3-base) to classify:
   - **ENTAILMENT**: Evidence supports claim
   - **CONTRADICTION**: Evidence contradicts claim
   - **NEUTRAL**: Evidence unrelated to claim
3. **Contradiction check**: If any evidence contradicts:
   - Mark claim as **failed (contradiction)**
   - Hallucination type: "contradiction"
4. **Entailment check**: If confidence < threshold (default 0.70):
   - Mark claim as **failed (entailment)**
   - Hallucination type: "entailment"
5. **Success**: If entailment score >= threshold:
   - Mark claim as **grounded**

**Optimization**: Early stopping if strong entailment found

**Code**: [unified_detector.py:354-411](../../src/detection/unified_detector.py#L354-L411)
**Component**: `EntailmentVerifier` ([entailment_verifier.py](../../src/detection/entailment_verifier.py))

**Output**: Per-claim verification result with label and confidence

---

### Layer 5: Aggregation & Result Construction

**Purpose**: Combine per-claim results into final verdict.

**Process**:
1. **Count failures**: `num_failed = sum(1 for r in claim_results if not r['grounded'])`
2. **Calculate failure rate**: `failure_rate = num_failed / total_claims`
3. **Determine verdict**: `is_hallucinated = (failure_rate > 0.0)`
4. **Calculate risk**: `hallucination_risk = failure_rate`
5. **Determine type**:
   - Priority: contradiction > semantic > entailment
   - Based on failure types found in claim results
6. **Calculate confidence**: Average of entailment scores
7. **Attach context**: Embed epistemic_context and warnings in result

**Code**: [unified_detector.py:420-468](../../src/detection/unified_detector.py#L420-L468)

**Output**: Complete `HallucinationResult` object

---

## Entry Point Selection

### Decision Tree: Which Detector to Use?

```
START: I have a response to evaluate
│
├─ Do I know how it was generated?
│  │
│  ├─ YES: Was retrieval used during generation?
│  │  │
│  │  ├─ YES: Retrieval-augmented generation (RAG)
│  │  │  │
│  │  │  ├─ Do I have the EXACT documents shown to model?
│  │  │  │
│  │  │  ├─ YES → Use RAGHallucinationDetector ✅
│  │  │  │        detector.detect(query, answer, retrieved_context)
│  │  │  │        Epistemic context: RAG, closed-world
│  │  │  │        No warnings
│  │  │  │
│  │  │  └─ NO → Cannot use RAG detector ❌
│  │  │           Must use NonRAG or Unified
│  │  │
│  │  └─ NO: Parametric generation (baseline LLM)
│  │     │
│  │     └─ Use NonRAGHallucinationDetector ✅
│  │        detector.detect(query, answer)
│  │        Auto-retrieves evidence for verification
│  │        Epistemic context: NON_RAG, open-world
│  │        No warnings
│  │
│  └─ NO: Unknown generation method
│     │
│     └─ Use UnifiedHallucinationDetector ⚠️
│        detector.detect(query, answer)
│        Auto-retrieves if needed
│        Epistemic context: UNKNOWN
│        Emits epistemic ambiguity warnings
│        Results not academically rigorous
```

### Detector Comparison Table

| Criterion | RAGDetector | NonRAGDetector | UnifiedDetector |
|-----------|-------------|----------------|-----------------|
| **For** | RAG systems | Baseline models | Unknown/hybrid |
| **retrieved_context** | REQUIRED | Optional | Optional |
| **Auto-retrieval** | Never | If context=None | If context=None |
| **Epistemic mode** | RAG (explicit) | NON_RAG (explicit) | Provided or UNKNOWN |
| **Epistemic warnings** | Never | Never | If mode UNKNOWN |
| **Academic rigor** | ✅ Always | ✅ Always | ⚠️ Only if context provided |
| **Production use** | RAG only | Non-RAG only | Any mode |
| **Research papers** | ✅ Valid | ✅ Valid | ❌ Not valid (UNKNOWN) |

---

## Example Trace

### Scenario: Evaluating a Parametric Response

**Input**:
- Query: "Does the inevitable discovery doctrine create a per se exception to the exclusionary rule?"
- Answer: "A great question in constitutional law! The inevitable discovery doctrine does not automatically create a per se exception..."
- Generation method: **Parametric** (no retrieval)

**Journey**:

```
1. Entry Point: NonRAGHallucinationDetector
   Code: nonrag_detector.py:87
   ↓

2. Create Epistemic Context
   Code: nonrag_detector.py:112
   Context: {
     generation_mode: NON_RAG,
     evidence_scope: "parametric",
     evidence_provided_to_model: False,
     closed_world_assumption: False
   }
   ↓

3. Call UnifiedDetector
   Code: nonrag_detector.py:121
   ↓

4. Layer 0: Epistemic Init
   Code: unified_detector.py:214
   Result: Use provided NON_RAG context, no warnings
   ↓

5. Auto-Retrieval
   Code: unified_detector.py:240
   Query: "Does the inevitable discovery... A great question..."
   Retrieved: 5 documents about inevitable discovery doctrine
   Note: These docs were NOT shown to model during generation
   ↓

6. Layer 1: Structural
   Code: unified_detector.py:268
   Check: Citations in answer
   Result: PASSED ✅
   ↓

7. Layer 2: Decomposition
   Code: unified_detector.py:296
   Extracted claims:
     - Claim 1: "The inevitable discovery doctrine does not automatically create..."
     - Claim 2: "Courts have applied the inevitable discovery doctrine..."
     - Claim 3: "The prosecution can still introduce evidence..."
   ↓

8. Layer 3: Semantic (Claim 1)
   Code: unified_detector.py:329
   Claim: "The inevitable discovery doctrine does not automatically..."
   Similarity vs evidence: 0.72
   Threshold: 0.50
   Result: PASSED ✅ (proceed to NLI)
   ↓

9. Layer 4: NLI (Claim 1)
   Code: unified_detector.py:356
   Claim vs evidence: ENTAILMENT
   Confidence: 0.87
   Threshold: 0.70
   Result: GROUNDED ✅
   ↓

   [Repeat layers 3-4 for Claims 2 and 3]
   ↓

10. Layer 5: Aggregation
    Code: unified_detector.py:422
    Failed claims: 0/3
    Failure rate: 0.0
    Hallucination risk: 0.0
    Confidence: 0.89 (average entailment)
    ↓

11. Return Result
    {
      is_hallucinated: False,
      hallucination_risk: 0.0,
      confidence: 0.89,
      semantic_scores: [0.72, 0.68, 0.75],
      entailment_scores: [0.87, 0.84, 0.91],
      epistemic_context: {mode: NON_RAG, ...},
      epistemic_warnings: []  // No warnings!
    }
```

**Total time**: ~450ms (auto-retrieval: 120ms, structural: 10ms, decomposition: 50ms, semantic: 90ms, NLI: 180ms)

---

## Code References

### Main Entry Points

| File | Class | Method | Line |
|------|-------|--------|------|
| [src/detection/rag_detector.py](../../src/detection/rag_detector.py) | RAGHallucinationDetector | detect() | 76-127 |
| [src/detection/nonrag_detector.py](../../src/detection/nonrag_detector.py) | NonRAGHallucinationDetector | detect() | 87-130 |
| [src/detection/unified_detector.py](../../src/detection/unified_detector.py) | UnifiedHallucinationDetector | detect() | 168-469 |

### Detection Layers

| Layer | Component | File | Key Method |
|-------|-----------|------|------------|
| 0 | Epistemic Context | unified_detector.py | Lines 214-226 |
| - | Auto-Retrieval | unified_detector.py | Lines 229-262 |
| 1 | Structural Verifier | structural_verifier.py | verify() |
| 2 | Claim Decomposer | claim_decomposer.py | decompose() |
| 3 | Semantic Verifier | semantic_matcher.py | verify_claim_grounding() |
| 4 | Entailment Verifier | entailment_verifier.py | verify_claim_against_multiple_sources() |
| 5 | Aggregation | unified_detector.py | Lines 420-468 |

### Data Structures

| Structure | File | Line |
|-----------|------|------|
| HallucinationResult | unified_detector.py | 30-93 |
| EpistemicContext | core/config.py | - |
| GenerationMode | core/config.py | - |
| DecomposedClaim | preprocessing/claim_decomposer.py | 22-30 |
| GroundingResult | detection/semantic_matcher.py | 22-42 |
| EntailmentResult | detection/entailment_verifier.py | 32-59 |

---

## See Also

- [WHY_THREE_PIPELINES.md](WHY_THREE_PIPELINES.md) - Epistemic rationale (aspirational design; NOT implemented)
- [RAG_DETECTION_ARCHITECTURE.md](RAG_DETECTION_ARCHITECTURE.md) - RAG detector details
- [NONRAG_DETECTION_ARCHITECTURE.md](NONRAG_DETECTION_ARCHITECTURE.md) - Non-RAG detector details
- [UNIFIED_DETECTION_ARCHITECTURE.md](UNIFIED_DETECTION_ARCHITECTURE.md) - Unified detector details
- [HALLUCINATION_DETECTION_ARCHITECTURE.md](../../HALLUCINATION_DETECTION_ARCHITECTURE.md) - Overall system

---

**Questions or corrections?** This diagram is verified against the actual code implementation. If you find discrepancies, please open an issue.
