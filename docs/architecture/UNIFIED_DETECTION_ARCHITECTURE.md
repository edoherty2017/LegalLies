# Unified Hallucination Detection Architecture

> ### WARNING - Implementation note (2026-10)
> `UnifiedHallucinationDetector` is a **backwards-compatibility alias** for
> `FactualityDetector` (`src/detection/unified_detector.py`, last line). It is not a
> separate detector and it does **not** track generation mode or emit epistemic
> ambiguity warnings - no such code exists. The `GenerationMode` / `EpistemicContext`
> types this document assumes have been removed as unused.
>
> Statements below about mode tracking, ambiguity warnings, or per-mode behavior
> describe an intended design, **not the shipped code**.

**Purpose**: Claim-level factuality detection against a fixed oracle corpus.

**Epistemic Assumption**: Symmetric - every claim is verified against the oracle
corpus regardless of how the answer was generated.

---

## Overview

`UnifiedHallucinationDetector` is the foundational 5-layer detection infrastructure that powers all three detection pipelines. It can operate in any epistemic mode when provided with explicit context, or default to UNKNOWN mode with warnings.

**Key Features**:
- ✅ 5-layer detection pipeline (structural, decomposition, semantic, NLI, aggregation)
- ✅ Explicit epistemic context tracking
- ✅ Automatic warnings for ambiguous modes
- ✅ Dynamic evidence retrieval
- ✅ Backward compatible with legacy code
- ✅ Production-ready for unknown/mixed modes

---

## When to Use

**Use Unified Detector for**:
- ✅ Production systems with mixed generation modes
- ✅ Systems where generation mode is determined at runtime
- ✅ Custom epistemic assumptions not covered by RAG/Non-RAG
- ✅ Hybrid approaches (RAG + parametric knowledge)
- ✅ Legacy integration where mode tracking isn't available

**Use Specific Detectors for**:
- ❌ Pure RAG evaluation → Use [RAGHallucinationDetector](RAG_DETECTION_ARCHITECTURE.md)
- ❌ Pure parametric evaluation → Use [NonRAGHallucinationDetector](NONRAG_DETECTION_ARCHITECTURE.md)
- ❌ Academic research requiring epistemic clarity → Use specific detectors

**Rule of thumb**: If you know the generation mode upfront, use the specific detector. If mode is unknown/dynamic, use Unified with explicit context.

---

## Quick Start

### Installation

```python
from src.detection import UnifiedHallucinationDetector
from src.core.config import EpistemicContext, GenerationMode
```

### Basic Usage (Unknown Mode)

```python
# Initialize detector
detector = UnifiedHallucinationDetector()

# Detect without epistemic context (production mode)
result = detector.detect(
    query="What is qualified immunity?",
    generated_answer="QI protects officers under 42 U.S.C. § 1983...",
    retrieved_context=passages  # Optional
)

# Check for epistemic warnings
if result.epistemic_warnings:
    print(f"Warning: {result.epistemic_warnings[0]}")
    # "EPISTEMIC_AMBIGUITY: Generation mode not specified..."

# Result is still usable, but not academically rigorous
print(f"Hallucinated: {result.is_hallucinated}")
```

### Advanced Usage (Explicit Context)

```python
# Create explicit epistemic context
epistemic_ctx = EpistemicContext(
    generation_mode=GenerationMode.RAG,
    evidence_scope="retrieved",
    evidence_provided_to_model=True,
    closed_world_assumption=True,
)

# Detect with explicit context (academically rigorous)
result = detector.detect(
    query="What is qualified immunity?",
    generated_answer="QI protects officers...",
    retrieved_context=passages,
    epistemic_context=epistemic_ctx  # No warnings!
)

# Verify epistemic tracking
assert result.epistemic_context.generation_mode == GenerationMode.RAG
assert len(result.epistemic_warnings) == 0
```

---

## Architecture

### 5-Layer Detection Pipeline

```
┌─────────────────────────────────────────────────────────┐
│         UnifiedHallucinationDetector                    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │ Layer 0: Epistemic Context Initialization      │    │
│  │  - Create/validate epistemic context           │    │
│  │  - Generate warnings for UNKNOWN mode          │    │
│  └────────────────────────────────────────────────┘    │
│                        │                                 │
│  ┌────────────────────▼────────────────────────────┐   │
│  │ Layer 1: Structural Verification                │   │
│  │  - Citation format validation                   │   │
│  │  - Citation grounding (if require_context=True) │   │
│  │  - Fast-fail on structural errors               │   │
│  └────────────────────┬────────────────────────────┘   │
│                        │                                 │
│  ┌────────────────────▼────────────────────────────┐   │
│  │ Layer 2: Claim Decomposition                    │   │
│  │  - Legal-aware sentence splitting               │   │
│  │  - Quality filtering (threshold=0.8)            │   │
│  │  - Minimum length enforcement                   │   │
│  └────────────────────┬────────────────────────────┘   │
│                        │                                 │
│  ┌────────────────────▼────────────────────────────┐   │
│  │ Layer 3: Semantic Grounding (Bi-Encoder)        │   │
│  │  - Fast semantic similarity filter              │   │
│  │  - Top-K evidence retrieval per claim           │   │
│  │  - Early rejection if similarity < threshold    │   │
│  └────────────────────┬────────────────────────────┘   │
│                        │                                 │
│  ┌────────────────────▼────────────────────────────┐   │
│  │ Layer 4: NLI Entailment (Cross-Encoder)         │   │
│  │  - Precise entailment verification              │   │
│  │  - Contradiction detection                      │   │
│  │  - Multi-source aggregation                     │   │
│  └────────────────────┬────────────────────────────┘   │
│                        │                                 │
│  ┌────────────────────▼────────────────────────────┐   │
│  │ Layer 5: Aggregation + Epistemic Tracking       │   │
│  │  - Combine per-claim results                    │   │
│  │  - Calculate hallucination risk                 │   │
│  │  - Attach epistemic context to result           │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Epistemic Modes

The detector operates in one of three epistemic modes:

**1. RAG Mode** (`GenerationMode.RAG`):
```python
epistemic_ctx = EpistemicContext(
    generation_mode=GenerationMode.RAG,
    evidence_scope="retrieved",
    evidence_provided_to_model=True,
    closed_world_assumption=True,
)
```
- Evidence: Only retrieved documents
- Hallucination: Grounding failure
- Verification: Strict (model should not use parametric knowledge)

**2. Non-RAG Mode** (`GenerationMode.NON_RAG`):
```python
epistemic_ctx = EpistemicContext(
    generation_mode=GenerationMode.NON_RAG,
    evidence_scope="parametric",
    evidence_provided_to_model=False,
    closed_world_assumption=False,
)
```
- Evidence: Model's parametric knowledge
- Hallucination: Parametric failure
- Verification: Auto-retrieval for post-hoc approximation

**3. Unknown Mode** (`GenerationMode.UNKNOWN`):
```python
# Created automatically when epistemic_context=None
epistemic_ctx = EpistemicContext(
    generation_mode=GenerationMode.UNKNOWN,
    evidence_scope="unknown",
    evidence_provided_to_model=False,
    closed_world_assumption=False,
)
# WARNING: Results may not be epistemically valid!
```
- Evidence: Ambiguous
- Hallucination: Undefined semantics
- Verification: Best-effort (not academically rigorous)

---

## Detection Process

### Step-by-Step Flow

```python
def detect(query, answer, context=None, epistemic_context=None):
    # 1. Initialize epistemic context
    if epistemic_context is None:
        epistemic_context = EpistemicContext(GenerationMode.UNKNOWN, ...)
        warnings.append("EPISTEMIC_AMBIGUITY: ...")

    # 2. Auto-retrieval (if needed)
    if not context and retriever_enabled:
        context = retriever.retrieve(query + answer)

    # 3. Structural verification (fast-fail)
    if structural_verifier.verify(answer, context).is_invalid:
        return HallucinationResult(
            is_hallucinated=True,
            hallucination_type="structural",
            epistemic_context=epistemic_context,
            epistemic_warnings=warnings,
        )

    # 4. Claim decomposition
    claims = decomposer.decompose(answer)

    if not claims:
        return HallucinationResult(is_hallucinated=False, ...)

    # 5. Per-claim verification
    for claim in claims:
        # 5a. Semantic grounding (fast filter)
        semantic_score = semantic_verifier.verify(claim, context)
        if semantic_score < threshold:
            claim.grounded = False
            continue

        # 5b. NLI entailment (expensive but accurate)
        entailment = nli_verifier.verify(claim, context)
        if entailment == "CONTRADICTION":
            claim.grounded = False
        elif entailment != "ENTAILMENT":
            claim.grounded = False
        else:
            claim.grounded = True

    # 6. Aggregation
    failure_rate = count_ungrounded(claims) / len(claims)

    return HallucinationResult(
        is_hallucinated=(failure_rate > 0),
        hallucination_risk=failure_rate,
        epistemic_context=epistemic_context,
        epistemic_warnings=warnings,
    )
```

---

## Configuration

### Default Configuration

```python
from src.core.config import DetectorConfig

config = DetectorConfig(
    # NLI thresholds
    nli_entailment_threshold=0.70,
    nli_contradiction_threshold=0.70,

    # Semantic thresholds
    semantic_similarity_threshold=0.50,
    bi_encoder_top_k=3,

    # Citation verification
    citation_require_context=True,
    citation_fuzzy_match_threshold=0.85,

    # Retrieval (for auto-retrieval)
    retrieval_enabled=True,
    retrieval_top_k=5,

    # Models
    bi_encoder_model="nlpaueb/legal-bert-base-uncased",
    cross_encoder_model="microsoft/deberta-v3-large",
)

detector = UnifiedHallucinationDetector(config=config)
```

### Custom Configuration

```python
# Strict configuration for high-precision
strict_config = DetectorConfig(
    nli_entailment_threshold=0.80,  # Higher bar
    semantic_similarity_threshold=0.60,
    citation_require_context=True,
)

# Lenient configuration for high-recall
lenient_config = DetectorConfig(
    nli_entailment_threshold=0.60,  # Lower bar
    semantic_similarity_threshold=0.40,
    citation_require_context=False,
)
```

---

## Integration Examples

### Example 1: Production System with Runtime Mode

```python
class ProductionRAGSystem:
    def __init__(self):
        self.detector = UnifiedHallucinationDetector()
        self.generator = MyGenerator()
        self.retriever = MyRetriever()

    def generate_and_detect(self, query, use_rag=True):
        if use_rag:
            # RAG mode
            passages = self.retriever.retrieve(query)
            answer = self.generator.generate(query, context=passages)

            epistemic_ctx = EpistemicContext(
                generation_mode=GenerationMode.RAG,
                evidence_scope="retrieved",
                evidence_provided_to_model=True,
                closed_world_assumption=True,
            )

            result = self.detector.detect(
                query, answer, passages, epistemic_context=epistemic_ctx
            )
        else:
            # Non-RAG mode
            answer = self.generator.generate(query)

            epistemic_ctx = EpistemicContext(
                generation_mode=GenerationMode.NON_RAG,
                evidence_scope="parametric",
                evidence_provided_to_model=False,
                closed_world_assumption=False,
            )

            result = self.detector.detect(
                query, answer, epistemic_context=epistemic_ctx
            )

        return {
            "answer": answer,
            "is_safe": not result.is_hallucinated,
            "risk": result.hallucination_risk,
            "mode": result.epistemic_context.generation_mode.value,
        }
```

### Example 2: Hybrid RAG + Parametric

```python
# Custom epistemic context for hybrid mode
hybrid_ctx = EpistemicContext(
    generation_mode=GenerationMode.RAG,  # Treat as RAG for grounding
    evidence_scope="hybrid",  # Custom scope
    evidence_provided_to_model=True,
    closed_world_assumption=False,  # Allow parametric knowledge
)

result = detector.detect(
    query, hybrid_answer, retrieved_passages, epistemic_context=hybrid_ctx
)

# Result tracks hybrid assumptions
assert result.epistemic_context.evidence_scope == "hybrid"
assert result.epistemic_context.closed_world_assumption == False
```

### Example 3: Legacy Integration (No Mode Tracking)

```python
# Legacy system without mode tracking
legacy_answer = legacy_system.generate(query)

# Use unified detector without context
result = detector.detect(query, legacy_answer)

# Check warnings
if result.epistemic_warnings:
    logger.warning(f"Epistemic ambiguity: {result.epistemic_warnings}")

# Result still usable, but flagged as ambiguous
return {
    "answer": legacy_answer,
    "hallucination_detected": result.is_hallucinated,
    "epistemic_mode": "unknown",  # Flag as uncertain
}
```

---

## Result Interpretation

### Success Case with Explicit Context

```python
result = detector.detect(query, answer, context, epistemic_context=rag_ctx)

if not result.is_hallucinated:
    print("✓ All claims grounded")
    print(f"  Mode: {result.epistemic_context.generation_mode.value}")
    print(f"  Evidence scope: {result.epistemic_context.evidence_scope}")
    print(f"  Warnings: {len(result.epistemic_warnings)}")  # 0
```

### Failure Case with Ambiguity

```python
result = detector.detect(query, answer)  # No context!

if result.epistemic_warnings:
    print("⚠ Epistemic warnings:")
    for warning in result.epistemic_warnings:
        print(f"  - {warning}")

if result.is_hallucinated:
    print(f"✗ Hallucination detected (mode={result.epistemic_context.generation_mode.value})")
    print(f"  Risk: {result.hallucination_risk:.2%}")
    print(f"  Note: Results may not be epistemically valid")
```

### Exporting Results

```python
# Export to dict (includes epistemic tracking)
result_dict = result.to_dict()

assert "epistemic_context" in result_dict
assert result_dict["epistemic_context"]["generation_mode"] == "rag"
assert result_dict["epistemic_warnings"] == []

# Save to JSON
import json
with open("detection_result.json", "w") as f:
    json.dump(result_dict, f, indent=2)
```

---

## Comparison with Specific Detectors

| Feature | UnifiedDetector | RAGDetector | NonRAGDetector |
|---------|-----------------|-------------|----------------|
| **Epistemic tracking** | ✅ Explicit | ✅ Automatic (RAG) | ✅ Automatic (Non-RAG) |
| **Mode flexibility** | ✅ All modes | ❌ RAG only | ❌ Non-RAG only |
| **Academic rigor** | ⚠️ If context provided | ✅ Always | ✅ Always |
| **Production use** | ✅ Best for mixed | ❌ RAG only | ❌ Non-RAG only |
| **Ambiguity warnings** | ✅ Yes | ❌ Not needed | ❌ Not needed |
| **API complexity** | ⚠️ Medium (optional params) | ✅ Simple (enforced) | ✅ Simple (enforced) |

**Recommendation**:
- **Research**: Use specific detectors (RAGDetector / NonRAGDetector)
- **Production**: Use UnifiedDetector with explicit context
- **Legacy**: Use UnifiedDetector without context (accept warnings)

---

## Best Practices

### DO ✅

1. **Always provide epistemic_context when possible**:
   ```python
   result = detector.detect(q, a, ctx, epistemic_context=rag_ctx)
   ```

2. **Check epistemic_warnings**:
   ```python
   if result.epistemic_warnings:
       logger.warning("Epistemic ambiguity detected")
   ```

3. **Use specific detectors for pure modes**:
   ```python
   # If you know it's RAG, use RAGDetector
   rag_detector = RAGHallucinationDetector()
   ```

4. **Document epistemic assumptions**:
   ```python
   # In production code
   # ASSUMPTION: Treats hybrid as RAG with open-world
   result = detector.detect(..., epistemic_context=hybrid_ctx)
   ```

### DON'T ❌

1. **Don't ignore epistemic warnings**:
   ```python
   # BAD: Silently accept ambiguous results
   result = detector.detect(q, a)  # No context!
   return result.is_hallucinated  # Epistemically invalid!
   ```

2. **Don't use for academic research without context**:
   ```python
   # BAD: Research paper with UNKNOWN mode
   results = [detector.detect(q, a) for q, a in test_set]
   # Reviewers will reject!
   ```

3. **Don't mix modes in comparison**:
   ```python
   # BAD: Comparing different epistemic assumptions
   rag_results = [detector.detect(q, a, ctx, rag_ctx) for ...]
   nonrag_results = [detector.detect(q, a) for ...]  # No context!
   # Comparison is epistemically invalid!
   ```

4. **Don't create custom modes without documentation**:
   ```python
   # BAD: Unclear custom mode
   weird_ctx = EpistemicContext(GenerationMode.RAG, "magic", ...)
   # What does "magic" evidence_scope mean?
   ```

---

## Limitations

### 1. UNKNOWN Mode Limitations

**Issue**: Without epistemic context, results have unclear semantics.

**Example**:
```python
result = detector.detect(query, answer)  # Mode unknown
# What does "hallucinated" mean here?
# - Grounding failure?
# - Parametric failure?
# - Ambiguous!
```

**Mitigation**: Always provide epistemic_context in production.

### 2. Flexibility vs. Enforcement

**Issue**: Flexible API allows epistemic errors.

**Example**:
```python
# Nothing prevents this mistake:
result = detector.detect(query, rag_answer, epistemic_context=nonrag_ctx)
# Labeled as non-RAG but actually from RAG!
```

**Mitigation**: Use specific detectors (RAGDetector/NonRAGDetector) which enforce correctness.

### 3. Legacy Integration Challenges

**Issue**: Legacy systems may not track generation mode.

**Example**:
```python
# Legacy system
answer = black_box_generator(query)  # Unknown mode!
result = detector.detect(query, answer)  # Best-effort
```

**Mitigation**: Document assumptions and flag results as "epistemic_unknown".

---

## See Also

- [Why Three Pipelines?](WHY_THREE_PIPELINES.md) - Epistemic rationale (aspirational; NOT implemented)
- [RAG Detection Architecture](RAG_DETECTION_ARCHITECTURE.md) - RAG-specific detector
- [Non-RAG Detection Architecture](NONRAG_DETECTION_ARCHITECTURE.md) - Non-RAG detector
- [Overall System Architecture](../../HALLUCINATION_DETECTION_ARCHITECTURE.md) - Full system

---

**Questions?** See docs/ or open an issue.
