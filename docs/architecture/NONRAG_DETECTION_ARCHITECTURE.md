# Non-RAG Hallucination Detection Architecture

**Purpose**: Detect parametric failures in pure generation without retrieval.

**Epistemic Assumption**: Open-world - model uses parametric knowledge from training.

---

## Overview

`NonRAGHallucinationDetector` is a specialized detector for baseline/parametric generation where models generate from internal knowledge without retrieval. It uses post-hoc evidence retrieval to approximate what the model "should" know, acknowledging that this evidence was NOT shown to the model during generation.

**Key Features**:
- ✅ Open-world assumption (model uses parametric knowledge)
- ✅ Automatic post-hoc evidence retrieval for verification
- ✅ Epistemic context tracking
- ✅ Academically rigorous for baseline evaluation
- ✅ Drop-in integration for parametric generation

---

## When to Use

**Use Non-RAG Detector for**:
- ✅ Baseline/parametric generation (no retrieval)
- ✅ Pure LLM outputs (GPT-4, Claude, Llama without RAG)
- ✅ Comparing baseline models against each other
- ✅ Evaluating model parametric knowledge
- ✅ Academic papers on non-RAG hallucination rates

**Do NOT use for**:
- ❌ Retrieval-augmented generation (use [RAGHallucinationDetector](RAG_DETECTION_ARCHITECTURE.md))
- ❌ Systems where evidence was shown during generation
- ❌ Unknown generation mode (use [UnifiedHallucinationDetector](UNIFIED_DETECTION_ARCHITECTURE.md))

---

## Quick Start

### Installation

```python
from src.detection import NonRAGHallucinationDetector
```

### Basic Usage

```python
# Initialize detector
detector = NonRAGHallucinationDetector()

# Parametric generation (no retrieval)
query = "What is qualified immunity?"
answer = baseline_model.generate(query)  # No context provided

# Detect hallucinations
result = detector.detect(
    query=query,
    generated_answer=answer,
    # No retrieved_context - detector auto-retrieves for verification
)

# Check results
if result.is_hallucinated:
    print(f"Hallucination detected: {result.failure_reason}")
    print(f"Risk: {result.hallucination_risk:.2%}")
    print(f"Failed claims: {sum(1 for c in result.claim_results if not c['grounded'])}")
else:
    print(f"✓ All claims verified against factual knowledge")

# Verify epistemic context
assert result.epistemic_context.generation_mode == GenerationMode.NON_RAG
assert result.epistemic_context.evidence_provided_to_model == False
```

---

## Architecture

### Design Pattern

`NonRAGHallucinationDetector` is a **wrapper** around `UnifiedHallucinationDetector` that enforces Non-RAG-specific epistemic assumptions:

```
┌─────────────────────────────────────────┐
│   NonRAGHallucinationDetector           │
│  (Wrapper with Non-RAG assumptions)     │
└───────────────┬─────────────────────────┘
                │
                │ Creates EpistemicContext:
                │  - generation_mode = NON_RAG
                │  - closed_world = False
                │  - evidence_provided = False
                │
                ▼
┌─────────────────────────────────────────┐
│   UnifiedHallucinationDetector          │
│  (5-layer detection infrastructure)     │
│                                          │
│  1. Structural Verification              │
│  2. Claim Decomposition                  │
│  3. Semantic Grounding (auto-retrieval)  │
│  4. NLI Entailment                       │
│  5. Aggregation + Epistemic Tracking     │
└─────────────────────────────────────────┘
```

### Epistemic Context

Every detection automatically creates:

```python
EpistemicContext(
    generation_mode=GenerationMode.NON_RAG,
    evidence_scope="parametric",  # Model's internal knowledge
    evidence_provided_to_model=False,  # No evidence shown during generation
    closed_world_assumption=False  # Model uses parametric knowledge
)
```

This context is embedded in every `HallucinationResult` for full traceability.

---

## Detection Process

### Step 1: Validation

```python
# No validation required - works with or without context
# If context is provided, uses it; otherwise auto-retrieves
```

**Rationale**: Non-RAG detection is lenient. The model generated from parametric knowledge, so we use post-hoc retrieval to approximate factual correctness.

### Step 2: Epistemic Context Creation

```python
epistemic_context = EpistemicContext(
    generation_mode=GenerationMode.NON_RAG,
    evidence_scope="parametric",
    evidence_provided_to_model=False,
    closed_world_assumption=False,
)
```

### Step 3: Auto-Retrieval (if needed)

```python
if not retrieved_context and retrieval_enabled:
    # Auto-retrieve evidence for verification
    retrieved_context = retriever.retrieve(
        query=query,
        answer=generated_answer,
        top_k=5
    )
```

**Key difference from RAG**: This evidence is retrieved **after** generation for verification purposes only. The model **never saw this evidence** during generation.

### Step 4: Detection via Unified Infrastructure

```python
result = self.unified_detector.detect(
    query=query,
    generated_answer=generated_answer,
    retrieved_context=retrieved_context,  # Post-hoc retrieval
    force_retrieve=True,  # Enable auto-retrieval
    epistemic_context=epistemic_context,
)
```

**Key**: `force_retrieve=True` enables automatic evidence retrieval when context is not provided.

### Step 5: Return with Epistemic Tracking

```python
return HallucinationResult(
    is_hallucinated=True/False,
    hallucination_risk=0.0-1.0,
    epistemic_context=epistemic_context,
    epistemic_warnings=[],  # No warnings for explicit Non-RAG mode
    ...
)
```

---

## Hallucination Types in Non-RAG

### 1. Parametric Fabrication

**Definition**: Claim contradicts factual knowledge (detected via post-hoc retrieval).

**Example**:
```
Query: "What is the exclusionary rule?"
Answer: "The exclusionary rule allows illegally obtained evidence in criminal trials."
```

**Detection**:
- Auto-retrieve: "The exclusionary rule prevents illegally obtained evidence..."
- ❌ NLI label: CONTRADICTION (0.89)
- **Verdict**: Hallucination (parametric failure)

### 2. Parametric Uncertainty

**Definition**: Claim is not verifiable against retrieved factual knowledge.

**Example**:
```
Query: "What is qualified immunity?"
Answer: "Qualified immunity was established in Smith v. Unknown, 1975."
```

**Detection**:
- Auto-retrieve: [Real cases about qualified immunity]
- ❌ Semantic similarity: LOW (0.28) - "Smith v. Unknown" not found
- **Verdict**: Hallucination (semantic failure)

### 3. Common Knowledge (Pass)

**Definition**: Claim uses common knowledge that's verifiable.

**Example**:
```
Query: "What is murder?"
Answer: "Murder is the unlawful killing of another human being with malice aforethought."
```

**Detection**:
- Auto-retrieve: Legal definitions of murder
- ✅ Semantic similarity: HIGH (0.91)
- ✅ NLI entailment: HIGH (0.87)
- **Verdict**: Not hallucinated (common knowledge verified)

### 4. Citation Hallucination (Lenient)

**Definition**: Answer cites a case, but we cannot verify it in auto-retrieved evidence.

**Example**:
```
Answer: "As established in Terry v. Ohio (1968)..."
```

**Detection**:
- ❌ Structural verification: "Terry v. Ohio" not in auto-retrieved evidence
- ⚠️ **Verdict**: Hallucination IF `citation_require_context=True`, otherwise passes

**Note**: For Non-RAG, citation verification is more lenient since the model didn't have explicit evidence. It may cite real cases from parametric knowledge that we fail to retrieve.

---

## Configuration

### Thresholds

Non-RAG detection uses the same thresholds as the unified detector:

```python
from src.core.config import DetectorConfig

config = DetectorConfig(
    # NLI thresholds
    nli_entailment_threshold=0.70,  # Min score for entailment
    nli_contradiction_threshold=0.70,  # Min score for contradiction

    # Semantic thresholds
    semantic_similarity_threshold=0.50,  # Pre-filter before NLI

    # Citation thresholds (MORE LENIENT for Non-RAG)
    citation_require_context=False,  # Don't require citations in auto-retrieved evidence
    citation_fuzzy_match_threshold=0.85,

    # Auto-retrieval settings
    retrieval_enabled=True,
    retrieval_top_k=5,
)

detector = NonRAGHallucinationDetector(config=config)
```

### Recommended Settings for Non-RAG

**Strict (academic evaluation)**:
```python
config = DetectorConfig(
    nli_entailment_threshold=0.75,  # High bar for factual correctness
    semantic_similarity_threshold=0.60,  # Stricter pre-filter
    citation_require_context=False,  # Lenient - model cites from parametric knowledge
    retrieval_top_k=10,  # More evidence for verification
)
```

**Lenient (production with recall focus)**:
```python
config = DetectorConfig(
    nli_entailment_threshold=0.60,  # More permissive
    semantic_similarity_threshold=0.40,  # Catch more potential issues
    citation_require_context=False,  # Allow parametric citations
    retrieval_top_k=5,  # Standard retrieval
)
```

---

## Batch Processing

For evaluation on test sets:

```python
detector = NonRAGHallucinationDetector()

# Batch detection
results = detector.detect_batch(
    queries=test_queries,
    answers=baseline_answers,
    # No contexts needed - auto-retrieval handles it
)

# Compute metrics
hallucination_rate = sum(r.is_hallucinated for r in results) / len(results)
avg_confidence = sum(r.confidence for r in results) / len(results)

print(f"Hallucination rate: {hallucination_rate:.2%}")
print(f"Average confidence: {avg_confidence:.2f}")
```

---

## Integration Examples

### With Baseline LLM

```python
class BaselineSystem:
    def __init__(self):
        self.generator = MyLLM()
        self.hallucination_detector = NonRAGHallucinationDetector()

    def generate_with_detection(self, query):
        # Parametric generation (no retrieval)
        answer = self.generator.generate(query)

        # Hallucination detection
        detection_result = self.hallucination_detector.detect(
            query=query,
            generated_answer=answer,
            # No context - detector auto-retrieves for verification
        )

        return {
            "answer": answer,
            "is_safe": not detection_result.is_hallucinated,
            "hallucination_risk": detection_result.hallucination_risk,
            "epistemic_mode": detection_result.epistemic_context.generation_mode.value,
        }
```

### With Custom Retrieval for Verification

```python
# Provide your own evidence for verification
query = "What is qualified immunity?"
answer = baseline_model.generate(query)

# Retrieve verification evidence from trusted source
verification_evidence = trusted_retriever.retrieve(query, top_k=10)

# Detect using custom evidence
result = nonrag_detector.detect(
    query=query,
    generated_answer=answer,
    retrieved_context=verification_evidence  # Custom verification evidence
)

# This evidence is still POST-HOC (not shown to model)
assert result.epistemic_context.evidence_provided_to_model == False
```

---

## Result Interpretation

### Success Case

```python
result = detector.detect(query, answer)

if not result.is_hallucinated:
    print("✓ All claims verified")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Claims verified: {len(result.claims)}")
    print(f"  Evidence scope: {result.epistemic_context.evidence_scope}")  # "parametric"
```

### Failure Case

```python
if result.is_hallucinated:
    print(f"✗ Hallucination detected: {result.hallucination_type}")
    print(f"  Reason: {result.failure_reason}")
    print(f"  Risk: {result.hallucination_risk:.2%}")

    # Per-claim analysis
    for claim_result in result.claim_results:
        if not claim_result['grounded']:
            print(f"  - Failed claim: {claim_result['claim']}")
            print(f"    Failure type: {claim_result['failure']}")
            print(f"    Semantic score: {claim_result.get('semantic_score', 'N/A'):.2f}")
```

---

## Academic Use

### Evaluation Methodology

```python
# Evaluate baseline model on benchmark
nonrag_detector = NonRAGHallucinationDetector()

results = []
for item in benchmark:
    # Generate without retrieval (parametric)
    answer = baseline_model.generate(item['query'])

    # Detect (auto-retrieves for verification)
    detection = nonrag_detector.detect(
        query=item['query'],
        generated_answer=answer,
    )

    results.append({
        "query": item['query'],
        "is_hallucinated": detection.is_hallucinated,
        "hallucination_risk": detection.hallucination_risk,
        "epistemic_mode": detection.epistemic_context.generation_mode.value,
    })

# Report metrics
pd.DataFrame(results).to_csv("nonrag_evaluation.csv")
```

### Comparing Baseline Models

```python
# Model A vs Model B
model_a_results = [nonrag_detector.detect(q, a_ans) for ...]
model_b_results = [nonrag_detector.detect(q, b_ans) for ...]

# Both use same epistemic assumptions (Non-RAG open-world)
# Comparison is valid
print(f"Model A hallucination rate: {sum(r.is_hallucinated for r in model_a_results) / len(model_a_results):.2%}")
print(f"Model B hallucination rate: {sum(r.is_hallucinated for r in model_b_results) / len(model_b_results):.2%}")
```

---

## Limitations

### 1. Post-Hoc Retrieval Approximation

**Limitation**: Auto-retrieved evidence may not match model's actual parametric knowledge.

**Impact**:
- False positives: Model knows fact, but we fail to retrieve it
- False negatives: Model doesn't know fact, but we retrieve supporting evidence

**Example**:
```
Model generates: "Qualified immunity was established in Pierson v. Ray (1967)."
Auto-retrieval finds: "Harlow v. Fitzgerald (1982) established modern QI."
Detection: Hallucination (but Pierson v. Ray is actually correct!)
```

**Mitigation**:
- Use comprehensive retrieval corpus
- Increase retrieval_top_k
- Use multiple retrieval sources

### 2. Common Knowledge Boundary

**Limitation**: Unclear what counts as "parametric knowledge" vs "hallucination".

**Example**:
```
Claim: "Murder is illegal in all U.S. states."
Is this parametric knowledge or should we verify it?
```

**Mitigation**: This is intentional ambiguity. Non-RAG detector is lenient and allows common knowledge claims that can be verified via post-hoc retrieval.

### 3. Citation Verification Ambiguity

**Limitation**: Model may cite real cases from parametric knowledge that we fail to retrieve.

**Example**:
```
Answer: "As held in Miranda v. Arizona (1966)..."
Auto-retrieval: [Documents that don't mention Miranda]
Detection: Citation hallucination (false positive!)
```

**Mitigation**: Set `citation_require_context=False` for Non-RAG to be lenient with citations.

### 4. Retrieval Quality Dependency

**Limitation**: Detection quality depends on retrieval quality.

**Impact**: Poor retrieval → poor hallucination detection

**Mitigation**:
- Use high-quality retrieval index
- Ensure comprehensive corpus coverage
- Monitor retrieval quality separately

---

## Key Differences from RAG Detector

| Dimension | Non-RAG Detector | RAG Detector |
|-----------|------------------|--------------|
| **Evidence source** | Post-hoc auto-retrieval | Exact generation-time evidence |
| **Evidence requirement** | Optional (auto-retrieves) | REQUIRED (must provide) |
| **Closed-world assumption** | ❌ False (open-world) | ✅ True (closed-world) |
| **Evidence shown to model** | ❌ No (post-hoc) | ✅ Yes (generation-time) |
| **Citation verification** | Lenient (`require_context=False`) | Strict (`require_context=True`) |
| **Hallucination semantics** | "Contradicts factual knowledge" | "Not grounded in evidence" |
| **Use case** | Baseline/parametric generation | RAG systems |

**Key insight**: Non-RAG detection uses **approximate verification** (post-hoc retrieval) while RAG detection uses **exact verification** (generation-time evidence). These are **epistemically different**.

---

## See Also

- [Why Three Pipelines?](WHY_THREE_PIPELINES.md) - Rationale for separate detectors (aspirational; NOT implemented)
- [RAG Detection Architecture](RAG_DETECTION_ARCHITECTURE.md) - For RAG systems
- [Unified Detection Architecture](UNIFIED_DETECTION_ARCHITECTURE.md) - For flexible/hybrid use
- [Overall Architecture](../../HALLUCINATION_DETECTION_ARCHITECTURE.md) - Full system documentation

---

**Questions?** See docs/ or open an issue.
