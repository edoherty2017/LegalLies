# RAG Hallucination Detection Architecture

> ### WARNING - Implementation note (2026-10)
> **Code examples below do not run as written.** They reference
> `result.epistemic_context`, `GenerationMode` and `EpistemicContext`, none of which
> exist: `FactualityResult` has no `epistemic_context` attribute, and the two config
> types were never wired into the detector and have been removed.
>
> What ships: `RAGHallucinationDetector` is a thin wrapper that passes your
> `retrieved_context` to `FactualityDetector` and returns a `FactualityResult`. It
> applies no closed-world logic - detection is symmetric across generation modes.
> See the README section "One Detector, Two Convenience Wrappers".

**Purpose**: Verify RAG output claims against the evidence supplied by the caller.

**Epistemic Assumption**: Symmetric - claims are checked against the oracle corpus
on the same terms as every other mode. (The closed-world framing below was a design
intent, never implemented.)

---

## Overview

`RAGHallucinationDetector` is a specialized detector for RAG systems where models receive explicit evidence documents during generation. It enforces strict grounding requirements: every claim must be supported by the retrieved evidence shown to the model.

**Key Features**:
- ✅ Closed-world assumption (no parametric knowledge allowed)
- ✅ Requires exact evidence documents from generation
- ✅ Automatic epistemic context tracking
- ✅ Academically rigorous for RAG evaluation
- ✅ Drop-in integration with RAG pipelines

---

## When to Use

**Use RAG Detector for**:
- ✅ Retrieval-augmented generation systems
- ✅ Systems where evidence is shown to model during generation
- ✅ Comparing different RAG approaches (retrieval strategies, reranking, etc.)
- ✅ Evaluating citation-guided generation
- ✅ Academic papers on RAG hallucination rates

**Do NOT use for**:
- ❌ Baseline/parametric generation (use [NonRAGHallucinationDetector](NONRAG_DETECTION_ARCHITECTURE.md))
- ❌ Unknown generation mode (use [UnifiedHallucinationDetector](UNIFIED_DETECTION_ARCHITECTURE.md))
- ❌ Systems where retrieval context is unavailable

---

## Quick Start

### Installation

```python
from src.detection import RAGHallucinationDetector
```

### Basic Usage

```python
# Initialize detector
detector = RAGHallucinationDetector()

# RAG generation pipeline
query = "What is qualified immunity?"
passages = retriever.retrieve(query, top_k=5)  # Retrieve evidence
answer = rag_model.generate(query, context=passages)  # Generate with evidence

# Detect hallucinations
result = detector.detect(
    query=query,
    generated_answer=answer,
    retrieved_context=passages  # MUST provide exact evidence shown to model
)

# Check results
if result.is_hallucinated:
    print(f"Hallucination detected: {result.failure_reason}")
    print(f"Risk: {result.hallucination_risk:.2%}")
    print(f"Failed claims: {sum(1 for c in result.claim_results if not c['grounded'])}")
else:
    print(f"✓ All claims grounded in evidence")

# Verify epistemic context
assert result.epistemic_context.generation_mode == GenerationMode.RAG
assert result.epistemic_context.closed_world_assumption == True
```

---

## Architecture

### Design Pattern

`RAGHallucinationDetector` is a **wrapper** around `UnifiedHallucinationDetector` that enforces RAG-specific epistemic assumptions:

```
┌─────────────────────────────────────────┐
│     RAGHallucinationDetector            │
│  (Wrapper with RAG assumptions)         │
└───────────────┬─────────────────────────┘
                │
                │ Creates EpistemicContext:
                │  - generation_mode = RAG
                │  - closed_world = True
                │  - evidence_provided = True
                │
                ▼
┌─────────────────────────────────────────┐
│   UnifiedHallucinationDetector          │
│  (5-layer detection infrastructure)     │
│                                          │
│  1. Structural Verification              │
│  2. Claim Decomposition                  │
│  3. Semantic Grounding                   │
│  4. NLI Entailment                       │
│  5. Aggregation + Epistemic Tracking     │
└─────────────────────────────────────────┘
```

### Epistemic Context

Every detection automatically creates:

```python
EpistemicContext(
    generation_mode=GenerationMode.RAG,
    evidence_scope="retrieved",  # Only retrieved documents
    evidence_provided_to_model=True,  # Evidence was shown during generation
    closed_world_assumption=True  # Model should NOT use parametric knowledge
)
```

This context is embedded in every `HallucinationResult` for full traceability.

---

## Detection Process

### Step 1: Validation

```python
if not retrieved_context:
    raise ValueError(
        "RAGHallucinationDetector requires retrieved_context. "
        "If you don't have evidence, use NonRAGHallucinationDetector."
    )
```

**Rationale**: RAG detection is meaningless without the evidence shown to the model. If context is unavailable, you're not evaluating RAG - you're evaluating parametric generation.

### Step 2: Epistemic Context Creation

```python
epistemic_context = EpistemicContext(
    generation_mode=GenerationMode.RAG,
    evidence_scope="retrieved",
    evidence_provided_to_model=True,
    closed_world_assumption=True,
)
```

### Step 3: Detection via Unified Infrastructure

```python
result = self.unified_detector.detect(
    query=query,
    generated_answer=generated_answer,
    retrieved_context=retrieved_context,  # Exact evidence from generation
    force_retrieve=False,  # NEVER auto-retrieve for RAG
    epistemic_context=epistemic_context,
)
```

**Key**: `force_retrieve=False` ensures the detector ONLY uses the provided evidence, not post-hoc retrieval.

### Step 4: Return with Epistemic Tracking

```python
return HallucinationResult(
    is_hallucinated=True/False,
    hallucination_risk=0.0-1.0,
    epistemic_context=epistemic_context,
    epistemic_warnings=[],  # No warnings for explicit RAG mode
    ...
)
```

---

## Hallucination Types in RAG

### 1. Grounding Failure (Semantic)

**Definition**: Claim is topically related to evidence but not explicitly supported.

**Example**:
```
Evidence: "The Fourth Amendment protects against unreasonable searches."
Claim: "The Fourth Amendment requires a warrant for all searches."
```

**Detection**:
- ❌ Semantic similarity: HIGH (0.85)
- ❌ NLI entailment: LOW (0.42) - not entailed
- **Verdict**: Hallucination (grounding failure)

### 2. Contradiction

**Definition**: Claim directly contradicts retrieved evidence.

**Example**:
```
Evidence: "Qualified immunity protects officers from civil lawsuits."
Claim: "Qualified immunity does not protect officers from lawsuits."
```

**Detection**:
- ✅ Semantic similarity: HIGH (0.92)
- ❌ NLI label: CONTRADICTION (0.91)
- **Verdict**: Hallucination (contradiction)

### 3. Fabrication

**Definition**: Claim is unrelated to retrieved evidence.

**Example**:
```
Evidence: [Documents about contract law]
Claim: "The exclusionary rule prevents illegally obtained evidence from being admitted."
```

**Detection**:
- ❌ Semantic similarity: LOW (0.31) - no grounding
- **Verdict**: Hallucination (semantic failure, skips NLI)

### 4. Citation Hallucination (Structural)

**Definition**: Answer cites a case not in retrieved evidence.

**Example**:
```
Retrieved evidence: [Miranda v. Arizona, Mapp v. Ohio]
Answer: "As established in Terry v. Ohio..."
```

**Detection**:
- ❌ Structural verification: "Terry v. Ohio" not in evidence
- **Verdict**: Hallucination (structural/citation)

---

## Configuration

### Thresholds

RAG detection uses the same thresholds as the unified detector:

```python
from src.core.config import DetectorConfig

config = DetectorConfig(
    # NLI thresholds
    nli_entailment_threshold=0.70,  # Min score for entailment
    nli_contradiction_threshold=0.70,  # Min score for contradiction

    # Semantic thresholds
    semantic_similarity_threshold=0.50,  # Pre-filter before NLI

    # Citation thresholds
    citation_require_context=True,  # Citations must be in retrieved_context
    citation_fuzzy_match_threshold=0.85,
)

detector = RAGHallucinationDetector(config=config)
```

### Recommended Settings for RAG

**Strict (academic evaluation)**:
```python
config = DetectorConfig(
    nli_entailment_threshold=0.75,  # High bar for grounding
    semantic_similarity_threshold=0.60,  # Stricter pre-filter
    citation_require_context=True,  # Enforce citation grounding
)
```

**Lenient (production with recall focus)**:
```python
config = DetectorConfig(
    nli_entailment_threshold=0.60,  # More permissive
    semantic_similarity_threshold=0.40,  # Catch more potential grounding
    citation_require_context=False,  # Allow external citations
)
```

---

## Batch Processing

For evaluation on test sets:

```python
detector = RAGHallucinationDetector()

# Batch detection
results = detector.detect_batch(
    queries=test_queries,
    answers=rag_answers,
    contexts=retrieved_contexts  # List[List[str]] - one context per answer
)

# Compute metrics
hallucination_rate = sum(r.is_hallucinated for r in results) / len(results)
avg_confidence = sum(r.confidence for r in results) / len(results)

print(f"Hallucination rate: {hallucination_rate:.2%}")
print(f"Average confidence: {avg_confidence:.2f}")
```

---

## Integration Examples

### With Custom RAG Pipeline

```python
class MyRAGSystem:
    def __init__(self):
        self.retriever = MyRetriever()
        self.generator = MyGenerator()
        self.hallucination_detector = RAGHallucinationDetector()

    def generate_with_detection(self, query):
        # Retrieval
        passages = self.retriever.retrieve(query)

        # Generation
        answer = self.generator.generate(query, context=passages)

        # Hallucination detection
        detection_result = self.hallucination_detector.detect(
            query=query,
            generated_answer=answer,
            retrieved_context=passages
        )

        return {
            "answer": answer,
            "passages": passages,
            "is_safe": not detection_result.is_hallucinated,
            "hallucination_risk": detection_result.hallucination_risk,
            "epistemic_mode": detection_result.epistemic_context.generation_mode.value,
        }
```

### With Citations

```python
# RAG with citation generation
query = "What is qualified immunity?"
passages = retriever.retrieve(query)
answer_with_citations = rag_model.generate(
    query,
    context=passages,
    require_citations=True
)
# Output: "Qualified immunity protects officers from civil liability (Harlow v. Fitzgerald, 1982)..."

# Detection includes citation verification
result = rag_detector.detect(query, answer_with_citations, passages)

# Structural layer checks:
# ✓ Is "Harlow v. Fitzgerald" in passages?
# ✓ Is the citation valid?
```

---

## Result Interpretation

### Success Case

```python
result = detector.detect(query, answer, passages)

if not result.is_hallucinated:
    print("✓ All claims grounded")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Claims verified: {len(result.claims)}")
    print(f"  Evidence scope: {result.epistemic_context.evidence_scope}")
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
            print(f"    Semantic score: {claim_result['semantic_score']:.2f}")
```

---

## Academic Use

### Evaluation Methodology

```python
# Evaluate RAG system on benchmark
rag_detector = RAGHallucinationDetector()

results = []
for item in benchmark:
    # Generate with RAG
    passages = retriever.retrieve(item['query'])
    answer = rag_model.generate(item['query'], context=passages)

    # Detect
    detection = rag_detector.detect(
        query=item['query'],
        generated_answer=answer,
        retrieved_context=passages
    )

    results.append({
        "query": item['query'],
        "is_hallucinated": detection.is_hallucinated,
        "hallucination_risk": detection.hallucination_risk,
        "epistemic_mode": detection.epistemic_context.generation_mode.value,
    })

# Report metrics
pd.DataFrame(results).to_csv("rag_evaluation.csv")
```

### Comparing RAG Systems

```python
# System A vs System B
system_a_results = [rag_detector.detect(q, a_ans, a_ctx) for ...]
system_b_results = [rag_detector.detect(q, b_ans, b_ctx) for ...]

# Both use same epistemic assumptions (RAG closed-world)
# Comparison is valid
print(f"System A hallucination rate: {sum(r.is_hallucinated for r in system_a_results) / len(system_a_results):.2%}")
print(f"System B hallucination rate: {sum(r.is_hallucinated for r in system_b_results) / len(system_b_results):.2%}")
```

---

## Limitations

### 1. Perfect Retrieval Assumption

**Limitation**: Detector assumes retrieved evidence is complete and correct.

**Impact**: If retrieval misses relevant documents, grounded claims may be flagged as hallucinations.

**Mitigation**: Evaluate retrieval quality separately (recall@k, MRR).

### 2. Closed-World Strictness

**Limitation**: Flags claims using common knowledge as hallucinations.

**Example**:
```
Evidence: [Legal documents about murder]
Claim: "Murder is illegal in all U.S. states."
Verdict: Hallucination (not in evidence)
```

**Mitigation**: This is **intentional** for RAG. If model should use common knowledge, you're not using RAG correctly.

### 3. Citation Format Sensitivity

**Limitation**: May miss citations in non-standard formats.

**Mitigation**: Configure `citation_fuzzy_match_threshold` and add custom citation patterns.

---

## See Also

- [Why Three Pipelines?](WHY_THREE_PIPELINES.md) - Rationale for separate detectors (aspirational; NOT implemented)
- [Non-RAG Detection Architecture](NONRAG_DETECTION_ARCHITECTURE.md) - For parametric generation
- [Unified Detection Architecture](UNIFIED_DETECTION_ARCHITECTURE.md) - For flexible/hybrid use
- [Overall Architecture](../../HALLUCINATION_DETECTION_ARCHITECTURE.md) - Full system documentation

---

**Questions?** See docs/ or open an issue.
