"""
Detection module for Legal Hallucination Detector.

FACTUALITY-BASED DETECTION (Epistemically Symmetric):
=====================================================

Pipeline: Claim Decomposition → Retrieval → NLI → Aggregation

NLI Label Mapping:
- Entailment   → SUPPORTED (evidence confirms claim)
- Contradiction → CONTRADICTED (evidence refutes claim = hallucination)
- Neutral      → UNSUPPORTED (no evidence either way, epistemic uncertainty)

DUAL METRICS (always report both):
==================================

1. Testability Rate = (Supported + Contradicted) / Total Claims
   "What fraction of claims could we adjudicate?"
   Measures epistemic coverage of the evidence corpus.

2. Hallucination Rate = Contradicted / (Supported + Contradicted)
   "Of the claims we could verify, what percentage were wrong?"
   Conditional error rate among testable (adjudicable) claims only.

This follows standard fact-verification frameworks (FEVER, etc.) which
exclude NOTENOUGHINFO claims from accuracy calculations. UNSUPPORTED claims
represent epistemic uncertainty, NOT factual error.

Always report BOTH metrics to make coverage vs. accuracy trade-offs transparent.
This prevents "RAG bias" where systems that answer fewer questions trivially
achieve lower error rates.

USAGE:
======

Main detector (recommended):
    from src.detection import FactualityDetector

    detector = FactualityDetector()
    result = detector.detect(
        query="What is qualified immunity?",
        answer="Qualified immunity protects all citizens."
    )

    # DUAL METRICS
    print(f"Testability: {result.testability_rate:.1%}")
    print(f"Hallucination rate: {result.hallucination_rate:.1%}")
    print(f"Breakdown: {result.num_supported} supported, "
          f"{result.num_contradicted} contradicted, "
          f"{result.num_unsupported} unsupported")

RAG-specific wrapper:
    from src.detection import RAGHallucinationDetector

    detector = RAGHallucinationDetector()
    result = detector.detect(query, answer, retrieved_context)

Non-RAG wrapper (auto-retrieves from oracle corpus):
    from src.detection import NonRAGHallucinationDetector

    detector = NonRAGHallucinationDetector()
    result = detector.detect(query, answer)
"""

# Main factuality detector
from .unified_detector import (
    FactualityDetector,
    FactualityResult,
    ClaimVerdict,
    # Backwards compatibility
    UnifiedHallucinationDetector,
    HallucinationResult,
)

# Thin wrappers for RAG/NonRAG
from .rag_detector import RAGHallucinationDetector, create_rag_detector
from .nonrag_detector import NonRAGHallucinationDetector, create_nonrag_detector

# NLI verifier (used internally)
from .entailment_verifier import EntailmentVerifier, EntailmentResult, EntailmentLabel

__all__ = [
    # Main detector
    "FactualityDetector",
    "FactualityResult",
    "ClaimVerdict",
    # Backwards compatibility aliases
    "UnifiedHallucinationDetector",
    "HallucinationResult",
    # Wrappers
    "RAGHallucinationDetector",
    "NonRAGHallucinationDetector",
    "create_rag_detector",
    "create_nonrag_detector",
    # Internal components
    "EntailmentVerifier",
    "EntailmentResult",
    "EntailmentLabel",
]
