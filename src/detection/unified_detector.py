"""
Factuality Detector - Contradiction-based hallucination detection.

Pipeline:
1. Claim Decomposition - Split answer into atomic claims
2. Retrieval - Find relevant evidence from oracle corpus
3. NLI - Check each claim against retrieved evidence
4. Aggregation - Per-claim verdict: CONTRADICTED / SUPPORTED / UNSUPPORTED

NLI Label Mapping:
- Entailment   → SUPPORTED (evidence confirms claim)
- Contradiction → CONTRADICTED (evidence refutes claim = hallucination)
- Neutral      → UNSUPPORTED (no evidence either way, epistemic uncertainty)

Hallucination Rate (Conditional Error Rate):
    hallucination_rate = Contradicted / (Supported + Contradicted)

This metric answers: "Of the claims we could actually verify, what
percentage were wrong?"

Only SUPPORTED + CONTRADICTED claims are "testable" - claims where
evidence exists to make a factual determination. UNSUPPORTED claims
represent epistemic uncertainty ("not enough evidence"), not factual
error. Including them in the denominator would conflate uncertainty
with error, which is conceptually incorrect and inconsistent with
standard fact-verification frameworks.

This ensures fair comparison between RAG and non-RAG systems.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.config import DetectorConfig
from ..preprocessing.claim_decomposer import ClaimDecomposer
from ..retrieval.document_retriever import DocumentRetriever
from .entailment_verifier import EntailmentVerifier


@dataclass
class ClaimVerdict:
    """Verdict for a single claim."""
    claim: str
    verdict: str  # "CONTRADICTED", "SUPPORTED", "UNSUPPORTED"
    confidence: float
    evidence_used: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "evidence_used": self.evidence_used[:200] if self.evidence_used else None,
        }


@dataclass
class FactualityResult:
    """
    Result of factuality detection.

    DUAL METRICS (always report both per guidance):

    1. Testability Rate = (Supported + Contradicted) / Total Claims
       "What fraction of claims could we adjudicate?"
       Measures epistemic coverage - how much of the output is verifiable.

    2. Hallucination Rate = Contradicted / (Supported + Contradicted)
       "Of the claims we could verify, what percentage were wrong?"
       A conditional error rate among testable (adjudicable) claims only.

    This follows standard fact-verification frameworks (FEVER, etc.) which
    classify claims as SUPPORTED, REFUTED, or NOTENOUGHINFO, and compute
    accuracy only over adjudicable claims.

    NLI Label Mapping:
        - Entailment   → SUPPORTED (evidence confirms claim)
        - Contradiction → CONTRADICTED (evidence refutes claim = hallucination)
        - Neutral      → UNSUPPORTED (insufficient evidence, epistemic uncertainty)

    Theoretical Justification (Dahl et al. 2024, Magesh et al. 2024):
        - UNSUPPORTED ≠ hallucination; it signals a knowledge gap, not an error
        - Including neutrals in denominator conflates uncertainty with falsehood
        - This yields a conservative lower-bound on hallucination rate
        - Separating testability from accuracy prevents "RAG bias" where systems
          that answer fewer questions trivially achieve lower error rates

    Always report BOTH metrics to make coverage vs. accuracy trade-offs transparent.
    """

    # Overall verdict
    is_hallucinated: bool  # True ONLY if any claim is CONTRADICTED

    # DUAL METRICS (always report both)
    # None when num_testable == 0: the rate is undefined, not zero. Never
    # average these per-answer values without filtering None; pool the counts.
    hallucination_rate: Optional[float]  # Contradicted / (Supported + Contradicted)
    testability_rate: float  # (Supported + Contradicted) / Total Claims
    num_testable: int  # Supported + Contradicted (adjudicable claims)

    # Raw rates over ALL claims (for detailed analysis)
    contradiction_rate: float  # num_contradicted / num_claims
    support_rate: float  # num_supported / num_claims
    unsupported_rate: float  # num_unsupported / num_claims

    # Claim-level details
    claims: List[ClaimVerdict]
    num_claims: int
    num_contradicted: int
    num_supported: int
    num_unsupported: int

    # Telemetry
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "is_hallucinated": self.is_hallucinated,
            # Dual metrics
            "hallucination_rate": self.hallucination_rate,
            "testability_rate": self.testability_rate,
            "num_testable": self.num_testable,
            # Raw rates
            "contradiction_rate": self.contradiction_rate,
            "support_rate": self.support_rate,
            "unsupported_rate": self.unsupported_rate,
            # Counts
            "num_claims": self.num_claims,
            "num_contradicted": self.num_contradicted,
            "num_supported": self.num_supported,
            "num_unsupported": self.num_unsupported,
            "claims": [c.to_dict() for c in self.claims],
            "latency_ms": self.latency_ms,
        }


# Keep old name for backwards compatibility
HallucinationResult = FactualityResult


class FactualityDetector:
    """
    Factuality detector using contradiction-based hallucination detection.

    Pipeline:
    1. Decompose answer into atomic claims
    2. Retrieve relevant evidence from oracle corpus
    3. Run NLI to detect contradictions
    4. Aggregate results (only contradictions = hallucinations)

    Usage:
        detector = FactualityDetector()
        result = detector.detect(
            query="What is qualified immunity?",
            answer="Qualified immunity protects all government employees."
        )

        if result.is_hallucinated:
            print(f"Found {result.num_contradicted} contradicted claims")
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        """
        Initialize factuality detector.

        Args:
            config: Detector configuration (uses defaults if not provided)
        """
        self.config = config or DetectorConfig()

        print("Initializing Factuality Detector...")

        # Claim decomposer
        self.claim_decomposer = ClaimDecomposer()

        # NLI entailment verifier with threshold enforcement
        self.nli_verifier = EntailmentVerifier(
            model_name=self.config.cross_encoder_model,
            device=self.config.device,
            entailment_threshold=self.config.nli_entailment_threshold,
            contradiction_threshold=self.config.nli_contradiction_threshold,
        )

        # Document retriever (for oracle corpus)
        self.retriever: Optional[DocumentRetriever] = None
        if self.config.retrieval_enabled:
            try:
                self.retriever = DocumentRetriever(
                    index_path=str(self.config.retrieval_index_path),
                    embedding_model=self.config.retrieval_embedding_model,
                    device=self.config.device,
                )
                print(f"[OK] Retrieval loaded: {self.config.retrieval_index_path}")
            except Exception as e:
                # Do not continue with retrieval silently disabled. Without a
                # retriever every claim becomes UNSUPPORTED, num_testable is 0,
                # and the run reports a clean 0% hallucination rate - a broken
                # environment that reads as a perfect result. Fail loudly.
                raise RuntimeError(
                    f"Retrieval is enabled but the index could not be loaded from "
                    f"{self.config.retrieval_index_path}: {e}. Every claim would be "
                    f"scored UNSUPPORTED, producing a meaningless 0% hallucination "
                    f"rate. Build the index (scripts/build_retrieval_index.py) or set "
                    f"retrieval_enabled=False deliberately."
                ) from e

        print("[OK] Factuality Detector initialized")

    def detect(
        self,
        query: str,
        answer: str,
        evidence: Optional[List[str]] = None,
    ) -> FactualityResult:
        """
        Detect factual contradictions in an answer.

        Args:
            query: Original question
            answer: Generated answer to verify
            evidence: Optional pre-provided evidence. If None, retrieves from corpus.

        Returns:
            FactualityResult with per-claim verdicts
        """
        start_time = time.time()

        # ========================================
        # Step 1: Claim Decomposition
        # ========================================
        decomposition = self.claim_decomposer.decompose(answer)
        claims = decomposition.claims if decomposition else []

        if not claims:
            # No claims to verify
            return FactualityResult(
                is_hallucinated=False,
                hallucination_rate=None,  # undefined: nothing was adjudicable
                testability_rate=0.0,
                num_testable=0,
                contradiction_rate=0.0,
                support_rate=0.0,
                unsupported_rate=1.0,
                claims=[],
                num_claims=0,
                num_contradicted=0,
                num_supported=0,
                num_unsupported=0,
                latency_ms=(time.time() - start_time) * 1000,
            )

        # ========================================
        # Step 2: Retrieval (if no evidence provided)
        # ========================================
        # Evidence supplied by the caller applies to the whole answer. When we
        # retrieve it ourselves we do so per claim, below: retrieving once for
        # f"{query} {answer}" returned one passage set covering the whole answer,
        # so every claim was checked against passages fetched for other claims.
        # In a corpus holding many cases on the same topic that is how an
        # unrelated holding ends up contradicting a correct claim.
        caller_supplied_evidence = bool(evidence)
        if not caller_supplied_evidence:
            evidence = []

        # ========================================
        # Step 3: NLI Verification (per claim)
        # ========================================
        claim_verdicts = []

        for claim in claims:
            claim_text = claim.claim_text

            # Retrieve evidence for this specific claim unless the caller gave us
            # an evidence set to use for the whole answer.
            if caller_supplied_evidence:
                claim_evidence = evidence
            elif self.retriever is not None:
                claim_evidence = self.retriever.retrieve_for_claim(
                    claim=claim_text,
                    top_k=self.config.retrieval_top_k,
                )
            else:
                claim_evidence = []

            if not claim_evidence:
                # No evidence available - unsupported
                claim_verdicts.append(ClaimVerdict(
                    claim=claim_text,
                    verdict="UNSUPPORTED",
                    confidence=0.0,
                    evidence_used=None,
                ))
                continue

            # Run NLI against this claim's evidence
            nli_result = self.nli_verifier.verify_claim_against_multiple_sources(
                claim_text, claim_evidence
            )

            overall_status = nli_result.get("overall_status", "UNSUPPORTED")
            confidence = nli_result.get("confidence", 0.0)

            # Map NLI status to verdict
            if overall_status == "CONTRADICTED":
                verdict = "CONTRADICTED"
            elif overall_status == "SUPPORTED":
                verdict = "SUPPORTED"
            else:
                verdict = "UNSUPPORTED"

            # Find the evidence that was most relevant
            per_source = nli_result.get("per_source_results", [])
            best_evidence = None
            if per_source:
                # Find the source with highest confidence for the verdict
                for src in per_source:
                    if src.get("label", "").upper() == verdict or (
                        verdict == "CONTRADICTED" and src.get("contradicted")
                    ) or (verdict == "SUPPORTED" and src.get("supported")):
                        best_evidence = src.get("source", "")
                        break
                if not best_evidence and per_source:
                    best_evidence = per_source[0].get("source", "")

            claim_verdicts.append(ClaimVerdict(
                claim=claim_text,
                verdict=verdict,
                confidence=confidence,
                evidence_used=best_evidence,
            ))

        # ========================================
        # Step 4: Aggregation
        # ========================================
        num_claims = len(claim_verdicts)
        num_contradicted = sum(1 for c in claim_verdicts if c.verdict == "CONTRADICTED")
        num_supported = sum(1 for c in claim_verdicts if c.verdict == "SUPPORTED")
        num_unsupported = sum(1 for c in claim_verdicts if c.verdict == "UNSUPPORTED")

        # Only contradictions = hallucinations
        is_hallucinated = num_contradicted > 0

        # TESTABLE (adjudicable) claims = claims where evidence exists
        # Unsupported claims (NLI neutral) represent epistemic uncertainty, not error
        num_testable = num_supported + num_contradicted

        # ========================================
        # DUAL METRICS (always report both per guidance)
        # ========================================
        # 1. Testability Rate: "What fraction of claims could we adjudicate?"
        #    Measures epistemic coverage of the evidence corpus
        testability_rate = num_testable / num_claims if num_claims > 0 else 0.0

        # 2. Hallucination Rate: "Of claims we could verify, what % were wrong?"
        #    Conditional error rate among testable claims only
        #    This is a conservative lower-bound (Dahl et al. 2024)
        #    When nothing was adjudicable the rate is undefined, not zero.
        #    Returning 0.0 made an answer whose claims could not be checked look
        #    identical to one whose claims were all verified correct, so averaging
        #    per-answer rates counted untestable answers as perfect scores and
        #    pulled the reported rate down - the "RAG bias" the dual metric is
        #    meant to prevent. Callers must now either skip None or pool counts
        #    (sum contradicted / sum testable), which is what the README reports.
        hallucination_rate = (
            num_contradicted / num_testable if num_testable > 0 else None
        )

        # Raw rates over all claims (for detailed analysis)
        contradiction_rate = num_contradicted / num_claims if num_claims > 0 else 0.0
        support_rate = num_supported / num_claims if num_claims > 0 else 0.0
        unsupported_rate = num_unsupported / num_claims if num_claims > 0 else 0.0

        return FactualityResult(
            is_hallucinated=is_hallucinated,
            hallucination_rate=hallucination_rate,
            testability_rate=testability_rate,
            num_testable=num_testable,
            contradiction_rate=contradiction_rate,
            support_rate=support_rate,
            unsupported_rate=unsupported_rate,
            claims=claim_verdicts,
            num_claims=num_claims,
            num_contradicted=num_contradicted,
            num_supported=num_supported,
            num_unsupported=num_unsupported,
            latency_ms=(time.time() - start_time) * 1000,
        )

    def __repr__(self) -> str:
        return f"FactualityDetector(device={self.config.device})"


# Backwards compatibility alias
UnifiedHallucinationDetector = FactualityDetector
