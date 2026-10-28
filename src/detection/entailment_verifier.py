"""
Natural Language Inference (NLI) based entailment verification.

Uses fine-tuned NLI models to verify if claims are logically entailed by
retrieved source documents. This catches subtle hallucinations where claims
contradict sources or go beyond what sources state.

CRITICAL FIXES from detector_package v2:
- Aggressive pre-truncation to avoid tensor shape errors with long legal texts
- Correct label mapping for different NLI model outputs
- Batched inference support for 10-20x speedup
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import CrossEncoder


class EntailmentLabel(Enum):
    """Classification of relationship between premise and hypothesis."""

    ENTAILMENT = "entailment"  # Premise supports the hypothesis
    CONTRADICTION = "contradiction"  # Premise contradicts the hypothesis
    NEUTRAL = "neutral"  # Premise is unrelated to hypothesis


@dataclass
class EntailmentResult:
    """Result of entailment verification."""

    claim: str
    source: str
    label: EntailmentLabel
    confidence: float  # 0-1: Confidence in classification
    is_supported: bool  # True if claim is entailed by source
    is_contradicted: bool  # True if claim is contradicted by source

    def __repr__(self) -> str:
        status = (
            "[PASS] SUPPORTED"
            if self.is_supported
            else ("[FAIL] CONTRADICTED" if self.is_contradicted else "? NEUTRAL")
        )
        return f"{status} ({self.confidence:.2%}): {self.claim[:50]}..."

    def to_dict(self) -> Dict:
        return {
            "claim": self.claim,
            "source": self.source[:200],  # Truncate source for readability
            "label": self.label.value,
            "confidence": self.confidence,
            "supported": self.is_supported,
            "contradicted": self.is_contradicted,
        }


class EntailmentVerifier:
    """
    Verifies logical entailment between claims and source documents.

    Uses cross-encoder NLI models which are more accurate than bi-encoders.
    Supports multiple model options:
    - cross-encoder/nli-deberta-v3-base (default, balanced, 3-class)
    - cross-encoder/nli-deberta-v3-large (slower, more accurate, 3-class)
    - cross-encoder/qnli-distilroberta-base (fast, 2-class only)

    THRESHOLD ENFORCEMENT:
    - Only marks ENTAILMENT if entailment_score >= entailment_threshold
    - Only marks CONTRADICTION if contradiction_score >= contradiction_threshold
    - Otherwise marks NEUTRAL (maps to UNSUPPORTED in hallucination detection)

    This prevents low-confidence predictions from being treated as definitive
    verdicts, which would inflate hallucination rates.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: str = "cpu",
        entailment_threshold: float = 0.70,
        contradiction_threshold: float = 0.70,
    ) -> None:
        """
        Initialize verifier with specified NLI model and thresholds.

        Args:
            model_name: HuggingFace cross-encoder model
            device: Device to run model on ('cpu', 'cuda', 'mps')
            entailment_threshold: Min confidence to mark as ENTAILMENT (default: 0.70)
            contradiction_threshold: Min confidence to mark as CONTRADICTION (default: 0.70)
        """
        self.model_name = model_name
        self.device = device
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

        print(f"[NLI] Thresholds: entailment >= {entailment_threshold}, contradiction >= {contradiction_threshold}")

        try:
            self.model = CrossEncoder(model_name, device=device)
            self.model.max_length = 512  # Enforce truncation
            # Force underlying HF model onto GPU
            if device == "cuda" and hasattr(self.model, "model"):
                try:
                    self.model.model = self.model.model.to("cuda")
                    print("Forced NLI HF model onto CUDA")
                except Exception as e:
                    print("Error moving HF model to CUDA:", e)
            print(f"Loaded NLI model: {model_name} on {device}")
        except Exception as e:
            print(f"Error loading NLI model {model_name}: {e}")
            self.model = None

    @staticmethod
    def _has_cuda() -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def verify_claim(
        self, claim: str, source_document: str, return_all_results: bool = False
    ) -> EntailmentResult:
        """
        Verify if a claim is entailed by a source document.

        Args:
            claim: Atomic claim to verify
            source_document: Source text that may support or refute the claim
            return_all_results: If True, return full result dict

        Returns:
            EntailmentResult with classification and confidence
        """
        if not self.model:
            return EntailmentResult(
                claim=claim,
                source=source_document,
                label=EntailmentLabel.NEUTRAL,
                confidence=0.0,
                is_supported=False,
                is_contradicted=False,
            )

        # CRITICAL FIX: Aggressive truncation to prevent tensor shape errors
        # Legal texts often exceed 512 tokens. Pre-truncate BEFORE tokenization.
        max_source_chars = 1500  # Approx 375 tokens
        max_claim_chars = 500  # Approx 125 tokens
        truncated_source = source_document[:max_source_chars]
        truncated_claim = claim[:max_claim_chars]

        # Run NLI with softmax to get probabilities
        scores = self.model.predict(
            [[truncated_source, truncated_claim]], apply_softmax=True
        )[0]

        # CRITICAL FIX: Correct label mapping for DeBERTa v3
        # Model config: {"0": "CONTRADICTION", "1": "ENTAILMENT", "2": "NEUTRAL"}
        label_mapping = [
            EntailmentLabel.CONTRADICTION,
            EntailmentLabel.ENTAILMENT,
            EntailmentLabel.NEUTRAL,
        ]

        # Handle different output types from sentence-transformers
        if isinstance(scores, (float, np.floating)):
            # Single score (2-class model)
            score = scores
            if score >= self.entailment_threshold:
                label = EntailmentLabel.ENTAILMENT
            else:
                label = EntailmentLabel.NEUTRAL
            confidence = float(score) if label == EntailmentLabel.ENTAILMENT else 1.0 - float(score)
        elif len(scores) == 3:
            # 3-class model (DeBERTa)
            # THRESHOLD ENFORCEMENT: Only assign label if confidence meets threshold
            # Scores are [CONTRADICTION, ENTAILMENT, NEUTRAL] for DeBERTa v3
            contradiction_score = float(scores[0])
            entailment_score = float(scores[1])
            neutral_score = float(scores[2])

            # Apply thresholds - only mark as ENTAILMENT/CONTRADICTION if confident enough
            if contradiction_score >= self.contradiction_threshold:
                label = EntailmentLabel.CONTRADICTION
                confidence = contradiction_score
            elif entailment_score >= self.entailment_threshold:
                label = EntailmentLabel.ENTAILMENT
                confidence = entailment_score
            else:
                # Below threshold = NEUTRAL (will become UNSUPPORTED)
                label = EntailmentLabel.NEUTRAL
                confidence = neutral_score
        elif len(scores) == 2:
            # 2-class QNLI model
            entailment_score = scores[1]
            if entailment_score >= self.entailment_threshold:
                label = EntailmentLabel.ENTAILMENT
            else:
                label = EntailmentLabel.NEUTRAL
            confidence = (
                float(entailment_score)
                if label == EntailmentLabel.ENTAILMENT
                else float(1.0 - entailment_score)
            )
        else:
            # Fallback for unexpected array sizes
            label = EntailmentLabel.NEUTRAL
            confidence = 0.5

        is_supported = label == EntailmentLabel.ENTAILMENT
        is_contradicted = label == EntailmentLabel.CONTRADICTION

        return EntailmentResult(
            claim=claim,
            source=source_document,
            label=label,
            confidence=confidence,
            is_supported=is_supported,
            is_contradicted=is_contradicted,
        )

    def verify_claim_against_multiple_sources(
        self, claim: str, source_documents: List[str]
    ) -> Dict:
        """
        Verify a claim against multiple source documents.

        A claim is considered:
        - SUPPORTED if entailed by at least one source and not contradicted
        - CONTRADICTED if contradicted by any source
        - UNSUPPORTED if not entailed by any source

        Args:
            claim: Claim to verify
            source_documents: List of source documents

        Returns:
            Dict with overall result and per-source details
        """
        if not source_documents:
            return {
                "claim": claim,
                "overall_status": "NO_SOURCES_PROVIDED",
                "confidence": 0.0,
                "num_sources": 0,
                "error": "No source documents provided",
            }

        if not claim or len(claim.strip()) < 5:
            return {
                "claim": claim,
                "overall_status": "INVALID_CLAIM",
                "confidence": 0.0,
                "num_sources": len(source_documents),
                "error": "Claim too short or empty",
            }

        results = []
        entailment_count = 0
        contradiction_count = 0

        for source in source_documents:
            result = self.verify_claim(claim, source)
            results.append(result)

            if result.is_supported:
                entailment_count += 1
            if result.is_contradicted:
                contradiction_count += 1

        # Determine overall status
        if contradiction_count > 0:
            overall_status = "CONTRADICTED"
            confidence = max(r.confidence for r in results if r.is_contradicted)
        elif entailment_count > 0:
            overall_status = "SUPPORTED"
            confidence = max(r.confidence for r in results if r.is_supported)
        else:
            overall_status = "UNSUPPORTED"
            confidence = max(r.confidence for r in results) if results else 0.0

        return {
            "claim": claim,
            "overall_status": overall_status,
            "confidence": confidence,
            "num_sources": len(source_documents),
            "supporting_sources": entailment_count,
            "contradicting_sources": contradiction_count,
            "neutral_sources": len(source_documents)
            - entailment_count
            - contradiction_count,
            "per_source_results": [r.to_dict() for r in results],
        }

    def verify_batch(
        self, claim_source_pairs: List[tuple[str, str]]
    ) -> List[EntailmentResult]:
        """
        PERFORMANCE FIX: Batch verification for 10-20x speedup.

        Args:
            claim_source_pairs: List of (claim, source) tuples

        Returns:
            List of EntailmentResult objects
        """
        if not self.model or not claim_source_pairs:
            return []

        # Pre-truncate all pairs
        max_source_chars = 1500
        max_claim_chars = 500
        truncated_pairs = [
            (source[:max_source_chars], claim[:max_claim_chars])
            for claim, source in claim_source_pairs
        ]

        # Batch inference (MUCH faster than sequential)
        scores_batch = self.model.predict(truncated_pairs, apply_softmax=True)

        # Label mapping for DeBERTa v3
        label_mapping = [
            EntailmentLabel.CONTRADICTION,
            EntailmentLabel.ENTAILMENT,
            EntailmentLabel.NEUTRAL,
        ]

        results = []
        for (claim, source), scores in zip(claim_source_pairs, scores_batch):
            if len(scores) == 3:
                # THRESHOLD ENFORCEMENT (same logic as verify_claim)
                # Scores are [CONTRADICTION, ENTAILMENT, NEUTRAL] for DeBERTa v3
                contradiction_score = float(scores[0])
                entailment_score = float(scores[1])
                neutral_score = float(scores[2])

                if contradiction_score >= self.contradiction_threshold:
                    label = EntailmentLabel.CONTRADICTION
                    confidence = contradiction_score
                elif entailment_score >= self.entailment_threshold:
                    label = EntailmentLabel.ENTAILMENT
                    confidence = entailment_score
                else:
                    label = EntailmentLabel.NEUTRAL
                    confidence = neutral_score
            else:
                # Fallback for 2-class or unexpected outputs
                label = EntailmentLabel.NEUTRAL
                confidence = 0.5

            is_supported = label == EntailmentLabel.ENTAILMENT
            is_contradicted = label == EntailmentLabel.CONTRADICTION

            result = EntailmentResult(
                claim=claim,
                source=source,
                label=label,
                confidence=confidence,
                is_supported=is_supported,
                is_contradicted=is_contradicted,
            )
            results.append(result)

        return results

    def generate_claim_verification_report(
        self, claims: List[str], sources: List[str]
    ) -> str:
        """
        Generate a human-readable report of claim verification results.

        Args:
            claims: List of claims
            sources: List of source documents

        Returns:
            Formatted report string
        """
        # Create all claim-source pairs
        pairs = [(claim, source) for claim in claims for source in sources]

        # Batch verification for efficiency
        results = self.verify_batch(pairs)

        # Group by claim
        claims_dict: Dict[str, List[EntailmentResult]] = {}
        for result in results:
            if result.claim not in claims_dict:
                claims_dict[result.claim] = []
            claims_dict[result.claim].append(result)

        # Generate report
        report = []
        report.append("=" * 80)
        report.append("ENTAILMENT VERIFICATION REPORT")
        report.append("=" * 80)
        report.append(f"Total Claims: {len(claims_dict)}")
        report.append(f"Total Sources: {len(sources)}")
        report.append("")

        for claim, claim_results in claims_dict.items():
            supported = sum(1 for r in claim_results if r.is_supported)
            contradicted = sum(1 for r in claim_results if r.is_contradicted)

            report.append("-" * 80)
            report.append(f"CLAIM: {claim}")
            report.append("-" * 80)

            if contradicted > 0:
                report.append(f"STATUS: CONTRADICTED by {contradicted} source(s)")
            elif supported > 0:
                report.append(f"STATUS: SUPPORTED by {supported} source(s)")
            else:
                report.append("STATUS: UNSUPPORTED (not entailed by any source)")

            report.append("")
            report.append("Per-Source Results:")
            for result in claim_results:
                icon = "[PASS]" if result.is_supported else ("[FAIL]" if result.is_contradicted else "?")
                report.append(
                    f"  {icon} {result.label.value.upper()} ({result.confidence:.2%})"
                )
                report.append(f"     Source: {result.source[:70]}...")

            report.append("")

        return "\n".join(report)
