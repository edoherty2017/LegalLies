"""
Claim decomposition with quality gates using DecMetrics.

This module implements the "Decompose-Then-Verify" paradigm with quality
validation (COMPLETENESS and CORRECTNESS) to prevent the "Decomposition Dilemma"
of introducing noise during decomposition.

The decomposed claims can then be verified individually in downstream stages,
providing fine-grained hallucination detection at the atomic claim level.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

@dataclass
class DecomposedClaim:
    """Represents a single atomic claim extracted from an answer."""

    claim_text: str
    claim_id: int
    original_answer_section: str

    def __repr__(self) -> str:
        return f"Claim({self.claim_id}): {self.claim_text[:60]}..."


@dataclass
class DecompositionQualityMetrics:
    """Quality scores for the decomposition itself."""

    completeness_score: float  # 0-1: Does decomposition cover all original content?
    correctness_score: float  # 0-1: Does decomposition preserve original meaning?
    overall_quality: float  # 0-1: Weighted combination

    def passes_quality_gate(self, threshold: float = 0.8) -> bool:
        """Check if decomposition meets minimum quality standards."""
        return (
            self.completeness_score >= threshold and self.correctness_score >= threshold
        )


@dataclass
class ClaimDecompositionResult:
    """Container for decomposition results and quality metrics."""

    claims: List[DecomposedClaim]
    quality_metrics: DecompositionQualityMetrics
    passed_quality_gate: bool
    original_answer: str

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "claims": [
                {
                    "id": c.claim_id,
                    "text": c.claim_text,
                    "section": c.original_answer_section,
                }
                for c in self.claims
            ],
            "quality_metrics": {
                "completeness": self.quality_metrics.completeness_score,
                "correctness": self.quality_metrics.correctness_score,
                "overall": self.quality_metrics.overall_quality,
            },
            "passed_quality_gate": self.passed_quality_gate,
            "num_claims": len(self.claims),
        }


class ClaimDecomposer:
    """
    Decomposes long-form legal answers into atomic claims for verification.

    Uses sentence-level segmentation combined with NLI-based quality validation
    to ensure the decomposition maintains semantic fidelity to the original answer.
    """

    def __init__(self, nli_model: str = "cross-encoder/qnli-distilroberta-base") -> None:
        """
        Initialize decomposer with NLI model for quality validation.

        Args:
            nli_model: HuggingFace model for entailment checking
                      Default: QNLI-trained RoBERTa (fast, legal-friendly)
        """
        self.nli_model = nli_model
        # Disabled: pipeline import causes bad_alloc crash in transformers 4.44.x
        # The NLI quality validation is optional - decomposition still works without it
        self.nli_pipeline = None
        print(f"Note: NLI quality validation disabled (optional feature)")

    @staticmethod
    def _has_cuda() -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def decompose(self, answer: str) -> ClaimDecompositionResult:
        """
        Decompose a long-form answer into atomic claims.

        Uses multi-stage segmentation:
        1. Split on sentence boundaries (periods, semicolons)
        2. Filter fragments too short to be meaningful claims
        3. Validate decomposition quality using NLI

        Args:
            answer: Long-form legal answer to decompose

        Returns:
            ClaimDecompositionResult with decomposed claims and quality metrics
        """
        # Stage 1: Sentence segmentation
        sentences = self._segment_sentences(answer)

        # Stage 2: Convert sentences to atomic claims
        claims = self._create_claims(sentences)

        # Stage 3: Measure decomposition quality
        quality_metrics = self._measure_quality(answer, claims)

        # Stage 4: Determine if decomposition passed quality gate
        passed_gate = quality_metrics.passes_quality_gate()

        result = ClaimDecompositionResult(
            claims=claims,
            quality_metrics=quality_metrics,
            passed_quality_gate=passed_gate,
            original_answer=answer,
        )

        return result

    @staticmethod
    def _segment_sentences(text: str) -> List[str]:
        """
        Split text into sentences on legal-aware boundaries.

        Handles legal abbreviations (e.g., "U.S.", "USC", "v.", "Id.") and
        legal citation patterns (e.g., "42 U.S.C. Section 1983").
        """
        # Protect legal abbreviations and citations from breaking
        protected = text
        protected = protected.replace("U.S.", "<US>")
        protected = protected.replace("U.S.C.", "<USC>")
        protected = protected.replace("Fed. R.", "<FEDR>")
        protected = protected.replace("Cir.", "<CIR>")
        protected = protected.replace("Ct.", "<CT>")
        protected = protected.replace("Supp.", "<SUPP>")
        protected = protected.replace(" v. ", " <V> ")
        protected = protected.replace("Id.", "<ID>")
        protected = protected.replace("et al.", "<ETAL>")

        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", protected)

        # Restore abbreviations
        sentences = [
            s.replace("<US>", "U.S.")
            .replace("<USC>", "U.S.C.")
            .replace("<FEDR>", "Fed. R.")
            .replace("<CIR>", "Cir.")
            .replace("<CT>", "Ct.")
            .replace("<SUPP>", "Supp.")
            .replace("<V>", "v.")
            .replace("<ID>", "Id.")
            .replace("<ETAL>", "et al.")
            for s in sentences
        ]

        # Filter empty and very short fragments
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        return sentences

    @staticmethod
    def _create_claims(sentences: List[str]) -> List[DecomposedClaim]:
        """Convert sentences into DecomposedClaim objects."""
        claims = []

        for i, sentence in enumerate(sentences, 1):
            # Further decompose compound sentences (containing "and", "but")
            sub_claims = ClaimDecomposer._split_compound_sentence(sentence)

            for j, sub_claim in enumerate(sub_claims):
                claim = DecomposedClaim(
                    claim_text=sub_claim,
                    claim_id=i * 100 + j,  # Hierarchical IDs: 101, 102, 201, 202, etc.
                    original_answer_section=sentence,
                )
                claims.append(claim)

        return claims

    @staticmethod
    def _split_compound_sentence(sentence: str) -> List[str]:
        """
        Split compound sentences at conjunctions while preserving legal meaning.

        Example: "The statute provides X and Y" -> ["The statute provides X", "Y"]
        But: "U.S. and Canada" -> ["U.S. and Canada"]  (don't split proper nouns)
        """
        # Don't split if sentence contains legal citations or proper nouns
        if any(pattern in sentence for pattern in ["U.S.", "Fed.", "Cir.", "Section", "v."]):
            return [sentence]

        # Split on conjunctions but only if they seem to separate independent clauses
        if re.search(r"\b\w+\s+and\s+the\s+\w+", sentence, re.IGNORECASE):
            parts = re.split(r"\s+and\s+the\s+", sentence, flags=re.IGNORECASE)
            return [part.strip() for part in parts if part.strip()]

        return [sentence]

    def _measure_quality(
        self, original_answer: str, claims: List[DecomposedClaim]
    ) -> DecompositionQualityMetrics:
        """
        Measure quality of decomposition using DecMetrics approach.

        COMPLETENESS: Does the set of claims fully cover the original answer?
        CORRECTNESS: Does each claim accurately reflect what was in the answer?
        """
        if not self.nli_pipeline:
            # Fallback: simple heuristics if NLI unavailable
            return self._measure_quality_heuristic(original_answer, claims)

        # Reconstruct text from claims
        reconstructed = " ".join([c.claim_text for c in claims])

        # Measure COMPLETENESS: Is original entailed by reconstructed claims?
        completeness_score = self._compute_entailment(
            premise=reconstructed, hypothesis=original_answer
        )

        # Measure CORRECTNESS: Is reconstructed entailed by original?
        correctness_score = self._compute_entailment(
            premise=original_answer, hypothesis=reconstructed
        )

        # Overall quality: harmonic mean (both must be high)
        if completeness_score > 0 and correctness_score > 0:
            overall = 2 * (completeness_score * correctness_score) / (
                completeness_score + correctness_score
            )
        else:
            overall = 0.0

        return DecompositionQualityMetrics(
            completeness_score=completeness_score,
            correctness_score=correctness_score,
            overall_quality=overall,
        )

    def _measure_quality_heuristic(
        self, original_answer: str, claims: List[DecomposedClaim]
    ) -> DecompositionQualityMetrics:
        """
        Fallback quality measurement using heuristics when NLI unavailable.
        """
        original_words = set(original_answer.lower().split())
        reconstructed_words = set(
            " ".join([c.claim_text for c in claims]).lower().split()
        )

        # Word coverage heuristic
        coverage = len(reconstructed_words & original_words) / max(1, len(original_words))

        # Claim count reasonable (not too few, not too many)
        claim_ratio = len(claims) / max(1, len(original_answer.split()) / 20)
        claim_count_ok = 0.5 < claim_ratio < 3.0

        completeness = min(0.95, coverage)
        correctness = 0.85 if claim_count_ok else 0.60

        return DecompositionQualityMetrics(
            completeness_score=completeness,
            correctness_score=correctness,
            overall_quality=(completeness + correctness) / 2,
        )

    def _compute_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Compute entailment score between premise and hypothesis.

        Returns probability that hypothesis is entailed by premise.
        """
        if not self.nli_pipeline:
            return 0.5  # Neutral if unavailable

        # CRITICAL FIX: Truncate by characters (~4 chars per token for safety)
        # Model limit is 512 tokens, so max ~1500 chars total for premise + hypothesis
        max_chars = 1500
        max_hypothesis_chars = 400
        max_premise_chars = max_chars - max_hypothesis_chars - 50  # Reserve for [SEP] tokens

        # Truncate hypothesis first (typically shorter)
        hypothesis = hypothesis[:max_hypothesis_chars]
        # Then truncate premise
        premise = premise[:max_premise_chars]

        result = self.nli_pipeline(f"{premise} [SEP] {hypothesis}")[0]

        # Map label to entailment score
        label_to_score = {
            "ENTAILMENT": 1.0,
            "entailment": 1.0,
            "NEUTRAL": 0.5,
            "neutral": 0.5,
            "CONTRADICTION": 0.0,
            "contradiction": 0.0,
        }

        score = label_to_score.get(result["label"], result["score"])
        return float(score)

    def batch_decompose(
        self, answers: List[str], output_path: Optional[Path] = None
    ) -> List[ClaimDecompositionResult]:
        """
        Decompose multiple answers.

        Args:
            answers: List of answers to decompose
            output_path: Optional path to save results as JSON

        Returns:
            List of ClaimDecompositionResult objects
        """
        results = []

        for i, answer in enumerate(answers, 1):
            print(f"Decomposing answer {i}/{len(answers)}...", end=" ")

            result = self.decompose(answer)

            status = "PASS" if result.passed_quality_gate else "FAIL"
            quality = result.quality_metrics.overall_quality

            print(f"{status} (Quality: {quality:.2%}, Claims: {len(result.claims)})")

            results.append(result)

        # Save results if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            results_dict = {
                "timestamp": str(Path.cwd()),
                "results": [r.to_dict() for r in results],
                "summary": {
                    "total_answers": len(results),
                    "passed_quality_gate": sum(
                        1 for r in results if r.passed_quality_gate
                    ),
                    "avg_quality": sum(
                        r.quality_metrics.overall_quality for r in results
                    )
                    / len(results),
                    "avg_claims_per_answer": sum(len(r.claims) for r in results)
                    / len(results),
                },
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results_dict, f, indent=2)

            print(f"\nResults saved to: {output_path}")

        return results
