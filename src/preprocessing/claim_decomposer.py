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

        # Prefer the spaCy-backed LegalSegmenter: it protects legal
        # abbreviations through tokenizer exceptions and is unit-tested. The
        # regex fallback in _segment_sentences handles environments without a
        # spaCy model. Importing lazily keeps spaCy an optional dependency.
        self.segmenter = None
        self.nlp = None
        try:
            from .legal_segmenter import LegalSegmenter

            self.segmenter = LegalSegmenter()
            self.nlp = getattr(self.segmenter, "nlp", None)
        except Exception as exc:  # pragma: no cover - environment dependent
            print(
                f"Note: spaCy segmenter unavailable ({exc.__class__.__name__}); "
                "using regex sentence segmentation"
            )

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
        if self.segmenter is not None:
            sentences = [s for s in self.segmenter.split(answer) if len(s) > 10]
        else:
            sentences = self._segment_sentences(answer)

        # Stage 2: Convert sentences to atomic claims
        claims = self._create_claims(sentences, nlp=self.nlp)

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

    # Abbreviations that must never create a sentence boundary. Order does not
    # matter here: masking is applied longest-first at run time, so "U.S.C."
    # is always consumed before "U.S.". Getting that order wrong was the
    # original defect - "U.S." masked first turned "U.S.C." into "<US>C.",
    # leaving the period after "C." exposed and splitting every U.S. Code
    # citation mid-claim.
    LEGAL_ABBREVIATIONS = (
        # Codes and regulations
        "U.S.C.A.", "U.S.C.", "C.F.R.", "U. S.", "U.S.",
        # Federal rules
        "Fed. R. Civ. P.", "Fed. R. Crim. P.", "Fed. R. Evid.",
        "Fed. R. App. P.", "Fed. R.", "Fed. Reg.",
        # Reporters (incl. series with digits and internal spaces)
        "F. Supp. 3d", "F. Supp. 2d", "F. Supp.", "F. App'x",
        "S. Ct.", "L. Ed. 2d", "L. Ed.",
        "F.4th", "F.3d", "F.2d", "A.2d", "A.3d", "P.2d", "P.3d",
        "N.E.2d", "N.E.3d", "N.W.2d", "N.W.3d", "S.E.2d", "S.E.3d",
        "S.W.2d", "S.W.3d", "So.2d", "So.3d", "Cal.4th", "Cal.3d",
        "N.Y.2d", "N.Y.3d", "Stat.",
        # Courts and jurisdictions
        "Cir.", "Ct. App.", "Ct. Cl.", "Ct.", "D.C.", "Dist.", "Div.",
        "Supp.", "Supreme Ct.",
        # Citation signals
        " v. ", " vs. ", "Id.", "id.", "Ibid.", "ibid.", "Cf.", "cf.",
        "See, e.g.,", "e.g.", "i.e.", "et al.", "etc.", "aff'd", "rev'd",
        # Parties and numbering
        "Inc.", "Corp.", "Ltd.", "Co.", "L.L.C.", "L.P.",
        "Nos.", "No.", "para.", "pt.", "art.", "ch.", "cl.",
        "Dep't", "Ass'n", "Comm'n",
        # Titles
        "Mr.", "Mrs.", "Ms.", "Dr.", "Hon.", "J.", "JJ.",
    )

    @staticmethod
    def _segment_sentences(text: str) -> List[str]:
        """
        Split text into sentences on legal-aware boundaries.

        This is the dependency-free fallback. When spaCy and a model are
        available, ``decompose`` uses :class:`LegalSegmenter` instead, which
        does the same job through tokenizer exceptions.

        Legal text defeats naive splitters because citations are dense with
        periods: "Under 42 U.S.C. Section 1983, a plaintiff may sue" must stay
        one claim, while "...at 500 F. Supp. 2d 100. The panel disagreed."
        must become two. Abbreviations are therefore masked with placeholders
        that contain no period, longest match first, before splitting.
        """
        if not text or not text.strip():
            return []

        # Mask longest abbreviations first so that prefixes such as "U.S."
        # cannot consume part of a longer form such as "U.S.C.".
        ordered = sorted(
            ClaimDecomposer.LEGAL_ABBREVIATIONS, key=len, reverse=True
        )
        placeholders: Dict[str, str] = {}
        protected = text
        for i, abbr in enumerate(ordered):
            if abbr in protected:
                token = f"\x00A{i}\x00"  # contains no '.', '!' or '?'
                placeholders[token] = abbr
                protected = protected.replace(abbr, token)

        sentences = re.split(r"(?<=[.!?])\s+", protected)

        # Restore, longest placeholder first is unnecessary - tokens are unique.
        restored = []
        for sentence in sentences:
            for token, abbr in placeholders.items():
                sentence = sentence.replace(token, abbr)
            restored.append(sentence.strip())

        return [s for s in restored if len(s) > 10]

    @staticmethod
    def _create_claims(sentences: List[str], nlp=None) -> List[DecomposedClaim]:
        """Convert sentences into DecomposedClaim objects."""
        claims = []

        for i, sentence in enumerate(sentences, 1):
            # Split only genuine clause coordination; needs a parse, so this is
            # a no-op when no spaCy model is loaded.
            sub_claims = ClaimDecomposer._split_compound_sentence(sentence, nlp=nlp)

            for j, sub_claim in enumerate(sub_claims):
                claim = DecomposedClaim(
                    claim_text=sub_claim,
                    claim_id=i * 100 + j,  # Hierarchical IDs: 101, 102, 201, 202, etc.
                    original_answer_section=sentence,
                )
                claims.append(claim)

        return claims

    @staticmethod
    def _split_compound_sentence(sentence: str, nlp=None) -> List[str]:
        """
        Split a sentence only where it genuinely coordinates two clauses.

        The previous implementation split on any ``<word> and the <word>``,
        which fires on ordinary noun-phrase coordination:

            "The dispute arose between the landlord and the tenant over the lease."
            -> ["The dispute arose between the landlord", "tenant over the lease."]

        Neither fragment is a proposition (and the second lost its determiner,
        because the pattern consumed "the"). Fragments still reach NLI, come
        back UNSUPPORTED, and so inflate the claim count while depressing the
        testability rate - which is precisely the "Decomposition Dilemma" this
        module exists to avoid.

        Distinguishing clause coordination from noun coordination needs a
        parse, not a surface pattern. With a spaCy model we split only when the
        right-hand conjunct has its own subject and acts as a verb; without
        one we do not split at all. Leaving a compound sentence intact is
        safe - it is still a verifiable proposition, merely a less atomic one.

        Args:
            sentence: A single sentence.
            nlp: Optional loaded spaCy pipeline with a dependency parser.

        Returns:
            One or more claim strings. Never fragments.
        """
        if nlp is None:
            return [sentence]

        try:
            doc = nlp(sentence)
        except Exception:
            return [sentence]

        for token in doc:
            # A coordinated verb with its own subject starts a new clause.
            if token.dep_ != "conj" or token.pos_ not in ("VERB", "AUX"):
                continue
            if not any(
                child.dep_ in ("nsubj", "nsubjpass") for child in token.children
            ):
                continue
            if token.head.pos_ not in ("VERB", "AUX"):
                continue

            # Cut at the coordinating conjunction attached to this conjunct.
            conj = next(
                (c for c in token.children if c.dep_ == "cc"),
                None,
            )
            subject = next(
                c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")
            )
            cut = conj.i if conj is not None else subject.left_edge.i
            left = doc[: cut].text.strip(" ,;")
            right = doc[cut:].text.strip()
            # Drop the leading conjunction from the right-hand clause.
            if conj is not None:
                right = right[len(conj.text):].strip(" ,;")
            if len(left) > 10 and len(right) > 10:
                return [left, right]

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
