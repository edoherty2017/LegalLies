"""
RAG Detector - wrapper around FactualityDetector for RAG outputs.

For RAG systems the model was shown evidence during generation. Two different
questions can be asked about the resulting answer, and conflating them was a
defect:

* **Factual correctness** - is the claim true, judged against authoritative law?
  Evidence: the fixed oracle corpus. This is the question the project's metric
  asks, and the only one that is comparable across generation modes.
* **Faithfulness (groundedness)** - does the claim follow from the passages the
  model was actually shown? Evidence: those passages. A useful question, but a
  different one: an answer can be perfectly faithful to a retrieved passage that
  is wrong or off-topic.

This wrapper previously always did the second while the docstring and the README
claimed the first. Because FactualityDetector only auto-retrieves from the oracle
corpus when no evidence is passed, and this wrapper always passed the retrieved
context, RAG answers were never checked against the corpus at all - so RAG and
non-RAG hallucination rates were computed from different evidence sets and could
not legitimately be compared.

Correctness grading is now the default. Faithfulness is still available, but you
have to ask for it.
"""

from typing import List, Optional

from ..core.config import DetectorConfig
from .unified_detector import FactualityDetector, FactualityResult

# Grade against authoritative law (comparable across modes) vs. against the
# passages the generator was shown (faithfulness only).
GRADE_CORPUS = "corpus"
GRADE_CONTEXT = "context"


class RAGHallucinationDetector:
    """
    RAG-specific wrapper around FactualityDetector.

    By default this grades claims against the oracle corpus, exactly as the
    non-RAG wrapper does, so that RAG and non-RAG rates are measured on the same
    terms. Pass ``grade_against="context"`` to measure faithfulness to the
    retrieved passages instead.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()

        # Corpus grading needs retrieval, same as the non-RAG path.
        if not self.config.retrieval_enabled:
            self.config.retrieval_enabled = True

        self.detector = FactualityDetector(config=self.config)

    def detect(
        self,
        query: str,
        generated_answer: str,
        retrieved_context: List[str],
        grade_against: str = GRADE_CORPUS,
    ) -> FactualityResult:
        """
        Detect contradictions in a RAG-generated answer.

        Args:
            query: Original question.
            generated_answer: Answer from the RAG system.
            retrieved_context: Evidence shown to the model during generation.
                Required, because it records what the generator saw, and used as
                the evidence set only when grade_against="context".
            grade_against: "corpus" (default) grades each claim against the
                oracle corpus, which is what makes the result comparable with
                non-RAG answers. "context" grades against retrieved_context,
                which measures faithfulness to the retrieved passages and is
                NOT comparable with non-RAG rates.

        Returns:
            FactualityResult with per-claim verdicts.

        Raises:
            ValueError: if retrieved_context is empty, or grade_against is not
                one of the two supported values.
        """
        if not retrieved_context:
            raise ValueError("RAG detector requires retrieved_context")

        if grade_against == GRADE_CORPUS:
            # evidence=None triggers per-claim oracle-corpus retrieval, the same
            # path the non-RAG wrapper uses.
            evidence = None
        elif grade_against == GRADE_CONTEXT:
            evidence = retrieved_context
        else:
            raise ValueError(
                f"grade_against must be {GRADE_CORPUS!r} (factual correctness, "
                f"comparable across modes) or {GRADE_CONTEXT!r} (faithfulness to "
                f"the retrieved passages); got {grade_against!r}"
            )

        return self.detector.detect(
            query=query,
            answer=generated_answer,
            evidence=evidence,
        )


def create_rag_detector(config: Optional[DetectorConfig] = None) -> RAGHallucinationDetector:
    """Factory function to create RAG detector."""
    return RAGHallucinationDetector(config=config)
