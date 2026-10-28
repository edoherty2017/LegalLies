"""
RAG Detector - Thin wrapper around FactualityDetector for RAG outputs.

For RAG systems, the model was shown evidence during generation.
This wrapper simply passes that evidence to the factuality detector.

Note: The factuality detector treats RAG and non-RAG identically.
Both are evaluated against the oracle corpus for contradictions.
"""

from typing import List, Optional

from ..core.config import DetectorConfig
from .unified_detector import FactualityDetector, FactualityResult


class RAGHallucinationDetector:
    """
    RAG-specific wrapper around FactualityDetector.

    Simply passes the retrieved context as evidence for NLI verification.
    The underlying factuality check is identical to non-RAG.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.detector = FactualityDetector(config=self.config)

    def detect(
        self,
        query: str,
        generated_answer: str,
        retrieved_context: List[str],
    ) -> FactualityResult:
        """
        Detect contradictions in RAG-generated answer.

        Args:
            query: Original question
            generated_answer: Answer from RAG system
            retrieved_context: Evidence shown to model during generation

        Returns:
            FactualityResult with per-claim verdicts
        """
        if not retrieved_context:
            raise ValueError("RAG detector requires retrieved_context")

        return self.detector.detect(
            query=query,
            answer=generated_answer,
            evidence=retrieved_context,
        )


def create_rag_detector(config: Optional[DetectorConfig] = None) -> RAGHallucinationDetector:
    """Factory function to create RAG detector."""
    return RAGHallucinationDetector(config=config)
