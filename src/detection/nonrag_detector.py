"""
NonRAG Detector - Thin wrapper around FactualityDetector for non-RAG outputs.

For non-RAG systems, the model generated from parametric knowledge.
This wrapper uses auto-retrieval from the oracle corpus for verification.

Note: The factuality detector treats RAG and non-RAG identically.
Both are evaluated against the oracle corpus for contradictions.
"""

from typing import List, Optional

from ..core.config import DetectorConfig
from .unified_detector import FactualityDetector, FactualityResult


class NonRAGHallucinationDetector:
    """
    Non-RAG wrapper around FactualityDetector.

    Uses auto-retrieval from oracle corpus to find evidence for verification.
    The underlying factuality check is identical to RAG.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()

        # Ensure retrieval is enabled for auto-retrieval
        if not self.config.retrieval_enabled:
            self.config.retrieval_enabled = True

        self.detector = FactualityDetector(config=self.config)

    def detect(
        self,
        query: str,
        generated_answer: str,
    ) -> FactualityResult:
        """
        Detect contradictions in non-RAG generated answer.

        Args:
            query: Original question
            generated_answer: Answer from parametric generation

        Returns:
            FactualityResult with per-claim verdicts
        """
        # Auto-retrieve evidence from oracle corpus
        return self.detector.detect(
            query=query,
            answer=generated_answer,
            evidence=None,  # Triggers auto-retrieval
        )


def create_nonrag_detector(config: Optional[DetectorConfig] = None) -> NonRAGHallucinationDetector:
    """Factory function to create non-RAG detector."""
    return NonRAGHallucinationDetector(config=config)
