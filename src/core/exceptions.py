"""
Custom exception hierarchy for hallucination detection.

Provides specific error types for different hallucination categories,
enabling fine-grained error handling and reporting.
"""


class HallucinationError(Exception):
    """Base exception for all hallucination-related errors."""

    def __init__(self, message: str, details: dict = None) -> None:
        """
        Initialize hallucination error.

        Args:
            message: Error message
            details: Optional dict with additional context (score, threshold, etc.)
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation with details."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class StructuralHallucinationError(HallucinationError):
    """
    Raised when structural validation fails.

    Examples:
    - Citation not found in context
    - Fabricated reporter (e.g., "999 F. Fake. 123")
    - Invalid U.S.C. section number
    - Date anachronism (2024 case citing 1969 U.S. decision)
    """

    pass


class SemanticHallucinationError(HallucinationError):
    """
    Raised when semantic grounding fails.

    Examples:
    - Claim embeddings too dissimilar from evidence (cosine < threshold)
    - No relevant evidence found in retrieved documents
    """

    pass


class ContradictionHallucinationError(HallucinationError):
    """
    Raised when NLI detects contradiction.

    Examples:
    - Generated: "Speed limit is 100mph"
    - Evidence: "The max speed is 65mph"
    - NLI: CONTRADICTION (score > threshold)
    """

    pass


class CalibrationError(Exception):
    """Raised when model calibration is poor (ECE too high)."""

    def __init__(self, message: str, ece: float, threshold: float) -> None:
        """
        Initialize calibration error.

        Args:
            message: Error message
            ece: Expected Calibration Error value
            threshold: Maximum acceptable ECE
        """
        super().__init__(message)
        self.ece = ece
        self.threshold = threshold

    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.args[0]} (ECE={self.ece:.3f}, threshold={self.threshold:.3f})"


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


class ModelLoadError(Exception):
    """Raised when model loading fails."""

    pass
