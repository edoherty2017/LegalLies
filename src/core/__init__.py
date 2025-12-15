"""
Core configuration and exceptions for Legal Hallucination Detector.
"""

from .config import DetectorConfig
from .exceptions import (
    ContradictionHallucinationError,
    HallucinationError,
    SemanticHallucinationError,
    StructuralHallucinationError,
)

__all__ = [
    "DetectorConfig",
    "HallucinationError",
    "StructuralHallucinationError",
    "SemanticHallucinationError",
    "ContradictionHallucinationError",
]
