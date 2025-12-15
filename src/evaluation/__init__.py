"""
Evaluation modules for Legal Hallucination Detector.

Includes:
- Annotation loading from lawyer data package
- Gold/Silver benchmark splitting
- Confidence-weighted metrics
- Threshold tuning on Gold set
- Stratified reporting by evaluability tier
"""

from .annotation_loader import AnnotationLoader
from .gold_silver_splitter import GoldSilverSplitter
from .metrics import ConfidenceWeightedMetrics
from .stratified_reporter import StratifiedReporter
from .threshold_tuner import ThresholdTuner

__all__ = [
    "AnnotationLoader",
    "GoldSilverSplitter",
    "ConfidenceWeightedMetrics",
    "ThresholdTuner",
    "StratifiedReporter",
]
