"""
Configuration management for Legal Hallucination Detector.

Uses Pydantic for type-safe configuration with validation and environment support.
Follows guide Section 5.1 for threshold configuration strategy.
"""

from enum import Enum
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from pydantic import Field
from pydantic_settings import BaseSettings


class GenerationMode(str, Enum):
    """
    Epistemic mode for generation - defines the evidence universe.

    This is critical for academically sound hallucination detection because
    different generation modes have different epistemic assumptions about
    what evidence was available to the model.

    See docs/architecture/WHY_THREE_PIPELINES.md for full rationale.
    """

    NON_RAG = "non_rag"
    """
    Pure parametric generation from model weights.

    Evidence universe: Model's parametric knowledge from training.
    Hallucination definition: Claims not supported by model's training data.
    Verification strategy: Retrieve evidence post-hoc to approximate parametric knowledge.
    """

    RAG = "rag"
    """
    Retrieval-augmented generation with explicit evidence.

    Evidence universe: Retrieved documents provided to the model.
    Hallucination definition: Claims not grounded in retrieved evidence.
    Verification strategy: Verify against the exact evidence shown to the model.
    Assumption: Closed-world - model should ONLY use retrieved evidence.
    """

    UNKNOWN = "unknown"
    """
    Epistemic mode not specified.

    Use unified detector with epistemic ambiguity warnings.
    Not recommended for academic evaluation - always specify mode.
    """


@dataclass
class EpistemicContext:
    """
    Epistemic context for hallucination detection.

    Tracks what evidence was available during generation and how
    the detector should interpret the output.

    Attributes:
        generation_mode: How the answer was generated (RAG vs non-RAG)
        evidence_scope: What evidence was available during generation
        evidence_provided_to_model: Whether retrieved_context was shown to model
        closed_world_assumption: Whether model should ONLY use retrieved evidence
    """

    generation_mode: GenerationMode
    evidence_scope: str  # "parametric", "retrieved", "hybrid"
    evidence_provided_to_model: bool = False
    closed_world_assumption: bool = False


class DetectorConfig(BaseSettings):
    """
    Main configuration for hallucination detector.

    Thresholds are tuned on Gold benchmark (Q1, Q3) and validated on Silver.
    See guide Section 3 for threshold tuning strategy.
    """

    # ========================================
    # NLI Entailment Thresholds
    # ========================================

    nli_entailment_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum entailment score to consider claim grounded. "
        "Lower = more permissive (fewer false positives), "
        "Higher = stricter (more false positives). "
        "Guide recommends 0.5 for legal reasoning, 0.8 for facts.",
    )

    nli_contradiction_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum contradiction score to flag contradiction hallucination. "
        "Guide recommends 0.7 with entailment < 0.5 for contradiction detection.",
    )

    # ========================================
    # Semantic Similarity Thresholds
    # ========================================

    semantic_similarity_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for semantic grounding (Legal-BERT). "
        "Pre-filters evidence before expensive NLI cross-encoder.",
    )

    bi_encoder_top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top evidence sentences to retrieve for NLI verification. "
        "Reduces complexity from O(NxM) to O(N). Guide recommends K=3.",
    )

    # ========================================
    # Citation Matching Thresholds
    # ========================================

    citation_fuzzy_match_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum fuzzy match score for citation equivalence. "
        "Handles variations like 'U.S. v. Garcia' vs 'United States v. Garcia'.",
    )

    citation_require_context: bool = Field(
        default=True,
        description="Require citations to exist in retrieved context. "
        "Set False to allow citations not in RAG sources (e.g., external case law).",
    )

    # ========================================
    # Claim Decomposition
    # ========================================

    claim_decomposition_quality_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for decomposed claims (from detector_package). "
        "Filters out low-quality sentence splits.",
    )

    claim_min_length: int = Field(
        default=10,
        ge=5,
        description="Minimum character length for a valid claim. Filters trivial fragments.",
    )

    # ========================================
    # Calibration
    # ========================================

    calibration_num_bins: int = Field(
        default=10,
        ge=5,
        le=20,
        description="Number of bins for Expected Calibration Error (ECE) computation.",
    )

    max_calibration_error: float = Field(
        default=0.10,
        ge=0.0,
        le=0.5,
        description="Maximum acceptable ECE before raising CalibrationError warning. "
        "Guide recommends 0.10 for well-calibrated models.",
    )

    # ========================================
    # Model Paths
    # ========================================

    bi_encoder_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Bi-encoder model for fast embedding similarity. "
        "Guide alternative: nlpaueb/legal-bert-base-uncased for legal domain.",
    )

    cross_encoder_model: str = Field(
        default="cross-encoder/nli-deberta-v3-base",
        description="Cross-encoder NLI model for accurate entailment verification. "
        "Must output [CONTRADICTION, ENTAILMENT, NEUTRAL] labels.",
    )

    spacy_model: str = Field(
        default="en_core_web_sm",
        description="Spacy model for legal sentence segmentation.",
    )

    # ========================================
    # Performance & Resource Management
    # ========================================

    device: str = Field(
        default="cpu",
        description="Device for model inference ('cpu', 'cuda', 'mps'). "
        "Auto-detected if not specified.",
    )

    max_source_chars: int = Field(
        default=1500,
        ge=500,
        le=5000,
        description="Maximum characters for source documents before truncation. "
        "CRITICAL FIX from detector_package to prevent tensor shape errors. "
        "~375 tokens at 4 chars/token.",
    )

    max_claim_chars: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Maximum characters for claims before truncation. "
        "~125 tokens at 4 chars/token.",
    )

    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for NLI inference. Larger = faster but more memory. "
        "Guide notes 10-20x speedup from batching.",
    )

    max_latency_ms: int = Field(
        default=200,
        ge=50,
        le=5000,
        description="Target P95 latency in milliseconds for 500-token input. "
        "Guide recommends 200ms budget.",
    )

    # ========================================
    # Early Stopping
    # ========================================

    early_stop_on_entailment: bool = Field(
        default=True,
        description="Stop NLI verification early if strong entailment found. "
        "Reduces latency on clearly grounded claims.",
    )

    early_stop_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Entailment score threshold for early stopping. "
        "If entailment > threshold, skip remaining evidence.",
    )

    # ========================================
    # Logging & Telemetry
    # ========================================

    enable_telemetry: bool = Field(
        default=True,
        description="Enable latency and drift metrics collection.",
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    log_format: str = Field(
        default="json",
        description="Log format ('json' for structured logging, 'text' for human-readable).",
    )

    # ========================================
    # Data Paths
    # ========================================

    data_dir: Path = Field(
        default=Path("data"),
        description="Root directory for data files.",
    )

    gold_benchmark_path: Optional[Path] = Field(
        default=None,
        description="Path to Gold benchmark JSON (Q1, Q3). "
        "Defaults to data/gold_standard/gold_benchmark.json.",
    )

    silver_benchmark_path: Optional[Path] = Field(
        default=None,
        description="Path to Silver benchmark JSON (Q2, Q4-Q11, Q14-Q15). "
        "Defaults to data/gold_standard/silver_benchmark.json.",
    )

    lookups_dir: Path = Field(
        default=Path("data/lookups"),
        description="Directory containing citation validation lookups "
        "(us_reporters.json, usc_sections.json, etc.).",
    )

    # ========================================
    # Retrieval Configuration (NEW)
    # ========================================

    retrieval_enabled: bool = Field(
        default=True,
        description="Enable automatic document retrieval for responses without sources. "
        "Applies to both Baseline and RAG+Web variants.",
    )

    retrieval_index_path: Optional[Path] = Field(
        default=None,
        description="Path to pre-built retrieval index directory. "
        "Defaults to data/indexes/legal_corpus.",
    )

    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top passages to retrieve for evidence. "
        "User plan recommends 3-5 ranked passages per claim.",
    )

    retrieval_bm25_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 (sparse) scores in hybrid retrieval. "
        "User plan: BM25 + dense embeddings.",
    )

    retrieval_dense_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for dense embedding scores in hybrid retrieval.",
    )

    retrieval_embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Embedding model for dense retrieval. "
        "Same as bi_encoder_model by default for consistency.",
    )

    # ========================================
    # CourtListener API Fallback
    # ========================================

    enable_api_fallback: bool = Field(
        default=False,
        description="Enable CourtListener API fallback when static corpus insufficient. "
        "Queries API for recent/specialized cases not in corpus.",
    )

    api_fallback_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Min similarity score to skip API query. "
        "If max static corpus similarity < threshold, query CourtListener API.",
    )

    api_fallback_top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of opinions to fetch from CourtListener API as fallback.",
    )

    force_retrieve: bool = Field(
        default=False,
        description="Force retrieval even if sources are provided. "
        "Useful for augmenting existing RAG sources with additional evidence.",
    )

    # ========================================
    # Pydantic Settings Configuration
    # ========================================

    class Config:
        """Pydantic settings configuration."""

        env_file = ".env"
        env_prefix = "DETECTOR_"
        case_sensitive = False

    # ========================================
    # Post-Initialization
    # ========================================

    def model_post_init(self, __context) -> None:
        """Set default paths after initialization."""
        if self.gold_benchmark_path is None:
            self.gold_benchmark_path = (
                self.data_dir / "gold_standard" / "gold_benchmark.json"
            )

        if self.silver_benchmark_path is None:
            self.silver_benchmark_path = (
                self.data_dir / "gold_standard" / "silver_benchmark.json"
            )

        if self.retrieval_index_path is None:
            # Default to oracle corpus if available, otherwise legal_corpus
            oracle_path = self.data_dir / "indexes" / "oracle_corpus"
            legacy_path = self.data_dir / "indexes" / "legal_corpus"
            if oracle_path.exists():
                self.retrieval_index_path = oracle_path
            else:
                self.retrieval_index_path = legacy_path

    # ========================================
    # Validation Methods
    # ========================================

    def validate_thresholds(self) -> None:
        """
        Validate threshold consistency.

        Raises:
            ValueError: If thresholds are inconsistent
        """
        if self.nli_entailment_threshold < 0.5:
            import warnings

            warnings.warn(
                f"NLI entailment threshold ({self.nli_entailment_threshold}) < 0.5 "
                "may produce excessive false negatives. "
                "Guide recommends 0.5-0.8 range.",
                stacklevel=2,
            )

        if self.early_stop_threshold <= self.nli_entailment_threshold:
            import warnings

            warnings.warn(
                f"Early stop threshold ({self.early_stop_threshold}) should be > "
                f"entailment threshold ({self.nli_entailment_threshold}) "
                "to avoid premature stopping.",
                stacklevel=2,
            )

    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        return self.model_dump()

    def __repr__(self) -> str:
        """Return concise representation."""
        return (
            f"DetectorConfig(nli_threshold={self.nli_entailment_threshold}, "
            f"semantic_threshold={self.semantic_similarity_threshold}, "
            f"device={self.device})"
        )
