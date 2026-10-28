"""
Legal Corpus Processing Modules.

This package implements the legal-specific document processing strategies
described in the Pile of Law framework, including:
- Semantic chunking with context preservation
- Statutory breadcrumb injection
- Dissent/opinion type tagging
- OCR correction
- PII redaction

References:
- Section 2: Corpus Composition - Pile of Law
- Section 4: Vectorization and Embedding Strategy
- Section 6: Deep Dive into Pile of Law Components
- Section 11: Detailed Corpus Curation
"""

from src.corpus.legal_chunker import LegalChunker, LegalChunk, OpinionType, SectionType
from src.corpus.statutory_parser import (
    StatutoryParser,
    StatutorySection,
    StatuteType,
)
from src.corpus.ocr_corrector import OCRCorrector, OCRError
from src.corpus.pii_redactor import PIIRedactor, RedactionSpan, PIIType, CaseType

__all__ = [
    # Legal chunking
    "LegalChunker",
    "LegalChunk",
    "OpinionType",
    "SectionType",
    # Statutory parsing
    "StatutoryParser",
    "StatutorySection",
    "StatuteType",
    # OCR correction
    "OCRCorrector",
    "OCRError",
    # PII redaction
    "PIIRedactor",
    "RedactionSpan",
    "PIIType",
    "CaseType",
]
