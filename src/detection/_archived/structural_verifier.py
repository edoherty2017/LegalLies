"""
Structural verification layer for fast-fail hallucination detection.

Performs lightweight checks before expensive semantic/NLI verification:
1. Citation extraction and normalization
2. Context cross-check (citations exist in retrieved documents)
3. Citation structure validation (blocklist, fake reporters)
4. Date consistency (anachronism detection)

Follows guide Section 3.1 for structural verification strategy.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

from ..core.exceptions import StructuralHallucinationError
from ..preprocessing.citation_normalizer import CitationNormalizer, CitationObj


@dataclass
class StructuralVerificationResult:
    """Result of structural verification."""

    is_valid: bool
    failed_citations: List[CitationObj]
    failure_reason: Optional[str] = None
    details: dict = None

    def __post_init__(self) -> None:
        """Initialize details if not provided."""
        if self.details is None:
            self.details = {}


class StructuralVerifier:
    """
    Fast-fail structural verification layer.

    Priority: Citation Extraction -> Context Cross-Check -> Structure Validation -> Date Check
    """

    def __init__(
        self,
        lookups_dir: Optional[Path] = None,
        require_context: bool = True,
        fuzzy_threshold: float = 0.85,
    ) -> None:
        """
        Initialize structural verifier.

        Args:
            lookups_dir: Directory containing validation lookups (us_reporters.json, etc.)
            require_context: Require citations to exist in retrieved context
            fuzzy_threshold: Minimum fuzzy match score for citation equivalence
        """
        self.citation_normalizer = CitationNormalizer()
        self.require_context = require_context
        self.fuzzy_threshold = fuzzy_threshold

        # Load lookup tables
        self.lookups_dir = lookups_dir or Path("data/lookups")
        self.valid_reporters = self._load_reporters()
        self.blocklist_patterns = self._load_blocklist()

    def _load_reporters(self) -> Set[str]:
        """
        Load valid reporter abbreviations from lookup file.

        Returns:
            Set of valid reporter strings (e.g., "U.S.", "F.3d")
        """
        reporters_file = self.lookups_dir / "us_reporters.json"
        if not reporters_file.exists():
            # Use built-in defaults from CitationNormalizer
            return self.citation_normalizer.KNOWN_REPORTERS

        try:
            with open(reporters_file, "r") as f:
                data = json.load(f)
                return set(data.get("reporters", []))
        except Exception as e:
            print(f"Warning: Failed to load reporters from {reporters_file}: {e}")
            return self.citation_normalizer.KNOWN_REPORTERS

    def _load_blocklist(self) -> List[re.Pattern]:
        """
        Load blocklist patterns for fake citations.

        Returns:
            List of compiled regex patterns
        """
        blocklist_file = self.lookups_dir / "citation_blocklist.json"
        if not blocklist_file.exists():
            # Default blocklist patterns
            return [
                re.compile(r"\d+\s+F\.\s*Fake", re.IGNORECASE),
                re.compile(r"\d+\s+Fictional\s+Reporter", re.IGNORECASE),
                re.compile(r"999\s+[A-Z]", re.IGNORECASE),  # Suspicious volume number
            ]

        try:
            with open(blocklist_file, "r") as f:
                data = json.load(f)
                patterns = data.get("patterns", [])
                return [re.compile(p, re.IGNORECASE) for p in patterns]
        except Exception as e:
            print(f"Warning: Failed to load blocklist from {blocklist_file}: {e}")
            return []

    def verify(
        self,
        generated_text: str,
        retrieved_context: List[str],
        current_year: Optional[int] = None,
    ) -> StructuralVerificationResult:
        """
        Perform structural verification on generated text.

        Args:
            generated_text: Generated answer to verify
            retrieved_context: List of retrieved source documents
            current_year: Current year for anachronism detection (default: now)

        Returns:
            StructuralVerificationResult with validation status

        Raises:
            StructuralHallucinationError: If verification fails with strict mode
        """
        if current_year is None:
            current_year = datetime.now().year

        # Step 1: Extract citations from generated text
        generated_citations = self.citation_normalizer.extract_citations(
            generated_text
        )

        if not generated_citations:
            # No citations to verify
            return StructuralVerificationResult(
                is_valid=True, failed_citations=[], details={"num_citations": 0}
            )

        failed_citations = []

        # Step 2: Validate citation structure
        for citation in generated_citations:
            # Check blocklist
            if self._is_blocklisted(citation):
                failed_citations.append(citation)
                continue

            # Validate structure
            if not self.citation_normalizer.validate_citation_structure(citation):
                failed_citations.append(citation)
                continue

        # Step 3: Context cross-check (if required)
        if self.require_context and retrieved_context:
            for citation in generated_citations:
                if citation in failed_citations:
                    continue  # Already failed

                if not self._citation_in_context(citation, retrieved_context):
                    failed_citations.append(citation)

        # Step 4: Date consistency check
        for citation in generated_citations:
            if citation in failed_citations:
                continue

            if citation.year and citation.year > current_year:
                # Future citation - anachronism
                failed_citations.append(citation)

        # Prepare result
        is_valid = len(failed_citations) == 0

        result = StructuralVerificationResult(
            is_valid=is_valid,
            failed_citations=failed_citations,
            failure_reason=self._get_failure_reason(failed_citations)
            if not is_valid
            else None,
            details={
                "num_citations": len(generated_citations),
                "num_failed": len(failed_citations),
                "failed_citation_texts": [c.original_text for c in failed_citations],
            },
        )

        return result

    def _is_blocklisted(self, citation: CitationObj) -> bool:
        """
        Check if citation matches blocklist patterns.

        Args:
            citation: Citation to check

        Returns:
            True if citation is blocklisted
        """
        for pattern in self.blocklist_patterns:
            if pattern.search(citation.original_text):
                return True
        return False

    def _citation_in_context(
        self, citation: CitationObj, context: List[str]
    ) -> bool:
        """
        Check if citation exists in retrieved context using fuzzy matching.

        Args:
            citation: Citation to search for
            context: List of context documents

        Returns:
            True if citation found in context
        """
        # Combine all context into single text
        combined_context = " ".join(context)

        # Use CitationNormalizer's fuzzy match method
        return self.citation_normalizer.is_citation_in_context(
            citation, combined_context, threshold=self.fuzzy_threshold
        )

    def _get_failure_reason(self, failed_citations: List[CitationObj]) -> str:
        """
        Generate human-readable failure reason.

        Args:
            failed_citations: List of citations that failed validation

        Returns:
            Failure reason string
        """
        if not failed_citations:
            return ""

        reasons = []

        # Check for blocklist failures
        blocklisted = [c for c in failed_citations if self._is_blocklisted(c)]
        if blocklisted:
            reasons.append(
                f"Blocklisted citations: {', '.join(c.original_text for c in blocklisted)}"
            )

        # Check for structure failures
        struct_failed = [
            c
            for c in failed_citations
            if not self.citation_normalizer.validate_citation_structure(c)
        ]
        if struct_failed:
            reasons.append(
                f"Invalid structure: {', '.join(c.original_text for c in struct_failed)}"
            )

        # Default reason
        if not reasons:
            reasons.append(
                f"Failed citations: {', '.join(c.original_text for c in failed_citations)}"
            )

        return "; ".join(reasons)

    def verify_strict(
        self,
        generated_text: str,
        retrieved_context: List[str],
        current_year: Optional[int] = None,
    ) -> StructuralVerificationResult:
        """
        Perform strict structural verification (raises exception on failure).

        Args:
            generated_text: Generated answer to verify
            retrieved_context: List of retrieved source documents
            current_year: Current year for anachronism detection

        Returns:
            StructuralVerificationResult if valid

        Raises:
            StructuralHallucinationError: If verification fails
        """
        result = self.verify(generated_text, retrieved_context, current_year)

        if not result.is_valid:
            raise StructuralHallucinationError(
                message=f"Structural verification failed: {result.failure_reason}",
                details=result.details,
            )

        return result

    def extract_and_validate_citations(
        self, text: str
    ) -> tuple[List[CitationObj], List[CitationObj]]:
        """
        Extract citations and separate valid from invalid.

        Args:
            text: Text to extract citations from

        Returns:
            Tuple of (valid_citations, invalid_citations)
        """
        citations = self.citation_normalizer.extract_citations(text)

        valid = []
        invalid = []

        for citation in citations:
            # Check blocklist
            if self._is_blocklisted(citation):
                invalid.append(citation)
                continue

            # Validate structure
            if not self.citation_normalizer.validate_citation_structure(citation):
                invalid.append(citation)
                continue

            valid.append(citation)

        return valid, invalid
