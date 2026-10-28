"""
Citation normalization using eyecite + fuzzy matching.

Handles legal citation extraction and normalization for:
- U.S. Code (e.g., "17 U.S.C. Section 107")
- Federal Reporters (e.g., "394 U.S. 705", "123 F.3d 456")
- Code of Federal Regulations (e.g., "17 C.F.R. Section 240.10b-5")

Combines eyecite library with fallback regex and fuzzy matching to handle
LLM formatting inconsistencies (extra spaces, missing periods, etc.).
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from rapidfuzz import fuzz

# Try to import eyecite, but don't fail if unavailable
try:
    import eyecite
    EYECITE_AVAILABLE = True
except ImportError:
    EYECITE_AVAILABLE = False
    print("Warning: eyecite not installed. Using regex-only citation extraction.")


@dataclass
class CitationObj:
    """Normalized legal citation object."""

    original_text: str
    citation_type: str  # "usc", "federal_reporter", "cfr", "case", "unknown"
    volume: Optional[str] = None
    reporter: Optional[str] = None
    page: Optional[str] = None
    year: Optional[int] = None
    section: Optional[str] = None
    normalized_string: str = ""

    def __post_init__(self) -> None:
        """Generate normalized string if not provided."""
        if not self.normalized_string:
            self.normalized_string = self._build_normalized()

    def _build_normalized(self) -> str:
        """Build canonical normalized citation string."""
        if self.citation_type == "usc":
            return f"{self.volume} U.S.C. Section {self.section}"
        elif self.citation_type == "cfr":
            return f"{self.volume} C.F.R. Section {self.section}"
        elif self.citation_type == "federal_reporter":
            return f"{self.volume} {self.reporter} {self.page}"
        elif self.citation_type == "case":
            # For case citations from eyecite
            return self.original_text
        else:
            return self.original_text


class CitationNormalizer:
    """
    Extract and normalize legal citations using hybrid approach.

    Priority: Eyecite -> Fallback Regex -> Context Resolution
    """

    # Regex patterns for fallback extraction
    SECTION_KEYWORDS = r"(?:section|sec\.?)"
    USC_PATTERN = (
        rf"(\d+)\s+U\.?S\.?C\.?\s+(?:{SECTION_KEYWORDS}\s*)?"
        rf"(\d+(?:[a-z])?(?:\(\d+\))?)"
    )
    CFR_PATTERN = (
        rf"(\d+)\s+C\.?F\.?R\.?\s+(?:pt\.?|part|{SECTION_KEYWORDS})?"
        rf"\s*(\d+(?:\.\d+)?)"
    )
    FEDERAL_REPORTER_PATTERN = r"(\d+)\s+([A-Za-z0-9\.\s]+?)\s+(\d+)"

    # Known reporter abbreviations for validation
    KNOWN_REPORTERS = {
        "U.S.",
        "F.3d",
        "F.2d",
        "F.",
        "F. Supp.",
        "F. Supp. 2d",
        "F. Supp. 3d",
        "S. Ct.",
        "L. Ed.",
        "L. Ed. 2d",
    }

    def __init__(self) -> None:
        """Initialize citation normalizer."""
        self.use_eyecite = EYECITE_AVAILABLE

    def extract_citations(self, text: str) -> List[CitationObj]:
        """
        Extract all citations from text using hybrid approach.

        Args:
            text: Text to extract citations from

        Returns:
            List of CitationObj instances
        """
        citations = []

        # Try eyecite first if available
        if self.use_eyecite:
            eyecite_citations = self._extract_with_eyecite(text)
            citations.extend(eyecite_citations)

        # Fallback regex extraction
        regex_citations = self._extract_with_regex(text)

        # Merge, avoiding duplicates
        for regex_cit in regex_citations:
            # Check if already found by eyecite (fuzzy match on normalized string)
            is_duplicate = any(
                fuzz.ratio(regex_cit.normalized_string, cit.normalized_string) > 90
                for cit in citations
            )
            if not is_duplicate:
                citations.append(regex_cit)

        return citations

    def _extract_with_eyecite(self, text: str) -> List[CitationObj]:
        """Extract citations using eyecite library."""
        if not EYECITE_AVAILABLE:
            return []

        try:
            found_citations = eyecite.get_citations(text)
            return [
                CitationObj(
                    original_text=str(cit),
                    citation_type="case",
                    normalized_string=str(cit),
                )
                for cit in found_citations
            ]
        except Exception as e:
            print(f"Warning: eyecite extraction failed: {e}")
            return []

    def _extract_with_regex(self, text: str) -> List[CitationObj]:
        """Extract citations using regex patterns."""
        citations = []

        # U.S. Code citations
        for match in re.finditer(self.USC_PATTERN, text, re.IGNORECASE):
            citations.append(
                CitationObj(
                    original_text=match.group(0),
                    citation_type="usc",
                    volume=match.group(1),
                    section=match.group(2),
                )
            )

        # C.F.R. citations
        for match in re.finditer(self.CFR_PATTERN, text, re.IGNORECASE):
            citations.append(
                CitationObj(
                    original_text=match.group(0),
                    citation_type="cfr",
                    volume=match.group(1),
                    section=match.group(2),
                )
            )

        # Federal Reporter citations
        for match in re.finditer(self.FEDERAL_REPORTER_PATTERN, text, re.IGNORECASE):
            reporter = match.group(2).strip()
            # Only accept if reporter looks valid
            if self._is_valid_reporter(reporter):
                citations.append(
                    CitationObj(
                        original_text=match.group(0),
                        citation_type="federal_reporter",
                        volume=match.group(1),
                        reporter=self._normalize_reporter(reporter),
                        page=match.group(3),
                    )
                )

        return citations

    def _is_valid_reporter(self, reporter: str) -> bool:
        """Check if reporter abbreviation is valid."""
        # Normalize spacing and periods
        normalized = reporter.replace(" ", "").replace(".", "").upper()

        # Check against known reporters (also normalized)
        for known in self.KNOWN_REPORTERS:
            known_normalized = known.replace(" ", "").replace(".", "").upper()
            if normalized == known_normalized:
                return True

        # Also accept if it looks like a reporter pattern (F.3d, F.2d, etc.)
        if re.match(r"^[A-Z]\.?\s*(?:\d[a-z]d?)?$", reporter, re.IGNORECASE):
            return True

        return False

    def _normalize_reporter(self, reporter: str) -> str:
        """Normalize reporter abbreviation to canonical form."""
        # Remove extra spaces
        reporter = " ".join(reporter.split())

        # Map common variations to canonical forms
        reporter_map = {
            "F 3d": "F.3d",
            "F3d": "F.3d",
            "F 2d": "F.2d",
            "F2d": "F.2d",
            "F Supp": "F. Supp.",
            "FSupp": "F. Supp.",
            "US": "U.S.",
        }

        # Normalize
        key = reporter.replace(".", "").replace(" ", "")
        return reporter_map.get(key, reporter)

    def normalize_case_name(self, name: str) -> str:
        """
        Normalize case name for fuzzy matching.

        Handles variations like:
        - "U.S." vs "United States"
        - "Inc." vs "Incorporated"
        - Whitespace normalization
        """
        replacements = {
            "U.S.": "United States",
            "U. S.": "United States",
            "Inc.": "Incorporated",
            "Corp.": "Corporation",
            "Ltd.": "Limited",
            "Co.": "Company",
            "v.": "versus",
        }

        normalized = name
        for abbr, full in replacements.items():
            normalized = normalized.replace(abbr, full)

        # Normalize whitespace
        normalized = " ".join(normalized.split())

        return normalized.strip().lower()

    def fuzzy_match_citations(self, citation1: str, citation2: str) -> float:
        """
        Compute fuzzy match score between two citations.

        Returns score 0-1, where 1.0 = exact match
        """
        norm1 = self.normalize_case_name(citation1)
        norm2 = self.normalize_case_name(citation2)

        return fuzz.ratio(norm1, norm2) / 100.0

    def is_citation_in_context(
        self, citation: CitationObj, context: str, threshold: float = 0.85
    ) -> bool:
        """
        Check if citation exists in context using fuzzy matching.

        Args:
            citation: CitationObj to check
            context: Context text to search in
            threshold: Fuzzy match threshold (default: 0.85)

        Returns:
            True if citation found in context above threshold
        """
        # Extract all citations from context
        context_citations = self.extract_citations(context)

        # Check for fuzzy matches
        for context_cit in context_citations:
            match_score = self.fuzzy_match_citations(
                citation.normalized_string, context_cit.normalized_string
            )
            if match_score >= threshold:
                return True

        return False

    def validate_citation_structure(self, citation: CitationObj) -> bool:
        """
        Validate that citation has proper structure.

        Checks for:
        - Fabricated reporters (e.g., "999 F. Fake. 123")
        - Invalid section numbers
        - Missing required components
        """
        if citation.citation_type == "federal_reporter":
            # Check reporter is known
            if citation.reporter not in self.KNOWN_REPORTERS:
                # Use fuzzy matching against known reporters
                max_similarity = max(
                    fuzz.ratio(citation.reporter or "", known)
                    for known in self.KNOWN_REPORTERS
                )
                if max_similarity < 70:  # Less than 70% similar
                    return False

        elif citation.citation_type in ["usc", "cfr"]:
            # Check section exists (basic validation)
            if not citation.section:
                return False

        return True
