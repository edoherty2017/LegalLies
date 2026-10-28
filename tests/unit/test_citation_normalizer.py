"""
Unit tests for citation normalizer.
"""

import pytest

from src.preprocessing.citation_normalizer import CitationNormalizer, CitationObj


class TestCitationNormalizer:
    """Tests for CitationNormalizer."""

    @pytest.fixture
    def normalizer(self) -> CitationNormalizer:
        """Create citation normalizer."""
        return CitationNormalizer()

    def test_extract_usc_citation(self, normalizer: CitationNormalizer) -> None:
        """Test U.S. Code citation extraction."""
        text = "This is covered under 17 U.S.C. Section 107."

        citations = normalizer.extract_citations(text)

        assert len(citations) > 0, "Failed to extract U.S.C. citation"

        # Check if any citation matches expected pattern
        usc_found = any(
            c.citation_type == "usc" and c.volume == "17" and c.section == "107"
            for c in citations
        )
        assert usc_found, "U.S.C. citation not properly extracted"

    def test_extract_cfr_citation(self, normalizer: CitationNormalizer) -> None:
        """Test C.F.R. citation extraction."""
        text = "See 17 C.F.R. Section 240.10b-5 for regulations."

        citations = normalizer.extract_citations(text)

        assert len(citations) > 0, "Failed to extract C.F.R. citation"

        cfr_found = any(
            c.citation_type == "cfr" and c.volume == "17"
            for c in citations
        )
        assert cfr_found, "C.F.R. citation not properly extracted"

    def test_extract_federal_reporter(self, normalizer: CitationNormalizer) -> None:
        """Test Federal Reporter citation extraction."""
        text = "The case is 394 U.S. 705."

        citations = normalizer.extract_citations(text)

        assert len(citations) > 0, "Failed to extract Federal Reporter citation"

    def test_normalize_case_name(self, normalizer: CitationNormalizer) -> None:
        """Test case name normalization."""
        # Test U.S. expansion
        normalized = normalizer.normalize_case_name("U.S. v. Garcia")
        assert "united states" in normalized.lower()
        assert "garcia" in normalized.lower()

        # Test Inc. expansion
        normalized = normalizer.normalize_case_name("Apple Inc. v. Samsung")
        assert "incorporated" in normalized.lower()

    def test_fuzzy_match_citations(self, normalizer: CitationNormalizer) -> None:
        """Test fuzzy matching of citations."""
        citation1 = "U.S. v. Garcia"
        citation2 = "United States v. Garcia"

        score = normalizer.fuzzy_match_citations(citation1, citation2)

        # Should have high similarity despite abbreviation
        assert score > 0.7, f"Fuzzy match score too low: {score}"

    def test_citation_obj_normalization(self) -> None:
        """Test CitationObj normalization."""
        # U.S.C. citation
        citation = CitationObj(
            original_text="17 USC 107",
            citation_type="usc",
            volume="17",
            section="107",
        )

        assert citation.normalized_string == "17 U.S.C. Section 107"

        # Federal Reporter citation
        citation = CitationObj(
            original_text="394 US 705",
            citation_type="federal_reporter",
            volume="394",
            reporter="U.S.",
            page="705",
        )

        assert citation.normalized_string == "394 U.S. 705"

    def test_validate_reporter(self, normalizer: CitationNormalizer) -> None:
        """Test reporter validation."""
        # Valid reporters
        assert normalizer._is_valid_reporter("U.S.")
        assert normalizer._is_valid_reporter("F.3d")
        assert normalizer._is_valid_reporter("F. Supp.")

        # Invalid reporters (should fail with strict validation)
        # Note: Current implementation is permissive, may need adjustment
        assert not normalizer._is_valid_reporter("Fake Reporter")

    def test_normalize_reporter(self, normalizer: CitationNormalizer) -> None:
        """Test reporter normalization."""
        # F3d -> F.3d
        normalized = normalizer._normalize_reporter("F3d")
        assert normalized == "F.3d"

        # US -> U.S.
        normalized = normalizer._normalize_reporter("US")
        assert normalized == "U.S."

    def test_extract_multiple_citations(self, normalizer: CitationNormalizer) -> None:
        """Test extracting multiple citations from text."""
        text = """
        See 17 U.S.C. Section 107 for fair use. The case 394 U.S. 705 established
        the precedent. Regulations at 17 C.F.R. Section 240.10b-5 apply.
        """

        citations = normalizer.extract_citations(text)

        # Should extract at least 3 citations (may extract more via eyecite)
        assert len(citations) >= 3, f"Expected >=3 citations, got {len(citations)}"

        # Check for each type
        types_found = {c.citation_type for c in citations}
        assert "usc" in types_found or "case" in types_found
        assert "cfr" in types_found or "case" in types_found

    def test_empty_text(self, normalizer: CitationNormalizer) -> None:
        """Test handling of empty text."""
        citations = normalizer.extract_citations("")
        assert len(citations) == 0

        citations = normalizer.extract_citations("   ")
        assert len(citations) == 0

    def test_no_citations(self, normalizer: CitationNormalizer) -> None:
        """Test text with no citations."""
        text = "This is plain text with no legal citations."

        citations = normalizer.extract_citations(text)

        assert len(citations) == 0, "False positive: extracted citations from plain text"
