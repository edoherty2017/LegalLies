"""
Unit tests for legal sentence segmenter.
"""

import pytest

from src.preprocessing.legal_segmenter import LegalSegmenter, get_legal_segmenter


class TestLegalSegmenter:
    """Tests for LegalSegmenter."""

    @pytest.fixture
    def segmenter(self) -> LegalSegmenter:
        """Create legal segmenter."""
        return LegalSegmenter()

    def test_basic_segmentation(self, segmenter: LegalSegmenter) -> None:
        """Test basic sentence splitting."""
        text = "This is sentence one. This is sentence two."

        sentences = segmenter.split(text)

        assert len(sentences) == 2
        assert "sentence one" in sentences[0].lower()
        assert "sentence two" in sentences[1].lower()

    def test_legal_abbreviation_v(self, segmenter: LegalSegmenter) -> None:
        """Test that 'v.' doesn't split sentences."""
        text = "The case is Smith v. Jones. It was decided in 1999."

        sentences = segmenter.split(text)

        # Should be 2 sentences, not split on "v."
        assert len(sentences) == 2, f"Expected 2 sentences, got {len(sentences)}: {sentences}"
        assert "Smith v. Jones" in sentences[0]

    def test_legal_abbreviation_us(self, segmenter: LegalSegmenter) -> None:
        """Test that 'U.S.' doesn't split sentences."""
        text = "The U.S. Supreme Court ruled. The decision was final."

        sentences = segmenter.split(text)

        assert len(sentences) == 2
        assert "U.S." in sentences[0]

    def test_legal_abbreviation_id(self, segmenter: LegalSegmenter) -> None:
        """Test that 'Id.' doesn't split sentences."""
        text = "See Id. at 405. The plaintiff prevailed."

        sentences = segmenter.split(text)

        assert len(sentences) == 2
        assert "Id." in sentences[0]

    def test_legal_abbreviation_et_al(self, segmenter: LegalSegmenter) -> None:
        """Test that 'et al.' doesn't split sentences."""
        text = "Smith et al. argued successfully. The court agreed."

        sentences = segmenter.split(text)

        assert len(sentences) == 2
        assert "et al." in sentences[0]

    def test_legal_abbreviation_eg_ie(self, segmenter: LegalSegmenter) -> None:
        """Test that 'e.g.' and 'i.e.' don't split sentences."""
        text = "There are exceptions, e.g., fair use. This means limited copying, i.e., not wholesale."

        sentences = segmenter.split(text)

        # Should preserve e.g. and i.e. within sentences
        assert any("e.g." in s for s in sentences)
        assert any("i.e." in s for s in sentences)

    def test_corporate_abbreviations(self, segmenter: LegalSegmenter) -> None:
        """Test that corporate abbreviations don't split sentences."""
        text = "Apple Inc. sued Samsung Corp. The case was settled."

        sentences = segmenter.split(text)

        assert len(sentences) == 2
        assert "Inc." in sentences[0]
        assert "Corp." in sentences[0]

    def test_complex_legal_text(self, segmenter: LegalSegmenter) -> None:
        """Test on complex legal text with multiple abbreviations."""
        text = """
        In Smith v. Jones, 394 U.S. 705 (1969), the U.S. Supreme Court held that
        fair use applies. See Id. at 710. The defendants, Apple Inc. et al., prevailed.
        This precedent controls, e.g., in copyright cases.
        """

        sentences = segmenter.split(text)

        # Should not over-split on abbreviations
        assert len(sentences) >= 3  # At least 3 real sentences
        assert len(sentences) <= 5  # But not too many fragments

        # Check that legal abbreviations are preserved
        full_text = " ".join(sentences)
        assert "v." in full_text
        assert "U.S." in full_text
        assert "Id." in full_text
        assert "et al." in full_text
        assert "e.g." in full_text

    def test_empty_text(self, segmenter: LegalSegmenter) -> None:
        """Test handling of empty text."""
        assert segmenter.split("") == []
        assert segmenter.split("   ") == []
        assert segmenter.split("\n\n") == []

    def test_single_sentence(self, segmenter: LegalSegmenter) -> None:
        """Test single sentence."""
        text = "This is a single sentence."

        sentences = segmenter.split(text)

        assert len(sentences) == 1
        assert sentences[0].strip() == "This is a single sentence."

    def test_filter_short_fragments(self, segmenter: LegalSegmenter) -> None:
        """Test that very short fragments are filtered."""
        text = "A. B. C. This is a real sentence."

        sentences = segmenter.split(text)

        # Short fragments (< 5 chars) should be filtered
        for sent in sentences:
            assert len(sent) >= 5, f"Short fragment not filtered: '{sent}'"

    def test_split_with_metadata(self, segmenter: LegalSegmenter) -> None:
        """Test split_with_metadata returns character offsets."""
        text = "First sentence. Second sentence."

        results = segmenter.split_with_metadata(text)

        assert len(results) == 2

        # Check metadata structure
        assert "text" in results[0]
        assert "start" in results[0]
        assert "end" in results[0]

        # Check offsets
        assert results[0]["text"] == "First sentence."
        assert results[0]["start"] >= 0
        assert results[0]["end"] > results[0]["start"]

        assert results[1]["text"] == "Second sentence."
        assert results[1]["start"] > results[0]["end"] - 1

    def test_singleton_pattern(self) -> None:
        """Test that get_legal_segmenter returns singleton instance."""
        seg1 = get_legal_segmenter()
        seg2 = get_legal_segmenter()

        # Should be same instance
        assert seg1 is seg2

    def test_uppercase_abbreviations(self, segmenter: LegalSegmenter) -> None:
        """Test that uppercase variants are also handled."""
        text = "The case SMITH V. JONES was decided. The court ruled."

        sentences = segmenter.split(text)

        # Should handle uppercase V. properly
        assert len(sentences) == 2
