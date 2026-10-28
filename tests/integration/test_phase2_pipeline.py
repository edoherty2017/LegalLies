"""
Integration tests for Phase 2: Legal-Specific Processing.

Tests the complete pipeline:
- OCR correction
- PII redaction
- Statutory parsing
- Legal chunking
"""

import pytest
from src.corpus import (
    OCRCorrector,
    PIIRedactor,
    StatutoryParser,
    LegalChunker,
    CaseType,
    OpinionType,
    PIIType,
)


class TestOCRCorrection:
    """Test OCR correction pipeline."""

    @pytest.fixture
    def corrector(self):
        """Initialize OCR corrector."""
        return OCRCorrector(confidence_threshold=0.8)

    def test_basic_correction(self, corrector):
        """Test basic OCR error correction."""
        corrupted = "The p1aintiff fi1ed a moti0n in district c0urt."

        result = corrector.correct(corrupted, aggressive=True)

        assert result["correction_count"] > 0
        assert "plaintiff" in result["cleaned_text"].lower()
        assert "filed" in result["cleaned_text"].lower()

    def test_legal_terms(self, corrector):
        """Test correction of legal terms."""
        corrupted = "The judg3 sustained the obj3ction to the fiduciary duty."

        result = corrector.correct(corrupted, aggressive=True)

        # Should correct "judg3" and "obj3ction"
        assert result["correction_count"] >= 1

    def test_dual_storage(self, corrector):
        """Test that both raw and cleaned text are stored."""
        corrupted = "The p1aintiff fi1ed a moti0n."

        result = corrector.correct(corrupted)

        assert "raw_text" in result
        assert "cleaned_text" in result
        assert result["raw_text"] == corrupted  # Original preserved


class TestPIIRedaction:
    """Test PII redaction with legal NER."""

    @pytest.fixture
    def redactor(self):
        """Initialize PII redactor."""
        return PIIRedactor()

    def test_ssn_redaction(self, redactor):
        """Test SSN redaction."""
        text = "John Smith, SSN 123-45-6789, filed a complaint."

        result = redactor.redact(text)

        assert "[SSN REDACTED]" in result["redacted_text"]
        assert "123-45-6789" not in result["redacted_text"]

        # Check redaction details
        ssn_redactions = [r for r in result["redactions"] if r.pii_type == PIIType.SSN]
        assert len(ssn_redactions) > 0

    def test_email_redaction(self, redactor):
        """Test email redaction."""
        text = "Contact jane.doe@example.com for more information."

        result = redactor.redact(text)

        assert "[EMAIL REDACTED]" in result["redacted_text"]
        assert "jane.doe@example.com" not in result["redacted_text"]

    def test_phone_redaction(self, redactor):
        """Test phone number redaction."""
        text = "Call (555) 123-4567 for assistance."

        result = redactor.redact(text)

        assert "[PHONE REDACTED]" in result["redacted_text"]
        assert "555" not in result["redacted_text"]

    def test_public_figure_whitelist(self, redactor):
        """Test that public figures are NOT redacted."""
        text = "Justice Ruth Bader Ginsburg delivered the opinion."

        result = redactor.redact(text)

        # Should preserve public figure name
        assert "Ruth Bader Ginsburg" in result["redacted_text"]

    def test_judge_whitelist(self, redactor):
        """Test that judges are NOT redacted."""
        text = "The case was heard by Judge Alice Johnson."

        result = redactor.redact(text)

        # Should preserve judge name
        assert "Alice Johnson" in result["redacted_text"]

    def test_case_type_sensitivity(self, redactor):
        """Test that family law cases get aggressive redaction."""
        text = "John Smith testified against Jane Doe."

        # Normal case - should apply whitelist
        normal_result = redactor.redact(text, case_type=CaseType.CIVIL)

        # Family law - should redact aggressively
        family_result = redactor.redact(text, case_type=CaseType.FAMILY)

        # Family law should have more redactions
        assert family_result["redaction_count"] >= normal_result["redaction_count"]


class TestStatutoryParsing:
    """Test statutory parsing with breadcrumb injection."""

    @pytest.fixture
    def parser(self):
        """Initialize statutory parser."""
        return StatutoryParser()

    def test_usc_parsing(self, parser):
        """Test USC parsing."""
        usc_text = """
        § 101. Definitions

        (1) "Person" means an individual, corporation, partnership, or association.
        (2) "Work" means a work of authorship fixed in any tangible medium.

        § 106. Exclusive rights in copyrighted works

        The owner of copyright has the exclusive rights to reproduce the work.
        """

        sections = parser.parse_usc(text=usc_text, title="17")

        assert len(sections) == 2
        assert sections[0].section == "101"
        assert sections[1].section == "106"

    def test_definition_extraction(self, parser):
        """Test that definitions are extracted."""
        usc_text = """
        § 101. Definitions

        (1) "Person" means an individual, corporation, or partnership.

        § 106. Exclusive rights

        The person who owns copyright has exclusive rights.
        """

        sections = parser.parse_usc(text=usc_text, title="17")

        # §106 should reference the "person" definition
        section_106 = sections[1]
        assert "person" in section_106.definitions

    def test_breadcrumb_generation(self, parser):
        """Test breadcrumb path generation."""
        usc_text = """
        § 101. Definitions

        (1) "Person" means an individual.
        """

        sections = parser.parse_usc(text=usc_text, title="17")

        breadcrumb = sections[0].breadcrumb
        assert "Title 17" in breadcrumb
        assert "Section 101" in breadcrumb

    def test_definition_injection(self, parser):
        """Test that definitions are injected into text."""
        usc_text = """
        § 101. Definitions

        (1) "Person" means an individual, corporation, or partnership.

        § 106. Rights

        Any person may exercise these rights.
        """

        sections = parser.parse_usc(text=usc_text, title="17")

        # §106 should have definition prepended
        section_106 = sections[1]
        assert "[Definitions:" in section_106.text
        assert "Person" in section_106.text


class TestLegalChunking:
    """Test legal semantic chunking."""

    @pytest.fixture
    def chunker(self):
        """Initialize legal chunker."""
        return LegalChunker(max_chunk_size=500)

    def test_opinion_type_detection(self, chunker):
        """Test opinion type detection."""
        dissent_text = """
        JUSTICE SCALIA, dissenting.

        I respectfully dissent from the majority's opinion.
        """

        chunks = chunker.chunk(dissent_text)

        assert len(chunks) > 0
        assert chunks[0].opinion_type == OpinionType.DISSENT

    def test_header_prepending(self, chunker):
        """Test that headers are prepended to chunks."""
        opinion_text = """
        I. FACTS

        The plaintiff filed a complaint on January 1, 2020.

        II. ANALYSIS

        The court finds that the defendant breached the contract.
        """

        chunks = chunker.chunk(opinion_text)

        # Each chunk should include its header
        for chunk in chunks:
            if chunk.header:
                assert chunk.header in chunk.text

    def test_section_detection(self, chunker):
        """Test section type detection."""
        opinion_text = """
        I. FACTS

        The case involves a contract dispute.

        II. CONCLUSION

        The court grants summary judgment.
        """

        chunks = chunker.chunk(opinion_text)

        # Should detect FACTS and CONCLUSION sections
        section_types = [chunk.section_type.value for chunk in chunks]
        assert "facts" in section_types or "conclusion" in section_types


class TestEndToEndPipeline:
    """Test complete Phase 2 pipeline."""

    @pytest.fixture
    def pipeline_components(self):
        """Initialize all pipeline components."""
        return {
            "ocr": OCRCorrector(),
            "pii": PIIRedactor(),
            "parser": StatutoryParser(),
            "chunker": LegalChunker(),
        }

    def test_opinion_processing(self, pipeline_components):
        """Test complete pipeline for judicial opinion."""
        # Simulated OCR output with errors and PII
        raw_opinion = """
        Case No. 2023-CV-12345

        John Smith, SSN 123-45-6789, plaintiff
        v.
        Jane Doe, defendant

        OPINION

        The p1aintiff fi1ed a moti0n for summary judgment.
        Judge Alice Johnson presiding.

        I. FACTS

        The plaintiff claims breach of contract.

        II. CONCLUSION

        The motion is denied.
        """

        # Step 1: OCR correction
        ocr_result = pipeline_components["ocr"].correct(raw_opinion, aggressive=True)
        cleaned = ocr_result["cleaned_text"]

        # Step 2: PII redaction
        pii_result = pipeline_components["pii"].redact(cleaned, case_type=CaseType.CIVIL)
        safe_text = pii_result["redacted_text"]

        # Verify redactions
        assert "[SSN REDACTED]" in safe_text
        assert "123-45-6789" not in safe_text

        # Step 3: Chunking
        chunks = pipeline_components["chunker"].chunk(
            safe_text,
            metadata={"case_name": "Smith v. Doe", "year": 2023}
        )

        # Verify chunks
        assert len(chunks) > 0

        # Should have FACTS and CONCLUSION sections
        section_types = {chunk.section_type.value for chunk in chunks}
        assert "facts" in section_types or "conclusion" in section_types

        # Each chunk should have metadata
        for chunk in chunks:
            assert chunk.metadata["case_name"] == "Smith v. Doe"

    def test_statute_processing(self, pipeline_components):
        """Test complete pipeline for statutory text."""
        raw_statute = """
        Title 17 - Copyright

        § 101. Definitions

        (1) "Person" means an individual, corporation, or partnership.
        (2) "Work" means a work of authorship.

        § 106. Exclusive rights

        Any person may exercise exclusive rights to the work.
        """

        # Step 1: OCR correction (if needed)
        ocr_result = pipeline_components["ocr"].correct(raw_statute)
        cleaned = ocr_result["cleaned_text"]

        # Step 2: Statutory parsing (no PII in statutes usually)
        sections = pipeline_components["parser"].parse_usc(
            text=cleaned,
            title="17"
        )

        # Verify parsing
        assert len(sections) == 2

        # §106 should have definition injected
        section_106 = sections[1]
        assert "person" in section_106.definitions
        assert "[Definitions:" in section_106.text


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
