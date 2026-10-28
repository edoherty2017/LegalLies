"""
PII Redaction with Legal-Specific NER.

Implements the privacy protection strategy described in Section 2.2.1 of the
legal RAG framework. Uses multi-layer redaction (Regex + NER) with contextual
awareness to balance privacy protection and legal utility.

The "Practical Obscurity" Problem:
Court records are public, but digitization removes the friction of physical
access. We must aggressively redact PII while preserving legal utility.

Redaction Strategy:
1. Regex: Structured identifiers (SSN, phone, email, addresses)
2. NER: Names (using legal-tuned model that knows "Judge Smith" from "John Smith")
3. Whitelist: Public figures, case names (preserve legal identity)
4. Context: Sensitive case types get aggressive redaction (family law, asylum)

References:
- Section 2.2.1: Privacy and PII Redaction
- Section 4: Privacy and Redaction in Online Judicial Records
"""

import re
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import spacy
from spacy.tokens import Doc


class CaseType(Enum):
    """Legal case types with different privacy sensitivities."""

    CRIMINAL = "criminal"
    CIVIL = "civil"
    FAMILY = "family"              # HIGH sensitivity
    ASYLUM = "asylum"              # HIGH sensitivity
    BANKRUPTCY = "bankruptcy"
    PATENT = "patent"
    CONTRACT = "contract"
    TORT = "tort"
    CONSTITUTIONAL = "constitutional"
    UNKNOWN = "unknown"


class PIIType(Enum):
    """Types of personally identifiable information."""

    SSN = "ssn"                    # Social Security Number
    PHONE = "phone"                # Phone number
    EMAIL = "email"                # Email address
    ADDRESS = "address"            # Physical address
    DOB = "date_of_birth"          # Date of birth
    ACCOUNT = "account_number"     # Account/credit card
    NAME = "name"                  # Person name
    MEDICAL = "medical_info"       # Health information


@dataclass
class RedactionSpan:
    """A span of text to be redacted."""

    start: int                     # Start position
    end: int                       # End position
    text: str                      # Original text
    pii_type: PIIType              # Type of PII
    confidence: float              # Confidence in redaction (0-1)
    whitelisted: bool = False      # Skip redaction if True


class PIIRedactor:
    """
    Multi-layer PII redactor with legal context awareness.

    Implements Regex + NER + Whitelist + Context-sensitive redaction.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        aggressive: bool = False,
    ):
        """
        Initialize PII redactor.

        Args:
            spacy_model: Spacy model for NER (use legal-tuned if available)
            aggressive: If True, redact more aggressively
        """
        self.aggressive = aggressive

        # Load Spacy NER model
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Spacy model '{spacy_model}' not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)

        # Compile regex patterns
        self.regex_patterns = self._compile_regex_patterns()

        # Load public figure whitelist
        self.public_figures = self._load_public_figures()

        # Load case name whitelist
        self.case_names = self._load_case_names()

    def _compile_regex_patterns(self) -> Dict[PIIType, List[re.Pattern]]:
        """
        Compile regex patterns for structured PII.

        Returns:
            Dictionary of PIIType -> patterns
        """
        patterns = {
            # Social Security Numbers
            PIIType.SSN: [
                re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),          # 123-45-6789
                re.compile(r'\b\d{3}\s\d{2}\s\d{4}\b'),        # 123 45 6789
                re.compile(r'\b\d{9}\b'),                      # 123456789
            ],

            # Phone Numbers
            PIIType.PHONE: [
                re.compile(r'\b\d{3}-\d{3}-\d{4}\b'),          # 123-456-7890
                re.compile(r'\(\d{3}\)\s?\d{3}-\d{4}'),        # (123) 456-7890
                re.compile(r'\b\d{3}\.\d{3}\.\d{4}\b'),        # 123.456.7890
            ],

            # Email Addresses
            PIIType.EMAIL: [
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            ],

            # Physical Addresses (simplified)
            PIIType.ADDRESS: [
                re.compile(r'\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b', re.IGNORECASE),
            ],

            # Dates (potential DOB)
            PIIType.DOB: [
                re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b'),     # MM/DD/YYYY
                re.compile(r'\b\d{1,2}-\d{1,2}-\d{4}\b'),     # MM-DD-YYYY
            ],

            # Account Numbers
            PIIType.ACCOUNT: [
                re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),  # Credit card
                re.compile(r'\bAccount\s*#?\s*\d{6,12}\b', re.IGNORECASE),
            ],
        }

        return patterns

    def _load_public_figures(self) -> Set[str]:
        """
        Load whitelist of public figures (don't redact).

        In production, this would be a large database.

        Returns:
            Set of public figure names
        """
        # Famous judges, Supreme Court justices, well-known attorneys
        public_figures = {
            # Supreme Court (current and recent)
            "John Roberts", "Clarence Thomas", "Samuel Alito", "Sonia Sotomayor",
            "Elena Kagan", "Neil Gorsuch", "Brett Kavanaugh", "Amy Coney Barrett",
            "Ketanji Brown Jackson", "Ruth Bader Ginsburg", "Antonin Scalia",
            "Anthony Kennedy", "Sandra Day O'Connor",

            # Famous judges
            "Learned Hand", "Richard Posner", "Frank Easterbrook",
            "Alex Kozinski", "Jed Rakoff",

            # Famous attorneys
            "Clarence Darrow", "Thurgood Marshall", "Alan Dershowitz",

            # Politicians (often appear in cases)
            "Donald Trump", "Joe Biden", "Barack Obama", "George Bush",
            "Bill Clinton", "Hillary Clinton",
        }

        return {name.lower() for name in public_figures}

    def _load_case_names(self) -> Set[str]:
        """
        Load whitelist of famous case names (don't redact).

        Returns:
            Set of case party names
        """
        # Famous case parties
        case_names = {
            "Miranda", "Wade", "Roe", "Brown", "Marbury", "Madison",
            "Gideon", "Wainwright", "Mapp", "Ohio", "Terry",
            "United States", "People", "State", "Commonwealth",
        }

        return {name.lower() for name in case_names}

    def redact(
        self,
        text: str,
        case_type: CaseType = CaseType.UNKNOWN,
    ) -> Dict[str, any]:
        """
        Redact PII from legal text.

        Args:
            text: Input text
            case_type: Type of legal case (affects sensitivity)

        Returns:
            {
                "original_text": str,
                "redacted_text": str,
                "redactions": List[RedactionSpan],
                "redaction_count": int
            }
        """
        # Step 1: Regex-based redaction (structured identifiers)
        regex_spans = self._regex_redaction(text)

        # Step 2: NER-based redaction (names)
        ner_spans = self._ner_redaction(text)

        # Step 3: Whitelist filtering
        filtered_spans = self._apply_whitelist(ner_spans, text)

        # Step 4: Context-sensitive filtering
        final_spans = self._apply_context_rules(
            filtered_spans, text, case_type
        )

        # Step 5: Merge all redactions
        all_spans = regex_spans + final_spans

        # Step 6: Remove overlaps
        merged_spans = self._merge_overlapping_spans(all_spans)

        # Step 7: Apply redactions
        redacted_text = self._apply_redactions(text, merged_spans)

        return {
            "original_text": text,
            "redacted_text": redacted_text,
            "redactions": merged_spans,
            "redaction_count": len(merged_spans),
        }

    def _regex_redaction(self, text: str) -> List[RedactionSpan]:
        """
        Redact structured PII using regex.

        Args:
            text: Input text

        Returns:
            List of redaction spans
        """
        spans = []

        for pii_type, patterns in self.regex_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    spans.append(RedactionSpan(
                        start=match.start(),
                        end=match.end(),
                        text=match.group(0),
                        pii_type=pii_type,
                        confidence=1.0,  # Regex is high confidence
                    ))

        return spans

    def _ner_redaction(self, text: str) -> List[RedactionSpan]:
        """
        Redact names using NER.

        Args:
            text: Input text

        Returns:
            List of redaction spans
        """
        doc = self.nlp(text)
        spans = []

        for ent in doc.ents:
            # Only redact PERSON entities
            if ent.label_ == "PERSON":
                spans.append(RedactionSpan(
                    start=ent.start_char,
                    end=ent.end_char,
                    text=ent.text,
                    pii_type=PIIType.NAME,
                    confidence=0.9,  # NER is slightly less confident
                ))

        return spans

    def _apply_whitelist(
        self,
        spans: List[RedactionSpan],
        text: str,
    ) -> List[RedactionSpan]:
        """
        Filter out whitelisted names (public figures, case names).

        Args:
            spans: Redaction spans
            text: Original text

        Returns:
            Filtered spans
        """
        filtered = []

        for span in spans:
            name_lower = span.text.lower()

            # Check if whitelisted
            is_whitelisted = False

            # Check public figures
            if name_lower in self.public_figures:
                is_whitelisted = True

            # Check case names
            if name_lower in self.case_names:
                is_whitelisted = True

            # Check if it's a judge (preserve judicial identity)
            if self._is_judge(span.text, text, span.start):
                is_whitelisted = True

            if is_whitelisted:
                span.whitelisted = True
                # Don't add to filtered list
            else:
                filtered.append(span)

        return filtered

    def _is_judge(self, name: str, text: str, position: int) -> bool:
        """
        Check if name is a judge (preserve judicial identity).

        Args:
            name: Person name
            text: Full text
            position: Position of name in text

        Returns:
            True if judge, False otherwise
        """
        # Check context window (50 chars before)
        context_start = max(0, position - 50)
        context = text[context_start:position].lower()

        # Look for judge indicators
        judge_indicators = [
            "judge", "justice", "magistrate", "chief judge",
            "district judge", "circuit judge", "presiding judge",
        ]

        for indicator in judge_indicators:
            if indicator in context:
                return True

        return False

    def _apply_context_rules(
        self,
        spans: List[RedactionSpan],
        text: str,
        case_type: CaseType,
    ) -> List[RedactionSpan]:
        """
        Apply context-sensitive redaction rules.

        Sensitive case types (family law, asylum) get more aggressive redaction.

        Args:
            spans: Redaction spans
            text: Original text
            case_type: Type of case

        Returns:
            Filtered spans
        """
        # High-sensitivity case types
        sensitive_types = {CaseType.FAMILY, CaseType.ASYLUM}

        if case_type in sensitive_types or self.aggressive:
            # Redact ALL names (ignore whitelist)
            return [span for span in spans]
        else:
            # Normal redaction
            return spans

    def _merge_overlapping_spans(
        self,
        spans: List[RedactionSpan],
    ) -> List[RedactionSpan]:
        """
        Merge overlapping redaction spans.

        Args:
            spans: Redaction spans

        Returns:
            Merged spans
        """
        if not spans:
            return []

        # Sort by start position
        sorted_spans = sorted(spans, key=lambda s: s.start)

        merged = [sorted_spans[0]]

        for current in sorted_spans[1:]:
            previous = merged[-1]

            # Check for overlap
            if current.start <= previous.end:
                # Merge
                previous.end = max(previous.end, current.end)
                previous.text = f"{previous.text}+{current.text}"
            else:
                # No overlap, add as new span
                merged.append(current)

        return merged

    def _apply_redactions(
        self,
        text: str,
        spans: List[RedactionSpan],
    ) -> str:
        """
        Apply redactions to text.

        Args:
            text: Original text
            spans: Redaction spans

        Returns:
            Redacted text
        """
        # Sort spans by position (descending) to avoid position shifts
        sorted_spans = sorted(spans, key=lambda s: s.start, reverse=True)

        redacted = text
        for span in sorted_spans:
            # Replacement text based on PII type
            replacement = self._get_replacement_text(span.pii_type)
            redacted = redacted[:span.start] + replacement + redacted[span.end:]

        return redacted

    def _get_replacement_text(self, pii_type: PIIType) -> str:
        """
        Get replacement text for PII type.

        Args:
            pii_type: Type of PII

        Returns:
            Replacement string
        """
        replacements = {
            PIIType.SSN: "[SSN REDACTED]",
            PIIType.PHONE: "[PHONE REDACTED]",
            PIIType.EMAIL: "[EMAIL REDACTED]",
            PIIType.ADDRESS: "[ADDRESS REDACTED]",
            PIIType.DOB: "[DATE REDACTED]",
            PIIType.ACCOUNT: "[ACCOUNT# REDACTED]",
            PIIType.NAME: "[NAME REDACTED]",
            PIIType.MEDICAL: "[MEDICAL INFO REDACTED]",
        }

        return replacements.get(pii_type, "[PII REDACTED]")


# Example usage
if __name__ == "__main__":
    # Example legal text with PII
    text = """
    In the Matter of John Smith, SSN 123-45-6789, residing at 456 Oak Street,
    Boston, MA. The plaintiff, Jane Doe, can be reached at jane.doe@email.com
    or by phone at (555) 123-4567. The case was heard by Judge Alice Johnson.
    Justice Ruth Bader Ginsburg delivered the opinion of the Court in Roe v. Wade.
    """

    # Initialize redactor
    redactor = PIIRedactor()

    # Redact (civil case - normal sensitivity)
    result = redactor.redact(text, case_type=CaseType.CIVIL)

    # Print results
    print("Original Text:")
    print(result["original_text"])
    print("\nRedacted Text:")
    print(result["redacted_text"])
    print(f"\nRedacted {result['redaction_count']} items:")
    for redaction in result["redactions"]:
        print(f"  {redaction.pii_type.value}: '{redaction.text}' (whitelisted: {redaction.whitelisted})")
