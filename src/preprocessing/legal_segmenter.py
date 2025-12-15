"""
Legal-aware sentence segmentation using Spacy.

Prevents splitting on legal abbreviations like "v.", "U.S.", "Id.", "et al."
which would fragment sentences and destroy context for NLI verification.

Standard sentence splitters fail on legal text due to high density of periods.
Example: "See Smith v. Jones, 123 U.S. 456." would incorrectly split into
"See Smith v.", "Jones, 123 U.", "S.", "456." without legal-aware handling.
"""

from typing import List

import spacy
from spacy.lang.en import English


class LegalSegmenter:
    """
    Sentence segmentation with legal abbreviation protection.

    Uses Spacy with custom tokenizer exceptions for legal terminology.
    """

    # Legal abbreviations that should NOT trigger sentence boundaries
    LEGAL_ABBREVIATIONS = [
        "v.",
        "vs.",
        "U.S.",
        "U. S.",
        "U.S.C.",
        "C.F.R.",
        "Fed. R.",
        "Fed. R. Civ. P.",
        "Fed. R. Crim. P.",
        "Fed. R. Evid.",
        "Fed. R. App. P.",
        "Cir.",
        "Ct.",
        "D.C.",
        "Supp.",
        "F.3d",
        "F.2d",
        "F. Supp.",
        "S. Ct.",
        "L. Ed.",
        "Id.",
        "id.",
        "Ibid.",
        "ibid.",
        "e.g.",
        "i.e.",
        "et al.",
        "etc.",
        "Inc.",
        "Corp.",
        "Ltd.",
        "Co.",
        "No.",
        "Section",
        "paragraph",
    ]

    def __init__(self, model: str = "en_core_web_sm") -> None:
        """
        Initialize legal segmenter with Spacy model.

        Args:
            model: Spacy model name (default: en_core_web_sm)
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Spacy model '{model}' not found. Using blank English model.")
            print("Run: python -m spacy download en_core_web_sm")
            self.nlp = English()

        self._add_special_cases()

    def _add_special_cases(self) -> None:
        """Add legal abbreviations as tokenizer exceptions."""
        for abbr in self.LEGAL_ABBREVIATIONS:
            # Add as special case to prevent splitting
            # The tokenizer will treat these as single tokens
            self.nlp.tokenizer.add_special_case(abbr, [{"ORTH": abbr}])

            # Also add uppercase variant
            upper_abbr = abbr.upper()
            if upper_abbr != abbr:
                self.nlp.tokenizer.add_special_case(upper_abbr, [{"ORTH": upper_abbr}])

    def split(self, text: str) -> List[str]:
        """
        Split text into sentences with legal-aware boundaries.

        Args:
            text: Text to segment

        Returns:
            List of sentence strings
        """
        if not text or not text.strip():
            return []

        # Process with Spacy
        doc = self.nlp(text)

        # Extract sentences
        sentences = [sent.text.strip() for sent in doc.sents]

        # Filter out very short fragments (< 5 characters)
        sentences = [s for s in sentences if len(s) >= 5]

        return sentences

    def split_with_metadata(self, text: str) -> List[dict]:
        """
        Split text and return sentences with metadata.

        Args:
            text: Text to segment

        Returns:
            List of dicts with 'text', 'start', 'end' keys
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)

        sentences_with_metadata = []
        for sent in doc.sents:
            if len(sent.text.strip()) >= 5:
                sentences_with_metadata.append(
                    {
                        "text": sent.text.strip(),
                        "start": sent.start_char,
                        "end": sent.end_char,
                    }
                )

        return sentences_with_metadata


# Singleton instance for reuse (avoid reloading model multiple times)
_segmenter_instance = None


def get_legal_segmenter() -> LegalSegmenter:
    """Get singleton legal segmenter instance."""
    global _segmenter_instance
    if _segmenter_instance is None:
        _segmenter_instance = LegalSegmenter()
    return _segmenter_instance
