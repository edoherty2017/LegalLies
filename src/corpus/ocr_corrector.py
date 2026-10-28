"""
OCR Correction Pipeline for Historical Legal Documents.

Implements the error correction strategy described in Section 2.2.2 and 6.1.2
of the legal RAG framework. Uses statistical language models to detect and
correct systematic OCR errors in scanned legal documents.

The Problem:
Historical case law exists only as scanned images. OCR introduces systematic
errors that fragment the vector space:
- "fiduciary" → "fiduciary" (correct)
- "fiduciary" → "fiduciary" (OCR error: "c" → "c")
- These become distinct tokens despite being the same word!

Solution:
BERT-based masked language modeling predicts correct tokens based on context.

References:
- Section 2.2.2: Optical Character Recognition (OCR) Correction
- Section 6.1.2: CourtListener Opinions - OCR Noise
"""

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import Counter

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


@dataclass
class OCRError:
    """Detected OCR error with correction suggestion."""

    original: str       # Original (likely wrong) token
    corrected: str      # Suggested correction
    confidence: float   # Confidence in correction (0-1)
    position: int       # Character position in text
    context: str        # Surrounding context


class OCRCorrector:
    """
    Statistical OCR error corrector using BERT masked language modeling.

    Dual Storage Strategy:
    - raw_text: Original OCR output (for citations, exact quotes)
    - cleaned_text: Corrected text (for vector search)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        confidence_threshold: float = 0.8,
        levenshtein_threshold: int = 2,
        device: str = "cpu",
    ):
        """
        Initialize OCR corrector.

        Args:
            model_name: HuggingFace model for masked LM
            confidence_threshold: Minimum confidence for correction (0-1)
            levenshtein_threshold: Maximum edit distance for candidate
            device: Device for model ('cpu' or 'cuda')
        """
        self.confidence_threshold = confidence_threshold
        self.levenshtein_threshold = levenshtein_threshold
        self.device = device

        # Load BERT model for masked language modeling
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # Common OCR error patterns (regex)
        self.ocr_patterns = self._compile_ocr_patterns()

        # Legal dictionary (common legal terms)
        self.legal_dictionary = self._load_legal_dictionary()

    def _compile_ocr_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """
        Compile common OCR error patterns.

        Returns:
            List of (pattern, description) tuples
        """
        patterns = [
            # Digit/letter confusion
            (re.compile(r'\b1\b'), "1 mistaken for I or l"),
            (re.compile(r'\b0\b'), "0 mistaken for O"),

            # Adjacent character merges
            (re.compile(r'rn'), "rn mistaken for m"),
            (re.compile(r'cl'), "cl mistaken for d"),
            (re.compile(r'vv'), "vv mistaken for w"),

            # Common legal term corruptions
            (re.compile(r'p1aintiff'), "plaintiff"),
            (re.compile(r'fi1ed'), "filed"),
            (re.compile(r'moti0n'), "motion"),
            (re.compile(r'c0urt'), "court"),
            (re.compile(r'judg[e3]'), "judge"),
        ]

        return patterns

    def _load_legal_dictionary(self) -> set:
        """
        Load legal dictionary for validation.

        Returns:
            Set of valid legal terms
        """
        # Common legal terms (expandable)
        legal_terms = {
            "plaintiff", "defendant", "motion", "court", "judge", "jury",
            "filed", "ruling", "appeal", "statute", "regulation", "precedent",
            "jurisdiction", "venue", "discovery", "deposition", "testimony",
            "evidence", "objection", "sustained", "overruled", "voir dire",
            "habeas corpus", "amicus curiae", "prima facie", "res judicata",
            "stare decisis", "certiorari", "mandamus", "subpoena", "summons",
            "arraignment", "indictment", "verdict", "sentence", "probation",
        }

        return legal_terms

    def correct(
        self,
        text: str,
        aggressive: bool = False,
    ) -> Dict[str, any]:
        """
        Correct OCR errors in text.

        Args:
            text: Raw OCR text
            aggressive: If True, apply more corrections (lower confidence)

        Returns:
            {
                "raw_text": str,           # Original text
                "cleaned_text": str,       # Corrected text
                "errors": List[OCRError],  # Detected errors
                "correction_count": int    # Number of corrections
            }
        """
        # Step 1: Detect suspicious tokens
        suspicious_tokens = self._detect_suspicious_tokens(text)

        # Step 2: For each suspicious token, predict correction
        errors = []
        for token, position, context in suspicious_tokens:
            correction = self._predict_correction(
                token, context, aggressive=aggressive
            )

            if correction:
                errors.append(OCRError(
                    original=token,
                    corrected=correction["token"],
                    confidence=correction["confidence"],
                    position=position,
                    context=context,
                ))

        # Step 3: Apply corrections
        cleaned_text = self._apply_corrections(text, errors)

        return {
            "raw_text": text,
            "cleaned_text": cleaned_text,
            "errors": errors,
            "correction_count": len(errors),
        }

    def _detect_suspicious_tokens(
        self, text: str
    ) -> List[Tuple[str, int, str]]:
        """
        Detect tokens that are likely OCR errors.

        Heuristics:
        1. Contains digits in unusual positions (e.g., "p1aintiff")
        2. Not in dictionary and has low probability
        3. Matches known OCR error patterns

        Args:
            text: Input text

        Returns:
            List of (token, position, context) tuples
        """
        suspicious = []
        words = re.finditer(r'\b\w+\b', text)

        for match in words:
            token = match.group(0).lower()
            position = match.start()

            # Get context window (50 chars before/after)
            context_start = max(0, position - 50)
            context_end = min(len(text), position + len(token) + 50)
            context = text[context_start:context_end]

            # Check heuristics
            is_suspicious = False

            # Heuristic 1: Contains digits in unusual positions
            if re.search(r'\d', token) and token not in ["1st", "2nd", "3rd"]:
                is_suspicious = True

            # Heuristic 2: Matches OCR error patterns
            for pattern, _ in self.ocr_patterns:
                if pattern.search(token):
                    is_suspicious = True
                    break

            # Heuristic 3: Not in legal dictionary (for legal terms only)
            if len(token) > 5 and token not in self.legal_dictionary:
                # Check if it's close to a dictionary word
                for dict_word in self.legal_dictionary:
                    if self._levenshtein_distance(token, dict_word) <= 2:
                        is_suspicious = True
                        break

            if is_suspicious:
                suspicious.append((token, position, context))

        return suspicious

    def _predict_correction(
        self,
        token: str,
        context: str,
        aggressive: bool = False,
    ) -> Optional[Dict[str, any]]:
        """
        Predict correction for a token using BERT masked LM.

        Args:
            token: Suspicious token
            context: Surrounding context
            aggressive: Use lower confidence threshold

        Returns:
            {
                "token": str,         # Corrected token
                "confidence": float   # Confidence (0-1)
            }
            or None if no good correction found
        """
        # Replace token with [MASK]
        masked_text = context.replace(token, self.tokenizer.mask_token)

        # Tokenize
        inputs = self.tokenizer(
            masked_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits

        # Find [MASK] position
        mask_token_index = (
            inputs["input_ids"] == self.tokenizer.mask_token_id
        ).nonzero(as_tuple=True)[1]

        if len(mask_token_index) == 0:
            return None

        # Get top predictions for masked token
        mask_token_logits = predictions[0, mask_token_index[0], :]
        top_k = torch.topk(mask_token_logits, k=10)

        # Convert to tokens
        top_tokens = [
            self.tokenizer.decode([token_id]).strip()
            for token_id in top_k.indices
        ]
        top_probs = torch.softmax(top_k.values, dim=0).tolist()

        # Find best candidate
        threshold = 0.5 if aggressive else self.confidence_threshold

        for predicted_token, prob in zip(top_tokens, top_probs):
            # Skip if same as original
            if predicted_token.lower() == token.lower():
                continue

            # Check confidence
            if prob < threshold:
                continue

            # Check Levenshtein distance (should be similar)
            distance = self._levenshtein_distance(token, predicted_token)
            if distance > self.levenshtein_threshold:
                continue

            # Found good candidate
            return {
                "token": predicted_token,
                "confidence": prob,
            }

        return None

    def _apply_corrections(
        self,
        text: str,
        errors: List[OCRError],
    ) -> str:
        """
        Apply corrections to text.

        Args:
            text: Original text
            errors: List of detected errors

        Returns:
            Corrected text
        """
        # Sort errors by position (descending) to avoid position shifts
        errors_sorted = sorted(errors, key=lambda e: e.position, reverse=True)

        corrected = text
        for error in errors_sorted:
            # Find exact match at position
            start = error.position
            end = start + len(error.original)

            if corrected[start:end].lower() == error.original.lower():
                # Preserve original case
                if corrected[start:end].isupper():
                    replacement = error.corrected.upper()
                elif corrected[start:end][0].isupper():
                    replacement = error.corrected.capitalize()
                else:
                    replacement = error.corrected

                corrected = corrected[:start] + replacement + corrected[end:]

        return corrected

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def batch_correct(
        self,
        texts: List[str],
        aggressive: bool = False,
    ) -> List[Dict[str, any]]:
        """
        Correct multiple texts (batched for efficiency).

        Args:
            texts: List of texts to correct
            aggressive: Use lower confidence threshold

        Returns:
            List of correction results
        """
        return [self.correct(text, aggressive=aggressive) for text in texts]


# Example usage
if __name__ == "__main__":
    # Example OCR-corrupted legal text
    corrupted_text = """
    The p1aintiff fi1ed a moti0n for summary judgment in the district c0urt.
    The judg3 sustained the defendant's obj3ction to the admission of evidence.
    The case involved a dispute over a fiduciary duty owed by the trustee.
    """

    # Initialize corrector
    corrector = OCRCorrector()

    # Correct text
    result = corrector.correct(corrupted_text)

    # Print results
    print("Original Text:")
    print(result["raw_text"])
    print("\nCorrected Text:")
    print(result["cleaned_text"])
    print(f"\nDetected {result['correction_count']} errors:")
    for error in result["errors"]:
        print(f"  '{error.original}' → '{error.corrected}' (confidence: {error.confidence:.2f})")
