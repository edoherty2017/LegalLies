"""
Legal-Aware Semantic Chunker.

Implements the semantic chunking strategy described in Section 4.1, 6.1.1, and 11.1.1
of the legal RAG framework. Unlike naive fixed-size chunking, this preserves legal
structure and meaning.

Key Features:
- **Header Prepending**: "Motion for Summary Judgment" stays with "Motion denied"
- **Opinion Type Detection**: Majority vs. Concurrence vs. Dissent
- **Section-Based Chunking**: Facts, Law, Analysis, Conclusion
- **Metadata Enrichment**: Jurisdiction, year, court, citation

The Failure of Fixed-Size Chunking:
```
Bad:  Chunk 1: "...Motion for Summary Judgment."
      Chunk 2: "The motion is denied."  # Missing context!

Good: Chunk 1: "Motion for Summary Judgment: ...the motion is denied."
```

References:
- Section 4.1: Structural vs. Sliding Window Chunking
- Section 6.1.1: Harvard Caselaw Access Project
- Section 11.1.1: Case Law Normalization
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class OpinionType(Enum):
    """Type of judicial opinion."""

    MAJORITY = "majority"
    CONCURRENCE = "concurrence"
    DISSENT = "dissent"
    PER_CURIAM = "per_curiam"
    UNKNOWN = "unknown"


class SectionType(Enum):
    """Type of legal document section."""

    FACTS = "facts"
    LAW = "law"
    ANALYSIS = "analysis"
    CONCLUSION = "conclusion"
    PROCEDURAL = "procedural"
    UNKNOWN = "unknown"


@dataclass
class LegalChunk:
    """
    A semantically meaningful chunk of legal text.

    Includes full context and metadata for accurate retrieval.
    """

    text: str  # The chunk text (with prepended header if applicable)
    raw_text: str  # Original text without header
    header: Optional[str]  # Section header (e.g., "Motion for Summary Judgment")
    opinion_type: OpinionType  # Majority, dissent, etc.
    section_type: SectionType  # Facts, law, analysis, etc.
    metadata: Dict[str, Any]  # Document metadata (jurisdiction, court, date, etc.)
    chunk_id: int  # Position in document
    char_start: int  # Start position in original document
    char_end: int  # End position in original document


class LegalChunker:
    """
    Legal-aware document chunker.

    Preserves legal structure and meaning through semantic chunking.
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,  # Characters per chunk
        overlap: int = 100,  # Overlap between chunks (for continuity)
        min_chunk_size: int = 200,  # Minimum chunk size
    ):
        """
        Initialize legal chunker.

        Args:
            max_chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size in characters
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for opinion and section detection."""
        # Opinion type patterns
        self.dissent_patterns = [
            r"(?:JUSTICE|JUDGE)\s+\w+,?\s+dissenting",
            r"I\s+dissent",
            r"dissenting\s+opinion",
            r"In\s+dissent",
        ]

        self.concurrence_patterns = [
            r"(?:JUSTICE|JUDGE)\s+\w+,?\s+concurring",
            r"I\s+concur",
            r"concurring\s+opinion",
            r"concurring\s+in\s+(?:part|judgment)",
        ]

        self.per_curiam_patterns = [
            r"PER\s+CURIAM",
            r"By\s+the\s+Court",
        ]

        # Section markers
        self.section_patterns = {
            SectionType.FACTS: [
                r"(?:^|\n)\s*(?:FACTS?|BACKGROUND)",
                r"(?:^|\n)\s*I\.\s+FACTS?",
            ],
            SectionType.LAW: [
                r"(?:^|\n)\s*(?:LAW|LEGAL\s+STANDARD)",
                r"(?:^|\n)\s*II\.\s+LAW",
            ],
            SectionType.ANALYSIS: [
                r"(?:^|\n)\s*(?:ANALYSIS|DISCUSSION)",
                r"(?:^|\n)\s*III\.\s+ANALYSIS",
            ],
            SectionType.CONCLUSION: [
                r"(?:^|\n)\s*(?:CONCLUSION|HOLDING)",
                r"(?:^|\n)\s*IV\.\s+CONCLUSION",
                r"It\s+is\s+so\s+ordered",
            ],
            SectionType.PROCEDURAL: [
                r"(?:^|\n)\s*(?:PROCEDURAL\s+HISTORY|PROCEDURE)",
            ],
        }

        # Compile patterns
        self.dissent_regex = [re.compile(p, re.IGNORECASE) for p in self.dissent_patterns]
        self.concurrence_regex = [re.compile(p, re.IGNORECASE) for p in self.concurrence_patterns]
        self.per_curiam_regex = [re.compile(p, re.IGNORECASE) for p in self.per_curiam_patterns]

        self.section_regex = {
            section_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for section_type, patterns in self.section_patterns.items()
        }

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[LegalChunk]:
        """
        Chunk legal document with semantic awareness.

        Args:
            text: Full document text
            metadata: Document metadata (jurisdiction, court, date, etc.)

        Returns:
            List of legal chunks with enriched metadata
        """
        if not text:
            return []

        metadata = metadata or {}

        # Step 1: Detect opinion type for entire document
        opinion_type = self._detect_opinion_type(text)

        # Step 2: Split into structural sections
        sections = self._split_into_sections(text)

        # Step 3: Chunk each section
        chunks = []
        chunk_id = 0

        for section_header, section_text, section_type, char_start in sections:
            # Split long sections into chunks
            section_chunks = self._chunk_text(section_text, self.max_chunk_size, self.overlap)

            for chunk_text, chunk_start, chunk_end in section_chunks:
                # Prepend header to chunk (critical for context)
                if section_header:
                    full_text = f"{section_header}\n\n{chunk_text}"
                else:
                    full_text = chunk_text

                chunk = LegalChunk(
                    text=full_text,
                    raw_text=chunk_text,
                    header=section_header,
                    opinion_type=opinion_type,
                    section_type=section_type,
                    metadata=metadata,
                    chunk_id=chunk_id,
                    char_start=char_start + chunk_start,
                    char_end=char_start + chunk_end,
                )

                chunks.append(chunk)
                chunk_id += 1

        return chunks

    def _detect_opinion_type(self, text: str) -> OpinionType:
        """
        Detect opinion type (majority, dissent, concurrence).

        Args:
            text: Full document text

        Returns:
            OpinionType
        """
        # Check for dissent markers
        for pattern in self.dissent_regex:
            if pattern.search(text):
                return OpinionType.DISSENT

        # Check for concurrence markers
        for pattern in self.concurrence_regex:
            if pattern.search(text):
                return OpinionType.CONCURRENCE

        # Check for per curiam
        for pattern in self.per_curiam_regex:
            if pattern.search(text):
                return OpinionType.PER_CURIAM

        # Default to majority
        return OpinionType.MAJORITY

    def _split_into_sections(
        self, text: str
    ) -> List[Tuple[Optional[str], str, SectionType, int]]:
        """
        Split document into structural sections.

        Returns:
            List of (header, text, section_type, char_start) tuples
        """
        sections = []

        # Find all section markers
        markers = []
        for section_type, patterns in self.section_regex.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    markers.append((match.start(), match.group(0), section_type))

        # Sort by position
        markers.sort(key=lambda x: x[0])

        # Extract sections
        if not markers:
            # No sections found, treat as single section
            return [(None, text, SectionType.UNKNOWN, 0)]

        for i, (start, header, section_type) in enumerate(markers):
            # Determine end of section
            if i < len(markers) - 1:
                end = markers[i + 1][0]
            else:
                end = len(text)

            # Extract section text (excluding header)
            section_text = text[start:end].strip()

            # Remove header from section text
            section_text = section_text[len(header):].strip()

            sections.append((header, section_text, section_type, start))

        return sections

    def _chunk_text(
        self,
        text: str,
        max_size: int,
        overlap: int,
    ) -> List[Tuple[str, int, int]]:
        """
        Chunk text with overlap.

        Args:
            text: Text to chunk
            max_size: Maximum chunk size
            overlap: Overlap between chunks

        Returns:
            List of (chunk_text, start, end) tuples
        """
        if len(text) <= max_size:
            return [(text, 0, len(text))]

        chunks = []
        start = 0

        while start < len(text):
            # Determine chunk end
            end = min(start + max_size, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence-ending punctuation within last 20% of chunk
                search_start = int(end * 0.8)
                sentence_end = self._find_sentence_boundary(
                    text, search_start, end
                )
                if sentence_end > start:
                    end = sentence_end

            # Extract chunk
            chunk_text = text[start:end].strip()

            # Only add if meets minimum size
            if len(chunk_text) >= self.min_chunk_size or start == 0:
                chunks.append((chunk_text, start, end))

            # Move start position (with overlap)
            start = end - overlap

        return chunks

    def _find_sentence_boundary(
        self, text: str, start: int, end: int
    ) -> int:
        """
        Find sentence boundary within range.

        Args:
            text: Full text
            start: Start of search range
            end: End of search range

        Returns:
            Position of sentence boundary, or end if not found
        """
        # Sentence-ending punctuation (but not legal abbreviations)
        sentence_pattern = r'[.!?]\s+(?=[A-Z])'

        # Search backwards from end
        for match in re.finditer(sentence_pattern, text[start:end]):
            # Return position after punctuation
            return start + match.end()

        # No sentence boundary found
        return end

    def to_dict(self, chunk: LegalChunk) -> Dict[str, Any]:
        """
        Convert chunk to dictionary for serialization.

        Args:
            chunk: LegalChunk instance

        Returns:
            Dictionary representation
        """
        return {
            "text": chunk.text,
            "raw_text": chunk.raw_text,
            "header": chunk.header,
            "opinion_type": chunk.opinion_type.value,
            "section_type": chunk.section_type.value,
            "metadata": chunk.metadata,
            "chunk_id": chunk.chunk_id,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
        }


# Example usage
if __name__ == "__main__":
    # Example legal opinion
    opinion = """
    SUPREME COURT OF THE UNITED STATES

    No. 12-345

    JOHNSON v. UNITED STATES

    Opinion delivered June 15, 2020

    JUSTICE SMITH delivered the opinion of the Court.

    I. FACTS

    The petitioner, John Johnson, was convicted of wire fraud under 18 U.S.C. § 1343.
    The evidence showed that Johnson operated a scheme to defraud investors through
    false representations about the profitability of his business ventures.

    II. ANALYSIS

    The question presented is whether the evidence was sufficient to support the
    conviction. We hold that it was. The standard for sufficiency of evidence
    requires that a reasonable jury could find the defendant guilty beyond a
    reasonable doubt.

    III. CONCLUSION

    For the foregoing reasons, we affirm the judgment of the Court of Appeals.

    It is so ordered.

    JUSTICE BROWN, dissenting.

    I respectfully dissent. The majority's analysis overlooks critical flaws in
    the prosecution's case. The evidence presented was entirely circumstantial
    and insufficient to prove intent beyond a reasonable doubt.
    """

    # Initialize chunker
    chunker = LegalChunker(max_chunk_size=500)

    # Chunk the opinion
    metadata = {
        "case_name": "Johnson v. United States",
        "court": "Supreme Court",
        "year": 2020,
        "citation": "No. 12-345",
    }

    chunks = chunker.chunk(opinion, metadata)

    # Print results
    print(f"Created {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"Chunk {chunk.chunk_id}:")
        print(f"  Opinion Type: {chunk.opinion_type.value}")
        print(f"  Section Type: {chunk.section_type.value}")
        print(f"  Header: {chunk.header}")
        print(f"  Text (first 100 chars): {chunk.text[:100]}...")
        print()
