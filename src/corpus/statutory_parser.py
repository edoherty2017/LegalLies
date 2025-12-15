"""
Statutory Parser with Breadcrumb Context Injection.

Implements the hierarchical context preservation strategy described in
Section 2.1.2, 6.2.1, and 11.1.2 of the legal RAG framework.

The Problem:
Statutes are highly structured and cross-referential. Section 504 might
reference a term defined in Section 101. If we chunk Section 504 in isolation,
the semantic meaning is lost.

Example:
```
Section 101: "Person" includes corporations.
...
Section 504: Penalties apply to any person who...
```

If we chunk Section 504 alone, the model doesn't know "person" includes corporations!

Solution:
**Breadcrumb Injection** - Prepend hierarchical context to every chunk:
```
Title 1, Chapter 1, Section 101 (Definition): "Person" includes corporations.
Title 1, Chapter 5, Section 504: Penalties apply to any person who...
                                 ^^^^^ Now includes definition context!
```

References:
- Section 2.1.2: Statutory Frameworks
- Section 6.2.1: The United States Code (USC) & CFR
- Section 11.1.2: Statutory Normalization
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class StatuteType(Enum):
    """Type of statutory source."""

    USC = "usc"              # United States Code
    CFR = "cfr"              # Code of Federal Regulations
    STATE = "state"          # State statute
    MUNICIPAL = "municipal"  # Municipal code


@dataclass
class StatutorySection:
    """A section of statutory text with hierarchical context."""

    # Hierarchy
    title: Optional[str]        # Title number (e.g., "17")
    chapter: Optional[str]      # Chapter number (e.g., "1")
    section: str                # Section number (e.g., "101")
    subsection: Optional[str]   # Subsection (e.g., "(a)(1)")

    # Content
    text: str                   # Section text
    heading: Optional[str]      # Section heading

    # Context
    definitions: List[str]      # Referenced definitions
    breadcrumb: str             # Full hierarchical path

    # Metadata
    statute_type: StatuteType
    effective_date: Optional[str]  # When this version became effective
    amended_date: Optional[str]    # When it was last amended


class StatutoryParser:
    """
    Parse statutory text and inject breadcrumb context.

    Handles USC, CFR, and state statutes with hierarchical structure preservation.
    """

    def __init__(self):
        """Initialize statutory parser."""
        # Definition markers
        self.definition_markers = [
            r'means',
            r'includes',
            r'shall mean',
            r'is defined as',
            r'for purposes of this (?:section|chapter|title)',
        ]

        # Compile regex for section detection
        self.section_pattern = re.compile(
            r'(?:§|Section)\s*(\d+(?:\.\d+)*)',
            re.IGNORECASE
        )

        # Definition cache (term -> definition section)
        self.definition_cache: Dict[str, str] = {}

    def parse_usc(
        self,
        text: str,
        title: str,
        metadata: Optional[Dict] = None,
    ) -> List[StatutorySection]:
        """
        Parse U.S. Code text.

        USC Structure:
        Title → Subtitle → Chapter → Subchapter → Part → Section → Subsection

        Args:
            text: Full USC text
            title: Title number (e.g., "17" for Copyright)
            metadata: Additional metadata

        Returns:
            List of statutory sections with breadcrumbs
        """
        metadata = metadata or {}

        # Step 1: Split into sections
        sections = self._split_usc_sections(text)

        # Step 2: Extract definitions
        definitions = self._extract_definitions(sections)
        self.definition_cache.update(definitions)

        # Step 3: Parse each section with context
        parsed_sections = []
        for section_num, section_text, heading in sections:
            # Determine chapter (simplified - would parse from structure in production)
            chapter = self._infer_chapter(section_num)

            # Find relevant definitions
            relevant_defs = self._find_relevant_definitions(section_text)

            # Build breadcrumb
            breadcrumb = self._build_breadcrumb(
                title=title,
                chapter=chapter,
                section=section_num,
                heading=heading,
            )

            # Inject definition context
            enriched_text = self._inject_definitions(section_text, relevant_defs)

            parsed_sections.append(StatutorySection(
                title=title,
                chapter=chapter,
                section=section_num,
                subsection=None,
                text=enriched_text,
                heading=heading,
                definitions=relevant_defs,
                breadcrumb=breadcrumb,
                statute_type=StatuteType.USC,
                effective_date=metadata.get("effective_date"),
                amended_date=metadata.get("amended_date"),
            ))

        return parsed_sections

    def parse_cfr(
        self,
        text: str,
        title: str,
        part: str,
        metadata: Optional[Dict] = None,
    ) -> List[StatutorySection]:
        """
        Parse Code of Federal Regulations text.

        CFR Structure:
        Title → Subtitle → Chapter → Subchapter → Part → Subpart → Section

        Args:
            text: Full CFR text
            title: Title number (e.g., "37" for Patents)
            part: Part number (e.g., "1")
            metadata: Additional metadata

        Returns:
            List of statutory sections with breadcrumbs
        """
        metadata = metadata or {}

        # Similar to USC, but with Part structure
        sections = self._split_cfr_sections(text)
        definitions = self._extract_definitions(sections)
        self.definition_cache.update(definitions)

        parsed_sections = []
        for section_num, section_text, heading in sections:
            relevant_defs = self._find_relevant_definitions(section_text)

            breadcrumb = self._build_breadcrumb_cfr(
                title=title,
                part=part,
                section=section_num,
                heading=heading,
            )

            enriched_text = self._inject_definitions(section_text, relevant_defs)

            parsed_sections.append(StatutorySection(
                title=title,
                chapter=part,  # CFR uses "part" instead of "chapter"
                section=section_num,
                subsection=None,
                text=enriched_text,
                heading=heading,
                definitions=relevant_defs,
                breadcrumb=breadcrumb,
                statute_type=StatuteType.CFR,
                effective_date=metadata.get("effective_date"),
                amended_date=metadata.get("amended_date"),
            ))

        return parsed_sections

    def _split_usc_sections(
        self, text: str
    ) -> List[Tuple[str, str, Optional[str]]]:
        """
        Split USC text into sections.

        Returns:
            List of (section_number, section_text, heading) tuples
        """
        sections = []

        # Pattern: § 101. Heading
        pattern = re.compile(
            r'§\s*(\d+)\.\s*([^\n]+)\n(.*?)(?=§\s*\d+\.|$)',
            re.DOTALL
        )

        for match in pattern.finditer(text):
            section_num = match.group(1)
            heading = match.group(2).strip()
            section_text = match.group(3).strip()
            sections.append((section_num, section_text, heading))

        return sections

    def _split_cfr_sections(
        self, text: str
    ) -> List[Tuple[str, str, Optional[str]]]:
        """
        Split CFR text into sections.

        Returns:
            List of (section_number, section_text, heading) tuples
        """
        # CFR sections are similar to USC
        # Pattern: § 1.101 Heading
        pattern = re.compile(
            r'§\s*(\d+\.\d+)\s+([^\n]+)\n(.*?)(?=§\s*\d+\.\d+|$)',
            re.DOTALL
        )

        sections = []
        for match in pattern.finditer(text):
            section_num = match.group(1)
            heading = match.group(2).strip()
            section_text = match.group(3).strip()
            sections.append((section_num, section_text, heading))

        return sections

    def _extract_definitions(
        self, sections: List[Tuple[str, str, Optional[str]]]
    ) -> Dict[str, str]:
        """
        Extract definitions from sections.

        Looks for "Definitions" sections or phrases like "means" or "includes".

        Args:
            sections: List of (section_number, text, heading) tuples

        Returns:
            Dictionary of term -> definition
        """
        definitions = {}

        for section_num, text, heading in sections:
            # Check if this is a definitions section
            is_def_section = heading and "definition" in heading.lower()

            if is_def_section:
                # Parse definitions
                defs = self._parse_definition_section(text)
                definitions.update(defs)
            else:
                # Look for inline definitions
                defs = self._parse_inline_definitions(text)
                definitions.update(defs)

        return definitions

    def _parse_definition_section(self, text: str) -> Dict[str, str]:
        """
        Parse a dedicated definitions section.

        Example:
        (1) "Person" means an individual, corporation, or partnership.
        (2) "Work" means a work of authorship.

        Returns:
            Dictionary of term -> definition
        """
        definitions = {}

        # Pattern: (n) "Term" means definition.
        pattern = re.compile(
            r'\((\d+)\)\s+"([^"]+)"\s+(?:means|includes?)\s+([^.]+)\.',
            re.IGNORECASE
        )

        for match in pattern.finditer(text):
            term = match.group(2).lower()
            definition = match.group(3).strip()
            definitions[term] = definition

        return definitions

    def _parse_inline_definitions(self, text: str) -> Dict[str, str]:
        """
        Parse inline definitions (not in dedicated section).

        Example:
        The term "copyright owner", with respect to any exclusive right...

        Returns:
            Dictionary of term -> definition
        """
        definitions = {}

        # Pattern: The term "Term" means definition.
        pattern = re.compile(
            r'(?:The term|As used in this section,?)\s+"([^"]+)"\s+(?:means|includes?)\s+([^.]+)\.',
            re.IGNORECASE
        )

        for match in pattern.finditer(text):
            term = match.group(1).lower()
            definition = match.group(2).strip()
            definitions[term] = definition

        return definitions

    def _find_relevant_definitions(self, text: str) -> List[str]:
        """
        Find definitions relevant to this section.

        Args:
            text: Section text

        Returns:
            List of definition terms referenced in text
        """
        relevant = []

        for term, definition in self.definition_cache.items():
            # Check if term appears in text
            if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                relevant.append(term)

        return relevant

    def _inject_definitions(
        self, text: str, relevant_defs: List[str]
    ) -> str:
        """
        Inject definition context into section text.

        Args:
            text: Section text
            relevant_defs: List of relevant definition terms

        Returns:
            Text with prepended definitions
        """
        if not relevant_defs:
            return text

        # Build definition preamble
        preamble_parts = []
        for term in relevant_defs:
            if term in self.definition_cache:
                definition = self.definition_cache[term]
                preamble_parts.append(f'"{term.title()}" means {definition}.')

        if preamble_parts:
            preamble = "\n".join(preamble_parts)
            return f"[Definitions: {preamble}]\n\n{text}"
        else:
            return text

    def _build_breadcrumb(
        self,
        title: str,
        chapter: Optional[str],
        section: str,
        heading: Optional[str],
    ) -> str:
        """
        Build breadcrumb path.

        Args:
            title: Title number
            chapter: Chapter number
            section: Section number
            heading: Section heading

        Returns:
            Breadcrumb string
        """
        parts = [f"Title {title}"]

        if chapter:
            parts.append(f"Chapter {chapter}")

        parts.append(f"Section {section}")

        if heading:
            parts.append(f"({heading})")

        return " → ".join(parts)

    def _build_breadcrumb_cfr(
        self,
        title: str,
        part: str,
        section: str,
        heading: Optional[str],
    ) -> str:
        """
        Build breadcrumb path for CFR.

        Args:
            title: Title number
            part: Part number
            section: Section number
            heading: Section heading

        Returns:
            Breadcrumb string
        """
        parts = [f"CFR Title {title}"]
        parts.append(f"Part {part}")
        parts.append(f"Section {section}")

        if heading:
            parts.append(f"({heading})")

        return " → ".join(parts)

    def _infer_chapter(self, section_num: str) -> Optional[str]:
        """
        Infer chapter from section number.

        USC chapters group sections by range (e.g., §§ 101-110 = Chapter 1).
        This is simplified; production would use official structure.

        Args:
            section_num: Section number

        Returns:
            Chapter number
        """
        section_int = int(section_num)

        # Simplified mapping (would use official USC structure)
        if section_int < 200:
            return "1"
        elif section_int < 300:
            return "2"
        else:
            return str(section_int // 100)


# Example usage
if __name__ == "__main__":
    # Example USC text (Title 17 - Copyright)
    usc_text = """
    § 101. Definitions

    (1) "Person" means an individual, corporation, partnership, or association.
    (2) "Work" means a work of authorship fixed in any tangible medium of expression.
    (3) "Copyright owner" means the owner of any exclusive right under a copyright.

    § 106. Exclusive rights in copyrighted works

    Subject to sections 107 through 122, the owner of copyright under this title
    has the exclusive rights to do and to authorize any of the following:
    (1) to reproduce the copyrighted work in copies or phonorecords;
    (2) to prepare derivative works based upon the copyrighted work;
    (3) to distribute copies or phonorecords of the copyrighted work to the public;
    """

    # Initialize parser
    parser = StatutoryParser()

    # Parse USC
    sections = parser.parse_usc(
        text=usc_text,
        title="17",
        metadata={"effective_date": "2021-01-01"}
    )

    # Print results
    for section in sections:
        print(f"\n{'='*70}")
        print(f"Breadcrumb: {section.breadcrumb}")
        print(f"Heading: {section.heading}")
        print(f"Definitions: {section.definitions}")
        print(f"\nEnriched Text:\n{section.text[:300]}...")
