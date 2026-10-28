"""
Citation Graph Builder with Authority Ranking.

Implements the citation-based authority system described in Section 3.2.2
and 11.2.1 of the legal RAG framework. Uses PageRank to score document
authority, then weights retrieval results by authority.

The Problem:
Vector similarity alone doesn't distinguish authoritative precedents from
obscure district court rulings. We need to weight results by legal authority.

Example:
- Query: "What is qualified immunity?"
- Result 1: Harlow v. Fitzgerald (Supreme Court, cited 10,000+ times) ⭐⭐⭐
- Result 2: Smith v. Jones (District Court, cited 2 times) ⭐

Both might have similar vector similarity, but Harlow is FAR more authoritative!

Solution:
**Citation Graph + PageRank** - Build directed graph where edges represent
citations, then use PageRank to score authority. This is exactly how Google
ranks webpages, adapted for legal citations.

Hybrid Ranking Formula:
```
final_score = (0.7 × vector_similarity) + (0.3 × authority_score)
```

References:
- Section 3.2.2: Citation Extraction and Graph Construction
- Section 11.2.1: Citation Graph Construction
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path

import networkx as nx
import numpy as np


@dataclass
class Citation:
    """A legal citation extracted from a document."""

    # Citation components
    case_name: Optional[str] = None      # "Smith v. Jones"
    volume: Optional[str] = None         # "500"
    reporter: Optional[str] = None       # "U.S."
    page: Optional[str] = None           # "123"
    year: Optional[str] = None           # "2000"

    # Citation string
    raw_citation: str = ""               # Full citation string
    citation_type: str = "case"          # "case", "statute", "regulation"

    # Context
    context: str = ""                    # Surrounding text
    position: int = 0                    # Character position in document

    # Parsed fields
    normalized_cite: Optional[str] = None  # "500-U.S.-123"

    def __post_init__(self):
        """Generate normalized citation."""
        if self.volume and self.reporter and self.page:
            self.normalized_cite = f"{self.volume}-{self.reporter}-{self.page}"


@dataclass
class DocumentNode:
    """A document node in the citation graph."""

    doc_id: str                          # Unique document identifier
    title: str                           # Document title
    citation: Optional[str] = None       # Canonical citation (e.g., "500 U.S. 123")

    # Metadata
    court: Optional[str] = None          # "Supreme Court", "9th Circuit"
    year: Optional[int] = None           # Year decided
    doc_type: str = "case"               # "case", "statute", "regulation"

    # Graph metrics (computed)
    out_degree: int = 0                  # Number of outgoing citations
    in_degree: int = 0                   # Number of incoming citations
    pagerank: float = 0.0                # PageRank authority score

    # Citations
    cited_docs: List[str] = field(default_factory=list)      # Documents this cites
    citing_docs: List[str] = field(default_factory=list)     # Documents citing this


def _canonical_cite(cite: str) -> str:
    """
    Reduce a case citation to a comparison key.

    "457 U.S. 800", "457-U.S.-800" and "457  u.s.  800" all become "457us800",
    so citations extracted from text can be matched against the citation a
    document was registered with regardless of spacing, hyphenation, internal
    periods or case.
    """
    if not cite:
        return ""
    return "".join(ch for ch in cite.lower() if ch.isalnum())


# Regex fragments used to build the reporter alternation.
PERIOD = "\\."
WS = "\\s*"
ESCAPED_SPACE = "\\ "


class CitationExtractor:
    """
    Extract legal citations from text using regex patterns.

    Supports:
    - Case citations (e.g., "500 U.S. 123")
    - Statutory citations (e.g., "17 U.S.C. § 106")
    - Regulatory citations (e.g., "37 C.F.R. § 1.1")
    """

    def __init__(self):
        """Initialize citation extractor with regex patterns."""
        # Reporter abbreviations. Loaded first because the case-citation
        # regex is built from them.
        self.reporters = self._load_reporters()

        # Compile citation patterns
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """
        Compile regex patterns for different citation types.

        Returns:
            Dictionary of pattern_name -> compiled regex
        """
        patterns = {
            # Case citations: "500 U.S. 123" or "500 U.S. 123 (2000)"
            # Case citations: "500 U.S. 123", "123 F.3d 456 (1999)",
            # "123 S. Ct. 456", "500 F. Supp. 2d 100", "99 L. Ed. 2d 1".
            #
            # Built from self.reporters rather than a generic character class.
            # The original group, ([A-Z][A-Za-z.]*\.?), admitted neither digits
            # nor internal spaces: for "123 F.3d 456" it matched only "F." and
            # then failed on the "3", and "S. Ct." / "F. Supp." failed on the
            # space. Almost no circuit, district or regional citation produced
            # an edge, so PageRank stayed near uniform and the 30% authority
            # weight in HybridRanker contributed only noise.
            "case_standard": re.compile(
                r'\b(\d+)\s+(' + self._reporter_alternation() + r')\s+(\d+)'
                r'(?:\s+\((\d{4})\))?',
            ),

            # U.S. Code: "17 U.S.C. § 106" or "17 USC 106"
            "usc": re.compile(
                r'\b(\d+)\s+U\.?S\.?C\.?\s+§?\s*(\d+)',
                re.IGNORECASE,
            ),

            # CFR: "37 C.F.R. § 1.1" or "37 CFR 1.1"
            "cfr": re.compile(
                r'\b(\d+)\s+C\.?F\.?R\.?\s+§?\s*(\d+(?:\.\d+)?)',
                re.IGNORECASE,
            ),

            # Public Law: "Pub. L. 117-58"
            "public_law": re.compile(
                r'\bPub\.?\s*L\.?\s*(\d+-\d+)',
                re.IGNORECASE,
            ),

            # Case name references: "Smith v. Jones"
            "case_name": re.compile(
                r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\s+v\.?\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)',
            ),
        }

        return patterns

    def _load_reporters(self) -> Set[str]:
        """
        Load valid reporter abbreviations.

        Returns:
            Set of reporter abbreviations
        """
        # Common reporters (expandable)
        reporters = {
            # Supreme Court
            "U.S.", "S.Ct.", "L.Ed.",

            # Federal Reporters
            "F.", "F.2d", "F.3d", "F.4th",          # Circuit courts
            "F.Supp.", "F.Supp.2d", "F.Supp.3d",    # District courts

            # State reporters (examples)
            "Cal.", "Cal.2d", "Cal.3d", "Cal.4th",  # California
            "N.Y.", "N.Y.2d", "N.Y.3d",             # New York
            "Tex.", "Ill.", "Pa.",                  # Other states

            # Regional reporters
            "A.", "A.2d", "A.3d",                   # Atlantic
            "P.", "P.2d", "P.3d",                   # Pacific
            "N.E.", "N.E.2d", "N.E.3d",             # North Eastern
            "S.E.", "S.E.2d",                       # South Eastern
            "S.W.", "S.W.2d", "S.W.3d",             # South Western
            "N.W.", "N.W.2d", "N.W.3d",             # North Western

            # Typeset (spaced) variants and series the corpus actually contains
            "S. Ct.", "L. Ed.", "L. Ed. 2d", "L.Ed.2d",
            "F. Supp.", "F. Supp. 2d", "F. Supp. 3d",
            "F. App'x", "F.App'x", "Fed. Cl.", "Fed. Appx.",
            "So.", "So.2d", "So.3d", "Cal. App.", "Cal.App.",
            "Ohio St.", "Ohio St.2d", "Ohio St.3d", "Wis.2d",
            "Mass.", "N.J.", "Mich.", "Minn.", "Md.", "Va.", "Ga.", "Fla.",
        }

        return reporters

    def _reporter_alternation(self) -> str:
        """
        Build a regex alternation matching any known reporter.

        Longest first, so "F. Supp. 2d" is preferred over "F." and "U.S.C." is
        never shortened to "U.S.". Every literal period may be followed by
        optional whitespace, and any space inside an entry becomes optional
        whitespace, so a single list entry matches both the canonical
        "F.Supp.2d" and the typeset "F. Supp. 2d".
        """
        parts = []
        for reporter in sorted(self.reporters, key=len, reverse=True):
            pattern = re.escape(reporter)
            # Allow optional whitespace after each literal period, and treat any
            # space inside the entry as optional whitespace.
            pattern = pattern.replace(PERIOD, PERIOD + WS)
            pattern = pattern.replace(ESCAPED_SPACE, WS).replace(" ", WS)
            parts.append(pattern)
        return "(?:" + "|".join(parts) + ")"

    def extract(self, text: str, doc_id: str) -> List[Citation]:
        """
        Extract all citations from text.

        Args:
            text: Document text
            doc_id: Document identifier

        Returns:
            List of extracted citations
        """
        citations = []

        # Extract case citations
        case_cites = self._extract_case_citations(text)
        citations.extend(case_cites)

        # Extract statutory citations
        usc_cites = self._extract_usc_citations(text)
        citations.extend(usc_cites)

        # Extract regulatory citations
        cfr_cites = self._extract_cfr_citations(text)
        citations.extend(cfr_cites)

        # Extract public law citations
        pub_law_cites = self._extract_public_law_citations(text)
        citations.extend(pub_law_cites)

        return citations

    def _extract_case_citations(self, text: str) -> List[Citation]:
        """Extract case citations from text."""
        citations = []

        for match in self.patterns["case_standard"].finditer(text):
            volume = match.group(1)
            reporter = match.group(2)
            page = match.group(3)
            year = match.group(4) if match.group(4) else None

            # Validate reporter
            if reporter not in self.reporters:
                continue

            # Get context
            context_start = max(0, match.start() - 50)
            context_end = min(len(text), match.end() + 50)
            context = text[context_start:context_end]

            citation = Citation(
                volume=volume,
                reporter=reporter,
                page=page,
                year=year,
                raw_citation=match.group(0),
                citation_type="case",
                context=context,
                position=match.start(),
            )

            citations.append(citation)

        return citations

    def _extract_usc_citations(self, text: str) -> List[Citation]:
        """Extract U.S. Code citations from text."""
        citations = []

        for match in self.patterns["usc"].finditer(text):
            title = match.group(1)
            section = match.group(2)

            # Get context
            context_start = max(0, match.start() - 50)
            context_end = min(len(text), match.end() + 50)
            context = text[context_start:context_end]

            citation = Citation(
                volume=title,      # Title is like "volume"
                reporter="U.S.C.",
                page=section,      # Section is like "page"
                raw_citation=match.group(0),
                citation_type="statute",
                context=context,
                position=match.start(),
            )

            citations.append(citation)

        return citations

    def _extract_cfr_citations(self, text: str) -> List[Citation]:
        """Extract CFR citations from text."""
        citations = []

        for match in self.patterns["cfr"].finditer(text):
            title = match.group(1)
            section = match.group(2)

            # Get context
            context_start = max(0, match.start() - 50)
            context_end = min(len(text), match.end() + 50)
            context = text[context_start:context_end]

            citation = Citation(
                volume=title,
                reporter="C.F.R.",
                page=section,
                raw_citation=match.group(0),
                citation_type="regulation",
                context=context,
                position=match.start(),
            )

            citations.append(citation)

        return citations

    def _extract_public_law_citations(self, text: str) -> List[Citation]:
        """Extract public law citations from text."""
        citations = []

        for match in self.patterns["public_law"].finditer(text):
            pub_law_num = match.group(1)

            # Get context
            context_start = max(0, match.start() - 50)
            context_end = min(len(text), match.end() + 50)
            context = text[context_start:context_end]

            citation = Citation(
                volume=pub_law_num,
                reporter="Pub.L.",
                raw_citation=match.group(0),
                citation_type="statute",
                context=context,
                position=match.start(),
            )

            citations.append(citation)

        return citations


class CitationGraph:
    """
    Citation graph with PageRank-based authority scoring.

    Builds a directed graph where:
    - Nodes = legal documents
    - Edges = citations (A → B means A cites B)

    Uses NetworkX for graph operations and PageRank calculation.
    """

    def __init__(self):
        """Initialize citation graph."""
        # NetworkX directed graph
        self.graph = nx.DiGraph()

        # Citations that did not resolve when their citing document was added,
        # because the cited document had not been added yet. Held as
        # (citing_doc_id, Citation) and retried whenever a new document arrives
        # and again in compute_pagerank. Without this, edge creation depended on
        # insertion order: a 2020 opinion citing Harlow added before Harlow
        # itself silently lost the edge for good, so landmark cases ended up with
        # near-minimum PageRank - inverting the authority boost ranking relies on.
        self._pending_citations = []

        # Document metadata
        self.documents: Dict[str, DocumentNode] = {}

        # Citation extractor
        self.extractor = CitationExtractor()

        # PageRank scores (cached)
        self.pagerank_scores: Dict[str, float] = {}

        # Authority tiers (for debugging/analysis)
        self.authority_tiers: Dict[str, str] = {}

    def add_document(
        self,
        doc_id: str,
        title: str,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> DocumentNode:
        """
        Add document to citation graph.

        Args:
            doc_id: Unique document identifier
            title: Document title
            text: Full document text
            metadata: Additional metadata (court, year, etc.)

        Returns:
            DocumentNode
        """
        metadata = metadata or {}

        # Create document node
        node = DocumentNode(
            doc_id=doc_id,
            title=title,
            citation=metadata.get("citation"),
            court=metadata.get("court"),
            year=metadata.get("year"),
            doc_type=metadata.get("doc_type", "case"),
        )

        # Add to graph
        self.graph.add_node(doc_id)
        self.documents[doc_id] = node

        # Extract citations from text
        citations = self.extractor.extract(text, doc_id)

        # Add citation edges, deferring any that cannot be resolved yet.
        for citation in citations:
            if not self._add_citation_edge(doc_id, citation):
                self._pending_citations.append((doc_id, citation))

        # This new document may resolve citations deferred earlier.
        self._resolve_pending_citations()

        # Update degrees
        self._refresh_degrees()

        # Invalidate cached PageRank
        self.pagerank_scores = {}

        return node

    def _add_citation_edge(self, citing_doc_id: str, citation: Citation) -> bool:
        """
        Add one citation edge if the cited document is known.

        Returns:
            True if the citation was handled (edge added, or deliberately
            skipped as a self-citation); False if it should be retried later.
        """
        cited_doc_id = self._resolve_citation(citation)

        if not cited_doc_id:
            # Distinguish "cannot ever resolve" from "not resolvable yet".
            # A citation with no normalized form is malformed, so retrying is
            # pointless; one that simply has no matching document yet must be
            # retried once more documents arrive, which is the whole point of
            # the pending list.
            return not citation.normalized_cite

        if cited_doc_id == citing_doc_id:
            # Opinions usually restate their own citation in the caption, which
            # produced a self-loop that inflated the document's own PageRank and
            # in/out-degree. A document citing itself conveys no authority.
            return True

        if cited_doc_id not in self.documents:
            return False

        if not self.graph.has_edge(citing_doc_id, cited_doc_id):
            self.graph.add_edge(citing_doc_id, cited_doc_id)
            self.documents[citing_doc_id].cited_docs.append(cited_doc_id)
            self.documents[cited_doc_id].citing_docs.append(citing_doc_id)

        return True

    def _resolve_pending_citations(self) -> int:
        """
        Retry deferred citations now that more documents are known.

        Returns:
            Number of citations resolved by this pass.
        """
        if not self._pending_citations:
            return 0

        still_pending = []
        resolved = 0
        for citing_doc_id, citation in self._pending_citations:
            if self._add_citation_edge(citing_doc_id, citation):
                resolved += 1
            else:
                still_pending.append((citing_doc_id, citation))

        self._pending_citations = still_pending
        return resolved

    def _refresh_degrees(self) -> None:
        """
        Recompute in/out degree for every node.

        Edges can now be added to documents that were inserted earlier, so it is
        not enough to set the degrees of the document being added - the cited
        document's in_degree used to be left stale and was saved that way to
        documents.json.
        """
        for node_id, node in self.documents.items():
            node.out_degree = self.graph.out_degree(node_id)
            node.in_degree = self.graph.in_degree(node_id)

    def _resolve_citation(self, citation: Citation) -> Optional[str]:
        """
        Resolve a citation to a document ID.

        This is a simplified implementation. In production, you would:
        1. Use a citation database (e.g., CourtListener's citation index)
        2. Fuzzy match case names
        3. Handle parallel citations

        Args:
            citation: Citation to resolve

        Returns:
            Document ID if found, None otherwise
        """
        if not citation.normalized_cite:
            return None

        target = _canonical_cite(citation.normalized_cite)
        if not target:
            return None

        # Search for matching document. Both sides are canonicalised first.
        #
        # Citation.normalized_cite is built as "{volume}-{reporter}-{page}", e.g.
        # "457-U.S.-800", while documents are registered with the citation as it
        # appears in the source, e.g. "457 U.S. 800" - which is the form
        # DocumentNode.citation documents and that the corpus builders pass. The
        # previous exact string comparison therefore matched almost nothing, so
        # the graph held virtually no case edges at all, PageRank was near
        # uniform, and the authority component of hybrid ranking carried no
        # information. Comparing canonical keys makes the two spellings meet.
        for doc_id, doc in self.documents.items():
            if doc.citation and _canonical_cite(doc.citation) == target:
                return doc_id

        return None

    def compute_pagerank(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> Dict[str, float]:
        """
        Compute PageRank scores for all documents.

        PageRank Formula:
        PR(A) = (1-d)/N + d * Σ(PR(Ti)/C(Ti))

        Where:
        - d = damping factor (0.85)
        - N = total number of documents
        - Ti = documents citing A
        - C(Ti) = number of outgoing citations from Ti

        Args:
            damping: Damping factor (0-1)
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Dictionary of doc_id -> pagerank score
        """

        # Final pass: documents added after their citers can now be linked.
        self._resolve_pending_citations()
        self._refresh_degrees()
        if not self.graph.nodes():
            return {}

        # Use NetworkX PageRank implementation
        pagerank = nx.pagerank(
            self.graph,
            alpha=damping,
            max_iter=max_iter,
            tol=tol,
        )

        # Update document nodes
        for doc_id, score in pagerank.items():
            if doc_id in self.documents:
                self.documents[doc_id].pagerank = score

        # Cache scores
        self.pagerank_scores = pagerank

        # Compute authority tiers
        self._compute_authority_tiers()

        return pagerank

    def _compute_authority_tiers(self):
        """
        Compute authority tiers for documents.

        Tiers:
        - Top 5%: "Landmark" (e.g., Brown v. Board, Roe v. Wade)
        - Top 20%: "Highly Authoritative"
        - Top 50%: "Authoritative"
        - Bottom 50%: "Standard"
        """
        if not self.pagerank_scores:
            return

        # Sort by PageRank
        sorted_docs = sorted(
            self.pagerank_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        n = len(sorted_docs)

        for i, (doc_id, score) in enumerate(sorted_docs):
            percentile = (i + 1) / n

            if percentile <= 0.05:
                tier = "Landmark"
            elif percentile <= 0.20:
                tier = "Highly Authoritative"
            elif percentile <= 0.50:
                tier = "Authoritative"
            else:
                tier = "Standard"

            self.authority_tiers[doc_id] = tier

    def get_authority_score(self, doc_id: str) -> float:
        """
        Get normalized authority score for document.

        Args:
            doc_id: Document identifier

        Returns:
            Authority score (0-1)
        """
        if not self.pagerank_scores:
            self.compute_pagerank()

        # A document outside the graph has no authority information; it must be
        # treated as neutral. Previously it fell back to 0.0 and was then
        # min-max normalized against PageRank scores whose minimum is always
        # above zero (PageRank gives every node at least (1-d)/N), producing a
        # NEGATIVE score. HybridRanker's 0.7*relevance + 0.3*authority then
        # ranked such a passage below the least authoritative passage in the
        # graph - a penalty no document had earned - and could push genuinely
        # relevant evidence out of top_k.
        if doc_id not in self.pagerank_scores:
            return 0.0

        score = self.pagerank_scores[doc_id]

        # Normalize to 0-1 range
        if self.pagerank_scores:
            max_score = max(self.pagerank_scores.values())
            min_score = min(self.pagerank_scores.values())

            if max_score > min_score:
                score = (score - min_score) / (max_score - min_score)
            else:
                # Every node is equally authoritative; say so rather than 0.
                score = 1.0 if score > 0 else 0.0

        # Defensive clamp: the contract is a 0-1 score.
        return max(0.0, min(1.0, score))

    def get_authority_tier(self, doc_id: str) -> str:
        """
        Get authority tier for document.

        Args:
            doc_id: Document identifier

        Returns:
            Tier name ("Landmark", "Highly Authoritative", etc.)
        """
        if not self.authority_tiers:
            self.compute_pagerank()

        return self.authority_tiers.get(doc_id, "Standard")

    def get_top_authorities(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most authoritative documents.

        Args:
            n: Number of documents to return

        Returns:
            List of (doc_id, pagerank_score) tuples
        """
        if not self.pagerank_scores:
            self.compute_pagerank()

        sorted_docs = sorted(
            self.pagerank_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_docs[:n]

    def save(self, path: Path):
        """
        Save citation graph to disk.

        Args:
            path: Save path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save graph (using pickle directly - nx.write_gpickle removed in NetworkX 3.0+)
        import pickle
        with open(path / "graph.gpickle", "wb") as f:
            pickle.dump(self.graph, f)

        # Save documents
        docs_data = {
            doc_id: {
                "title": doc.title,
                "citation": doc.citation,
                "court": doc.court,
                "year": doc.year,
                "doc_type": doc.doc_type,
                "out_degree": doc.out_degree,
                "in_degree": doc.in_degree,
                "pagerank": doc.pagerank,
                "cited_docs": doc.cited_docs,
                "citing_docs": doc.citing_docs,
            }
            for doc_id, doc in self.documents.items()
        }

        with open(path / "documents.json", "w") as f:
            json.dump(docs_data, f, indent=2)

        # Save PageRank scores
        with open(path / "pagerank.json", "w") as f:
            json.dump(self.pagerank_scores, f, indent=2)

    def load(self, path: Path):
        """
        Load citation graph from disk.

        Args:
            path: Load path
        """
        path = Path(path)

        # Load graph (using pickle directly - nx.read_gpickle removed in NetworkX 3.0+)
        import pickle
        with open(path / "graph.gpickle", "rb") as f:
            self.graph = pickle.load(f)

        # Load documents
        with open(path / "documents.json") as f:
            docs_data = json.load(f)

        self.documents = {
            doc_id: DocumentNode(
                doc_id=doc_id,
                title=data["title"],
                citation=data.get("citation"),
                court=data.get("court"),
                year=data.get("year"),
                doc_type=data.get("doc_type", "case"),
                out_degree=data.get("out_degree", 0),
                in_degree=data.get("in_degree", 0),
                pagerank=data.get("pagerank", 0.0),
                cited_docs=data.get("cited_docs", []),
                citing_docs=data.get("citing_docs", []),
            )
            for doc_id, data in docs_data.items()
        }

        # Load PageRank scores
        with open(path / "pagerank.json") as f:
            self.pagerank_scores = json.load(f)

        # Compute authority tiers
        self._compute_authority_tiers()


# Example usage
if __name__ == "__main__":
    # Create citation graph
    graph = CitationGraph()

    # Example documents
    doc1_text = """
    In Marbury v. Madison, 5 U.S. 137 (1803), the Court held that
    it has the power to review acts of Congress. This principle was
    later affirmed in Cooper v. Aaron, 358 U.S. 1 (1958).
    """

    doc2_text = """
    In Cooper v. Aaron, 358 U.S. 1 (1958), the Court reaffirmed
    the holding of Marbury v. Madison, 5 U.S. 137 (1803), regarding
    judicial review.
    """

    doc3_text = """
    As established in Marbury v. Madison, 5 U.S. 137 (1803), the
    judiciary has the final say on constitutional interpretation.
    """

    # Add documents
    graph.add_document(
        doc_id="marbury-v-madison",
        title="Marbury v. Madison",
        text=doc1_text,
        metadata={
            "citation": "5-U.S.-137",
            "court": "Supreme Court",
            "year": 1803,
        },
    )

    graph.add_document(
        doc_id="cooper-v-aaron",
        title="Cooper v. Aaron",
        text=doc2_text,
        metadata={
            "citation": "358-U.S.-1",
            "court": "Supreme Court",
            "year": 1958,
        },
    )

    graph.add_document(
        doc_id="doc-3",
        title="Example Case",
        text=doc3_text,
        metadata={
            "court": "District Court",
            "year": 2020,
        },
    )

    # Compute PageRank
    pagerank = graph.compute_pagerank()

    # Print results
    print("PageRank Scores:")
    for doc_id, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True):
        doc = graph.documents[doc_id]
        tier = graph.get_authority_tier(doc_id)
        print(f"  {doc.title}: {score:.4f} ({tier})")
        print(f"    In-degree: {doc.in_degree}, Out-degree: {doc.out_degree}")
