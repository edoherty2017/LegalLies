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
        # Compile citation patterns
        self.patterns = self._compile_patterns()

        # Reporter abbreviations (for validation)
        self.reporters = self._load_reporters()

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """
        Compile regex patterns for different citation types.

        Returns:
            Dictionary of pattern_name -> compiled regex
        """
        patterns = {
            # Case citations: "500 U.S. 123" or "500 U.S. 123 (2000)"
            "case_standard": re.compile(
                r'\b(\d+)\s+([A-Z][A-Za-z.]*\.?)\s+(\d+)(?:\s+\((\d{4})\))?',
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
            "N.W.", "N.W.2d",                       # North Western
        }

        return reporters

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

        # Add citation edges
        for citation in citations:
            # Try to resolve citation to doc_id
            cited_doc_id = self._resolve_citation(citation)

            if cited_doc_id and cited_doc_id in self.documents:
                # Add edge: doc_id → cited_doc_id
                self.graph.add_edge(doc_id, cited_doc_id)

                # Update metadata
                node.cited_docs.append(cited_doc_id)
                self.documents[cited_doc_id].citing_docs.append(doc_id)

        # Update degrees
        node.out_degree = self.graph.out_degree(doc_id)
        node.in_degree = self.graph.in_degree(doc_id)

        # Invalidate cached PageRank
        self.pagerank_scores = {}

        return node

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

        # Search for matching document
        for doc_id, doc in self.documents.items():
            if doc.citation == citation.normalized_cite:
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

        score = self.pagerank_scores.get(doc_id, 0.0)

        # Normalize to 0-1 range
        if self.pagerank_scores:
            max_score = max(self.pagerank_scores.values())
            min_score = min(self.pagerank_scores.values())

            if max_score > min_score:
                score = (score - min_score) / (max_score - min_score)

        return score

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
