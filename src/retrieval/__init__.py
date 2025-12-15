"""
Retrieval System with Authority-Weighted Ranking.

Phase 3 Components:
- Citation Graph: Extract citations and build directed graph
- PageRank: Compute authority scores
- Hybrid Ranking: Combine vector similarity + authority

Usage:
    from src.retrieval import CitationGraph, HybridRanker

    # Build citation graph
    graph = CitationGraph()
    graph.add_document(doc_id, title, text, metadata)
    graph.compute_pagerank()

    # Re-rank search results
    ranker = HybridRanker(graph)
    ranked_results = ranker.rank(query, vector_results)
"""

from src.retrieval.citation_graph import (
    Citation,
    CitationExtractor,
    CitationGraph,
    DocumentNode,
)
from src.retrieval.ranking import (
    HybridRanker,
    RankedDocument,
    AuthorityBooster,
)

__all__ = [
    # Citation graph
    "Citation",
    "CitationExtractor",
    "CitationGraph",
    "DocumentNode",
    # Ranking
    "HybridRanker",
    "RankedDocument",
    "AuthorityBooster",
]
