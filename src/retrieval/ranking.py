"""
Authority-Weighted Hybrid Ranking.

Implements the hybrid ranking strategy described in Section 3.2.2 of the
legal RAG framework. Combines vector similarity with citation-based authority
to produce final ranking scores.

The Problem:
Pure vector similarity treats all documents equally. A district court case
with perfect semantic match gets ranked above a Supreme Court landmark case
with slightly lower similarity. This is wrong for legal research.

Example Query: "What is qualified immunity?"
Rankings without authority:
1. District Court Case (vector_sim=0.95, authority=0.01) → score=0.95
2. Harlow v. Fitzgerald (vector_sim=0.90, authority=0.99) → score=0.90
❌ Wrong! Harlow should rank higher - it's THE landmark case on qualified immunity.

Rankings WITH authority (0.7 × vector + 0.3 × authority):
1. Harlow v. Fitzgerald (0.7×0.90 + 0.3×0.99) = 0.927 ✅
2. District Court Case (0.7×0.95 + 0.3×0.01) = 0.668

Hybrid Ranking Formula:
```
final_score = (α × vector_similarity) + (β × authority_score)
```

Default weights: α=0.7, β=0.3
- Semantic match is PRIMARY (70%)
- Authority is SECONDARY (30%) but significant

References:
- Section 3.2.2: Authority-Weighted Ranking
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np

from src.retrieval.citation_graph import CitationGraph


@dataclass
class RankedDocument:
    """A document with ranking scores."""

    doc_id: str                          # Document identifier
    content: str                         # Document text
    metadata: Dict                       # Document metadata

    # Ranking scores
    vector_score: float                  # Vector similarity (0-1)
    authority_score: float               # Authority score (0-1)
    final_score: float                   # Hybrid score (0-1)

    # Breakdown (for debugging)
    rank: int = 0                        # Final rank position
    authority_tier: str = "Standard"     # Authority tier


class HybridRanker:
    """
    Hybrid ranker combining vector similarity and citation authority.

    This is a re-ranker - it takes initial vector search results and
    re-ranks them using authority scores.
    """

    def __init__(
        self,
        citation_graph: CitationGraph,
        vector_weight: float = 0.7,
        authority_weight: float = 0.3,
    ):
        """
        Initialize hybrid ranker.

        Args:
            citation_graph: Citation graph with authority scores
            vector_weight: Weight for vector similarity (default: 0.7)
            authority_weight: Weight for authority score (default: 0.3)
        """
        self.citation_graph = citation_graph
        self.vector_weight = vector_weight
        self.authority_weight = authority_weight

        # Validate weights
        assert abs(vector_weight + authority_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"

    def rank(
        self,
        query: str,
        results: List[Tuple[str, float, str, Dict]],
        top_k: Optional[int] = None,
    ) -> List[RankedDocument]:
        """
        Re-rank search results using hybrid scoring.

        Args:
            query: Search query
            results: List of (doc_id, vector_score, content, metadata) tuples
            top_k: Number of results to return (None = all)

        Returns:
            List of ranked documents, sorted by final_score (descending)
        """
        ranked_docs = []

        for doc_id, vector_score, content, metadata in results:
            # Get authority score from citation graph
            authority_score = self.citation_graph.get_authority_score(doc_id)

            # Compute hybrid score
            final_score = (
                self.vector_weight * vector_score +
                self.authority_weight * authority_score
            )

            # Get authority tier (for debugging)
            authority_tier = self.citation_graph.get_authority_tier(doc_id)

            ranked_doc = RankedDocument(
                doc_id=doc_id,
                content=content,
                metadata=metadata,
                vector_score=vector_score,
                authority_score=authority_score,
                final_score=final_score,
                authority_tier=authority_tier,
            )

            ranked_docs.append(ranked_doc)

        # Sort by final score (descending)
        ranked_docs.sort(key=lambda x: x.final_score, reverse=True)

        # Assign ranks
        for i, doc in enumerate(ranked_docs):
            doc.rank = i + 1

        # Return top-k
        if top_k is not None:
            ranked_docs = ranked_docs[:top_k]

        return ranked_docs

    def explain_ranking(self, ranked_doc: RankedDocument) -> str:
        """
        Generate human-readable explanation of ranking.

        Args:
            ranked_doc: Ranked document

        Returns:
            Explanation string
        """
        explanation = f"""
Rank #{ranked_doc.rank}: {ranked_doc.metadata.get('title', 'Unknown')}

Vector Similarity:  {ranked_doc.vector_score:.3f} (weight: {self.vector_weight})
Authority Score:    {ranked_doc.authority_score:.3f} (weight: {self.authority_weight})
-----------------------------------------
Final Score:        {ranked_doc.final_score:.3f}

Authority Tier: {ranked_doc.authority_tier}

Breakdown:
  Vector component:    {self.vector_weight} x {ranked_doc.vector_score:.3f} = {self.vector_weight * ranked_doc.vector_score:.3f}
  Authority component: {self.authority_weight} x {ranked_doc.authority_score:.3f} = {self.authority_weight * ranked_doc.authority_score:.3f}
        """.strip()

        return explanation

    def adjust_weights(
        self,
        vector_weight: float,
        authority_weight: float,
    ):
        """
        Adjust ranking weights.

        Args:
            vector_weight: New vector weight
            authority_weight: New authority weight
        """
        assert abs(vector_weight + authority_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"

        self.vector_weight = vector_weight
        self.authority_weight = authority_weight

    def get_authority_boost_stats(
        self,
        results: List[RankedDocument],
    ) -> Dict[str, any]:
        """
        Compute statistics on authority boost.

        Shows how much authority scoring changed the rankings.

        Args:
            results: Ranked documents

        Returns:
            Dictionary with statistics
        """
        # Sort by vector score only
        vector_only_ranks = sorted(
            enumerate(results),
            key=lambda x: x[1].vector_score,
            reverse=True,
        )

        # Compute rank changes
        rank_changes = []
        for vector_rank, (hybrid_rank, doc) in enumerate(vector_only_ranks):
            rank_change = vector_rank - doc.rank + 1
            rank_changes.append(rank_change)

        # Statistics
        avg_rank_change = np.mean(np.abs(rank_changes))
        max_rank_change = np.max(np.abs(rank_changes))

        # Documents with significant boost (moved up >2 positions)
        boosted_docs = [
            (doc.doc_id, doc.metadata.get("title", "Unknown"), change)
            for (_, doc), change in zip(vector_only_ranks, rank_changes)
            if change > 2
        ]

        return {
            "avg_rank_change": avg_rank_change,
            "max_rank_change": max_rank_change,
            "boosted_docs": boosted_docs,
            "total_docs": len(results),
        }


class AuthorityBooster:
    """
    Authority-based score boosting with configurable strategies.

    Different boost strategies:
    1. Linear: Simple weighted average
    2. Exponential: Amplify authority differences
    3. Threshold: Only boost high-authority documents
    """

    @staticmethod
    def linear_boost(
        vector_score: float,
        authority_score: float,
        vector_weight: float = 0.7,
        authority_weight: float = 0.3,
    ) -> float:
        """
        Linear boost (default strategy).

        Score = α × vector + β × authority

        Args:
            vector_score: Vector similarity (0-1)
            authority_score: Authority score (0-1)
            vector_weight: Vector weight
            authority_weight: Authority weight

        Returns:
            Boosted score
        """
        return vector_weight * vector_score + authority_weight * authority_score

    @staticmethod
    def exponential_boost(
        vector_score: float,
        authority_score: float,
        vector_weight: float = 0.7,
        authority_weight: float = 0.3,
        exponent: float = 2.0,
    ) -> float:
        """
        Exponential boost (amplifies authority differences).

        Score = α × vector + β × authority^e

        This makes high-authority documents MUCH more prominent.

        Args:
            vector_score: Vector similarity (0-1)
            authority_score: Authority score (0-1)
            vector_weight: Vector weight
            authority_weight: Authority weight
            exponent: Exponent for authority (default: 2.0)

        Returns:
            Boosted score
        """
        boosted_authority = authority_score ** exponent
        return vector_weight * vector_score + authority_weight * boosted_authority

    @staticmethod
    def threshold_boost(
        vector_score: float,
        authority_score: float,
        vector_weight: float = 0.7,
        authority_weight: float = 0.3,
        threshold: float = 0.5,
    ) -> float:
        """
        Threshold boost (only boost high-authority documents).

        If authority >= threshold:
            Score = α × vector + β × authority
        Else:
            Score = vector

        This prevents low-authority documents from getting boosted.

        Args:
            vector_score: Vector similarity (0-1)
            authority_score: Authority score (0-1)
            vector_weight: Vector weight
            authority_weight: Authority weight
            threshold: Authority threshold (default: 0.5)

        Returns:
            Boosted score
        """
        if authority_score >= threshold:
            return vector_weight * vector_score + authority_weight * authority_score
        else:
            return vector_score


# Example usage
if __name__ == "__main__":
    from src.retrieval.citation_graph import CitationGraph

    # Create citation graph
    graph = CitationGraph()

    # Add example documents
    graph.add_document(
        doc_id="harlow-v-fitzgerald",
        title="Harlow v. Fitzgerald",
        text="Landmark case establishing qualified immunity standard...",
        metadata={
            "citation": "457-U.S.-800",
            "court": "Supreme Court",
            "year": 1982,
        },
    )

    graph.add_document(
        doc_id="district-case",
        title="Smith v. Jones",
        text="Qualified immunity discussion in a district court case...",
        metadata={
            "court": "District Court",
            "year": 2020,
        },
    )

    # Compute PageRank
    graph.compute_pagerank()

    # Simulate vector search results
    results = [
        # (doc_id, vector_score, content, metadata)
        (
            "district-case",
            0.95,
            "Qualified immunity discussion...",
            {"title": "Smith v. Jones", "court": "District Court"},
        ),
        (
            "harlow-v-fitzgerald",
            0.90,
            "Landmark case establishing qualified immunity...",
            {"title": "Harlow v. Fitzgerald", "court": "Supreme Court"},
        ),
    ]

    # Create hybrid ranker
    ranker = HybridRanker(graph)

    # Re-rank results
    ranked = ranker.rank("What is qualified immunity?", results)

    # Print results
    print("Hybrid Ranking Results:")
    print("=" * 60)
    for doc in ranked:
        print(f"\n{ranker.explain_ranking(doc)}")
        print()

    # Show authority boost statistics
    stats = ranker.get_authority_boost_stats(ranked)
    print("\nAuthority Boost Statistics:")
    print(f"  Average rank change: {stats['avg_rank_change']:.2f}")
    print(f"  Maximum rank change: {stats['max_rank_change']}")
    print(f"  Boosted documents: {len(stats['boosted_docs'])}")
