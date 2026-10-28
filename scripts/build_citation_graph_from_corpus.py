"""
Build Citation Graph from Downloaded Legal Corpus.

Integrates Phase 3 (Citation Graph) with Phase 4 (Pile of Law).

This script:
1. Loads opinions from CourtListener corpus
2. Extracts citations from each opinion
3. Builds directed citation graph
4. Computes PageRank authority scores
5. Saves graph for use in hybrid retrieval

Architecture:
  CourtListener Corpus → Citation Graph → PageRank → Authority Scores
                              ↓
                         Hybrid Retrieval (0.7 vector + 0.3 authority)
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.citation_graph import CitationGraph


def load_courtlistener_corpus(corpus_path: Path) -> List[Dict]:
    """Load opinions from CourtListener JSON."""
    if not corpus_path.exists():
        print(f"[ERROR] Corpus not found: {corpus_path}")
        print("[INFO] Run: python scripts/courtlistener_fetcher.py")
        return []

    with open(corpus_path, "r", encoding="utf-8") as f:
        opinions = json.load(f)

    print(f"[OK] Loaded {len(opinions)} opinions from {corpus_path.name}")
    return opinions


def main():
    """Build citation graph from legal corpus."""
    print("=" * 60)
    print("Building Citation Graph from Legal Corpus")
    print("=" * 60)

    # Load corpus
    corpus_path = Path("data/corpus/courtlistener_opinions.json")
    opinions = load_courtlistener_corpus(corpus_path)

    if not opinions:
        print("[ERROR] No opinions to process")
        return

    # Initialize citation graph
    print("\nInitializing citation graph...")
    graph = CitationGraph()

    # Process each opinion
    print("\nAdding opinions to citation graph...")
    for i, opinion in enumerate(opinions, 1):
        doc_id = f"opinion-{i}"
        title = opinion.get("case_name", f"Case {i}")
        text = opinion.get("text", "")

        # Extract metadata
        metadata = {
            "citation": opinion.get("citations", [""])[0] if opinion.get("citations") else None,
            "court": opinion.get("court_id", "Unknown"),
            "year": int(opinion.get("date_filed", "0")[:4]) if opinion.get("date_filed") else None,
            "doc_type": "opinion",
        }

        # Add to graph (automatically extracts citations)
        graph.add_document(doc_id, title, text, metadata)

        if i % 50 == 0:
            print(f"  Processed {i}/{len(opinions)} opinions...")

    print(f"[OK] Added {len(opinions)} opinions to citation graph")

    # Graph statistics
    print("\n" + "=" * 60)
    print("Citation Graph Statistics")
    print("=" * 60)
    print(f"Total documents: {len(graph.documents)}")
    print(f"Total citations (edges): {graph.graph.number_of_edges()}")

    # Compute PageRank
    print("\nComputing PageRank authority scores...")
    pagerank_scores = graph.compute_pagerank()
    print(f"[OK] Computed PageRank for {len(pagerank_scores)} documents")

    # Top authorities
    print("\n" + "=" * 60)
    print("Top 10 Most Authoritative Cases")
    print("=" * 60)
    top_authorities = graph.get_top_authorities(n=10)
    for i, (doc_id, score) in enumerate(top_authorities, 1):
        doc = graph.documents[doc_id]
        tier = graph.get_authority_tier(doc_id)
        print(f"{i:2d}. {doc.title[:50]:50s} (score: {score:.4f}, {tier})")
        print(f"    In-degree: {doc.in_degree}, Out-degree: {doc.out_degree}")

    # Save graph
    print("\n" + "=" * 60)
    print("Saving Citation Graph")
    print("=" * 60)
    output_path = Path("data/indexes/legal_corpus/citation_graph")
    graph.save(output_path)
    print(f"[OK] Citation graph saved to: {output_path}")

    # Save authority scores separately for easy access
    authority_scores = {
        doc_id: {
            "title": graph.documents[doc_id].title,
            "authority_score": graph.get_authority_score(doc_id),
            "authority_tier": graph.get_authority_tier(doc_id),
            "pagerank": pagerank_scores.get(doc_id, 0.0),
            "in_degree": graph.documents[doc_id].in_degree,
            "out_degree": graph.documents[doc_id].out_degree,
        }
        for doc_id in graph.documents
    }

    authority_path = Path("data/indexes/legal_corpus/authority_scores.json")
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    with open(authority_path, "w") as f:
        json.dump(authority_scores, f, indent=2)
    print(f"[OK] Authority scores saved to: {authority_path}")

    print("\n" + "=" * 60)
    print("Citation Graph Build Complete")
    print("=" * 60)
    print("[INFO] Next step: Run build_retrieval_index.py to create hybrid retrieval system")


if __name__ == "__main__":
    main()
