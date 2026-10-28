#!/usr/bin/env python3
"""
Build retrieval index for the oracle corpus.

Creates BM25 and dense embeddings index for the legal oracle corpus
to enable hallucination detection via retrieval-based verification.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.document_retriever import DocumentRetriever


def load_oracle_corpus(corpus_path: Path) -> tuple[list[str], list[dict]]:
    """Load documents and metadata from oracle corpus JSONL file."""
    documents = []
    metadata = []

    print(f"Loading corpus from {corpus_path}...")

    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print(f"  Loaded {i} documents...")

            doc = json.loads(line)

            # Extract text content
            text = doc.get("text", "")
            if not text or len(text) < 500:
                continue  # Skip empty or too-short documents

            documents.append(text)

            # Store metadata for each document
            metadata.append({
                "doc_id": doc.get("id", str(i)),
                "title": doc.get("title", "Unknown"),
                "court": doc.get("court", "unknown"),
                "date": doc.get("date", ""),
                "year": doc.get("year", 0),
                "jurisdiction": doc.get("jurisdiction", "federal"),
                "source_url": doc.get("source_url", ""),
                "precedential_status": doc.get("precedential_status", ""),
                "citation_count": doc.get("citation_count", 0),
            })

    print(f"Loaded {len(documents)} documents from corpus")
    return documents, metadata


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    corpus_path = project_root / "data" / "oracle_corpus" / "legal_oracle.jsonl"
    index_path = project_root / "data" / "indexes" / "oracle_corpus"

    if not corpus_path.exists():
        print(f"Error: Corpus not found at {corpus_path}")
        sys.exit(1)

    # Load corpus
    documents, metadata = load_oracle_corpus(corpus_path)

    if len(documents) == 0:
        print("Error: No documents loaded from corpus")
        sys.exit(1)

    # Initialize retriever (will use GPU if available)
    print("\nInitializing document retriever...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    retriever = DocumentRetriever(
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        device=device,
        use_authority_ranking=False,  # We'll build citation graph separately if needed
    )

    # Build index
    print(f"\nBuilding index for {len(documents)} documents...")
    print("This will take a while for dense embeddings (~12k docs)...")

    retriever.build_index(
        documents=documents,
        metadata=metadata,
        save_path=str(index_path),
    )

    # Verify index
    print("\nVerifying index...")
    stats = retriever.get_corpus_stats()
    print(f"  Documents indexed: {stats['num_documents']}")
    print(f"  BM25 index: {'✓' if stats['has_bm25_index'] else '✗'}")
    print(f"  Dense embeddings: {'✓' if stats['has_dense_embeddings'] else '✗'}")

    # Test retrieval
    print("\nTesting retrieval with sample query...")
    test_query = "What is qualified immunity for police officers?"
    results = retriever.retrieve_evidence(test_query, top_k=3)

    print(f"Query: {test_query}")
    print(f"Retrieved {len(results)} documents:")
    for i, doc in enumerate(results):
        preview = doc[:200].replace("\n", " ")
        print(f"  {i+1}. {preview}...")

    print(f"\n✓ Index built successfully at {index_path}")
    print(f"  - corpus.json: Document texts and metadata")
    print(f"  - bm25.pkl: BM25 sparse index")
    print(f"  - dense_embeddings.npy: Dense vector embeddings")


if __name__ == "__main__":
    main()
