"""
Document Retrieval Module with hybrid BM25 + dense embeddings.

Provides automatic retrieval for any LLM output, whether from RAG or not.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.retrieval.citation_graph import CitationGraph
from src.retrieval.ranking import HybridRanker


class DocumentRetriever:
    """
    Hybrid retrieval system combining BM25 (sparse) and dense embeddings.

    Designed to retrieve supporting evidence for hallucination detection,
    applicable to both Baseline (no RAG) and RAG+Web variants.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu",
        use_authority_ranking: bool = True,
    ):
        """
        Initialize document retriever.

        Args:
            index_path: Path to pre-built index directory (contains corpus, BM25, embeddings)
            embedding_model: HuggingFace model for dense retrieval
            device: Device for embedding model ('cpu' or 'cuda')
            use_authority_ranking: Enable authority-weighted ranking (requires citation graph)
        """
        self.index_path = Path(index_path) if index_path else None
        self.device = device
        self.use_authority_ranking = use_authority_ranking

        # Initialize embedding model for dense retrieval
        self.embedding_model = SentenceTransformer(embedding_model, device=device)

        # Document corpus and indices
        self.corpus: List[str] = []
        self.corpus_metadata: List[Dict] = []  # Store doc_id, source, chunk_id, etc.
        self.bm25: Optional[BM25Okapi] = None
        self.dense_embeddings: Optional[np.ndarray] = None

        # Store last retrieval scores for dynamic retrieval
        self.last_scores: List[float] = []

        # Authority ranking components
        self.citation_graph: Optional[CitationGraph] = None
        self.hybrid_ranker: Optional[HybridRanker] = None

        # Load index if provided. A configured-but-missing path used to be
        # skipped silently: retrieve_evidence then returned [] for every query
        # while the detector still printed "[OK] Retrieval loaded", so a run
        # against a nonexistent index looked successful and scored every claim
        # UNSUPPORTED.
        if self.index_path:
            if not self.index_path.exists():
                raise FileNotFoundError(
                    f"Retrieval index not found at {self.index_path}. Retrieval "
                    f"would return no evidence for every claim, scoring them all "
                    f"UNSUPPORTED and reporting a meaningless 0% hallucination "
                    f"rate. Build it with scripts/build_retrieval_index.py."
                )
            self._load_index()

    def _load_index(self):
        """Load pre-built index from disk."""
        print(f"Loading index from {self.index_path}...")

        # Load corpus
        corpus_path = self.index_path / "corpus.json"
        if corpus_path.exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.corpus = data.get("documents", [])
                self.corpus_metadata = data.get("metadata", [])

        # Load BM25 index
        bm25_path = self.index_path / "bm25.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)

        # Load dense embeddings
        embeddings_path = self.index_path / "dense_embeddings.npy"
        if embeddings_path.exists():
            self.dense_embeddings = np.load(embeddings_path)

        print(f"Loaded {len(self.corpus)} documents")

        # Load citation graph if available and authority ranking enabled
        if self.use_authority_ranking:
            citation_graph_path = self.index_path / "citation_graph"
            if citation_graph_path.exists():
                print("Loading citation graph for authority ranking...")
                self.citation_graph = CitationGraph()
                self.citation_graph.load(citation_graph_path)
                self.hybrid_ranker = HybridRanker(self.citation_graph)
                print(f"Citation graph loaded: {len(self.citation_graph.documents)} documents")
            else:
                print("Citation graph not found - using vector similarity only")

    def build_index(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        save_path: Optional[str] = None,
    ):
        """
        Build BM25 + dense index from document corpus.

        Args:
            documents: List of text passages to index
            metadata: Optional metadata for each document (doc_id, source, etc.)
            save_path: Path to save index for future use
        """
        print(f"Building index for {len(documents)} documents...")

        self.corpus = documents
        self.corpus_metadata = metadata or [{"doc_id": i} for i in range(len(documents))]

        # Build BM25 index
        tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Build dense embeddings
        print("Encoding documents with dense embeddings...")
        self.dense_embeddings = self.embedding_model.encode(
            documents,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Save index if path provided
        if save_path:
            self._save_index(save_path)

        print("Index built successfully")

    def _save_index(self, save_path: str):
        """Save index to disk."""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save corpus
        with open(save_dir / "corpus.json", "w", encoding="utf-8") as f:
            json.dump(
                {"documents": self.corpus, "metadata": self.corpus_metadata},
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Save BM25 index
        with open(save_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

        # Save dense embeddings
        np.save(save_dir / "dense_embeddings.npy", self.dense_embeddings)

        print(f"Index saved to {save_dir}")

    def retrieve_evidence(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> List[str]:
        """
        Retrieve top-k passages for a query (question + answer combined).

        Args:
            query: Search query (typically question + generated answer)
            top_k: Number of passages to return
            bm25_weight: Weight for BM25 scores (0-1)
            dense_weight: Weight for dense similarity scores (0-1)

        Returns:
            List of top-k retrieved passages
        """
        if not self.corpus:
            print("Warning: No corpus loaded, returning empty results")
            return []

        # Hybrid search
        results = self.hybrid_search(query, bm25_weight, dense_weight, top_k)

        # Store scores for dynamic retrieval fallback logic
        self.last_scores = [score for doc, score in results]

        # Extract passages only
        passages = [doc for doc, score in results]
        return passages

    def retrieve_for_claim(
        self,
        claim: str,
        top_k: int = 3,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> List[str]:
        """
        Retrieve top-k passages for a specific claim.

        Args:
            claim: Individual claim to verify
            top_k: Number of passages to return
            bm25_weight: Weight for BM25 scores
            dense_weight: Weight for dense similarity scores

        Returns:
            List of top-k retrieved passages
        """
        return self.retrieve_evidence(claim, top_k, bm25_weight, dense_weight)

    def hybrid_search(
        self,
        query: str,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Combine BM25 and dense retrieval with weighting.

        Optionally applies authority-weighted re-ranking if citation graph available.

        Args:
            query: Search query
            bm25_weight: Weight for BM25 scores (0-1)
            dense_weight: Weight for dense similarity scores (0-1)
            top_k: Number of results to return

        Returns:
            List of (document, score) tuples sorted by score
        """
        if not self.corpus:
            return []

        # BM25 retrieval (sparse)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Dense retrieval (semantic)
        query_embedding = self.embedding_model.encode(
            query, convert_to_numpy=True
        )
        dense_scores = np.dot(self.dense_embeddings, query_embedding)

        # Normalize scores to [0, 1]
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        dense_scores_norm = self._normalize_scores(dense_scores)

        # Hybrid scoring (BM25 + Dense)
        hybrid_scores = (
            bm25_weight * bm25_scores_norm + dense_weight * dense_scores_norm
        )

        # Get top candidates (fetch more for re-ranking if using authority)
        candidate_k = top_k * 2 if self.hybrid_ranker else top_k
        top_indices = np.argsort(hybrid_scores)[::-1][:candidate_k]

        # Apply authority re-ranking if available
        if self.hybrid_ranker:
            # Prepare data for hybrid ranker
            candidates = [
                (
                    str(idx),  # doc_id
                    float(hybrid_scores[idx]),  # vector_score
                    self.corpus[idx],  # content
                    self.corpus_metadata[idx] if idx < len(self.corpus_metadata) else {}  # metadata
                )
                for idx in top_indices
            ]

            # Re-rank using authority scores
            ranked_docs = self.hybrid_ranker.rank(query, candidates, top_k=top_k)

            # Convert back to (document, score) tuples
            results = [
                (doc.content, doc.final_score)
                for doc in ranked_docs
            ]
        else:
            # No authority ranking - use hybrid scores only
            results = [
                (self.corpus[idx], float(hybrid_scores[idx]))
                for idx in top_indices[:top_k]
            ]

        return results

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores to [0, 1]."""
        min_score = scores.min()
        max_score = scores.max()

        if max_score - min_score < 1e-9:
            return np.zeros_like(scores)

        return (scores - min_score) / (max_score - min_score)

    def get_corpus_stats(self) -> Dict:
        """Get statistics about the loaded corpus."""
        return {
            "num_documents": len(self.corpus),
            "has_bm25_index": self.bm25 is not None,
            "has_dense_embeddings": self.dense_embeddings is not None,
            "embedding_model": self.embedding_model.get_model_card()["name"]
            if hasattr(self.embedding_model, "get_model_card")
            else "unknown",
        }
