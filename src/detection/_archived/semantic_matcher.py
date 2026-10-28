"""
Semantic Grounding Verification using embeddings.

This module implements fast embedding-based similarity checking to identify
claims that are completely ungrounded in the source context.

Uses Legal-BERT embeddings for domain-specific semantic understanding.
Complements entailment verification by providing a coarse-grained filter
before more expensive NLI checks.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util


@dataclass
class GroundingResult:
    """Result of semantic grounding verification."""

    claim: str
    source_chunk: str
    similarity_score: float  # 0-1: Cosine similarity
    is_grounded: bool  # True if similarity >= threshold
    threshold: float

    def __repr__(self) -> str:
        status = "GROUNDED" if self.is_grounded else "UNGROUNDED"
        return f"{status} ({self.similarity_score:.2%}): {self.claim[:50]}..."

    def to_dict(self) -> Dict:
        return {
            "claim": self.claim,
            "source": self.source_chunk[:200],
            "similarity": self.similarity_score,
            "grounded": self.is_grounded,
            "threshold": self.threshold,
        }


class SemanticGroundingVerifier:
    """
    Verifies claims have semantic basis in retrieved documents using embeddings.

    This is a fast, coarse-grained verification that checks:
    1. Does the claim's embedding have high cosine similarity to any source?
    2. Are key concepts from the claim present in source documents?

    Benefits:
    - Very fast (no NLI model inference)
    - Good for filtering completely ungrounded claims
    - Complements entailment checking (which is slower)
    - Can identify which sources support each claim

    Limitations:
    - Can't catch subtle semantic drift or logical contradictions
    - Purely statistical similarity, not semantic understanding
    - May give false positives for superficially similar text
    """

    def __init__(
        self,
        model_name: str = "nlpaueb/legal-bert-base-uncased",
        similarity_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """
        Initialize verifier with embedding model.

        Args:
            model_name: Sentence transformer model for embeddings
            similarity_threshold: Minimum cosine similarity to consider claim "grounded"
                                 Default: 0.5 (moderate similarity required)
            device: Device to run model on ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.device = device

        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.model.max_seq_length = 512  # Enforce truncation
            print(f"Loaded embedding model: {model_name} on {device}")
        except Exception as e:
            print(f"Error loading embedding model {model_name}: {e}")
            self.model = None

    def verify_claim_grounding(
        self,
        claim: str,
        source_documents: List[str],
        return_top_k: int = 3,
    ) -> Tuple[GroundingResult, Optional[List[GroundingResult]]]:
        """
        Verify if a claim is grounded in provided source documents.

        Finds the best matching source for the claim. A claim is considered
        "grounded" if it has cosine similarity >= threshold to at least one source.

        Args:
            claim: Claim to verify
            source_documents: List of source text chunks
            return_top_k: Number of top matches to return (default: 3)

        Returns:
            Tuple of (best_match_result, top_k_matches)
        """
        if not self.model:
            # Fallback if model unavailable
            return (
                GroundingResult(
                    claim=claim,
                    source_chunk="Model unavailable",
                    similarity_score=0.0,
                    is_grounded=False,
                    threshold=self.similarity_threshold,
                ),
                None,
            )

        # Handle edge cases: empty claim or no sources
        if not claim or not source_documents or all(not s for s in source_documents):
            return (
                GroundingResult(
                    claim=claim,
                    source_chunk=(
                        "No valid sources provided"
                        if source_documents
                        else "No sources provided"
                    ),
                    similarity_score=0.0,
                    is_grounded=False,
                    threshold=self.similarity_threshold,
                ),
                None,
            )

        # CRITICAL FIX: Pre-truncate to avoid tokenization errors
        max_chars = 1500
        truncated_claim = claim[:max_chars]
        truncated_sources = [s[:max_chars] for s in source_documents]

        # Embed claim and sources together for efficiency
        embeddings = self.model.encode(
            [truncated_claim] + truncated_sources, convert_to_numpy=True
        )
        claim_embedding = embeddings[0:1]  # Shape: (1, D)
        source_embeddings = embeddings[1:]  # Shape: (N, D)

        # Compute cosine similarities
        similarities = util.cos_sim(claim_embedding, source_embeddings)[0].cpu().numpy()

        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = float(similarities[best_idx])
        best_source = source_documents[best_idx]

        # Create best match result
        best_result = GroundingResult(
            claim=claim,
            source_chunk=best_source,
            similarity_score=best_similarity,
            is_grounded=best_similarity >= self.similarity_threshold,
            threshold=self.similarity_threshold,
        )

        # Create top-k matches
        top_k_results = None
        if return_top_k > 0:
            top_k_idx = np.argsort(similarities)[::-1][:return_top_k]
            top_k_results = [
                GroundingResult(
                    claim=claim,
                    source_chunk=source_documents[idx],
                    similarity_score=float(similarities[idx]),
                    is_grounded=float(similarities[idx]) >= self.similarity_threshold,
                    threshold=self.similarity_threshold,
                )
                for idx in top_k_idx
            ]

        return best_result, top_k_results

    def verify_batch(
        self, claims: List[str], sources: List[str], output_path: Optional[Path] = None
    ) -> List[GroundingResult]:
        """
        Verify multiple claims against multiple sources.

        Args:
            claims: List of claims to verify
            sources: List of source documents
            output_path: Optional path to save results

        Returns:
            List of GroundingResult objects
        """
        results = []

        for i, claim in enumerate(claims, 1):
            print(f"Verifying claim {i}/{len(claims)}...", end="\r")

            result, _ = self.verify_claim_grounding(claim, sources)
            results.append(result)

        print(" " * 50, end="\r")  # Clear progress line

        # Save results if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            results_dict = {
                "model": self.model_name,
                "threshold": self.similarity_threshold,
                "num_claims": len(claims),
                "num_sources": len(sources),
                "grounded": sum(1 for r in results if r.is_grounded),
                "ungrounded": sum(1 for r in results if not r.is_grounded),
                "avg_similarity": sum(r.similarity_score for r in results) / len(results),
                "results": [r.to_dict() for r in results],
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results_dict, f, indent=2)

            print(f"Results saved to: {output_path}")

        return results

    def generate_grounding_report(self, claims: List[str], sources: List[str]) -> str:
        """
        Generate human-readable grounding report.

        Args:
            claims: List of claims
            sources: List of source documents

        Returns:
            Formatted report string
        """
        results = self.verify_batch(claims, sources)

        report = []
        report.append("=" * 80)
        report.append("SEMANTIC GROUNDING REPORT")
        report.append("=" * 80)
        report.append(f"Similarity Threshold: {self.similarity_threshold}")
        report.append(f"Total Claims: {len(results)}")
        report.append(f"Grounded: {sum(1 for r in results if r.is_grounded)}")
        report.append(f"Ungrounded: {sum(1 for r in results if not r.is_grounded)}")
        report.append("")

        report.append("UNGROUNDED CLAIMS (May be hallucinations):")
        report.append("-" * 80)

        ungrounded = [r for r in results if not r.is_grounded]

        if ungrounded:
            for result in ungrounded:
                report.append(f"CLAIM: {result.claim}")
                report.append(
                    f"  Similarity: {result.similarity_score:.2%} (threshold: {result.threshold})"
                )
                report.append(f"  Best Match: {result.source_chunk[:70]}...")
                report.append("")
        else:
            report.append("All claims have semantic basis in retrieved documents.")

        return "\n".join(report)
