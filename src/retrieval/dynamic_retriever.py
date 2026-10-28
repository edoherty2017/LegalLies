"""
Dynamic Evidence Retriever with CourtListener API Fallback.

Combines static corpus with real-time API queries for comprehensive evidence retrieval.

Strategy:
1. Query static corpus (fast, reliable)
2. If insufficient evidence (low similarity), query CourtListener API
3. Combine and re-rank results

Use case: Detecting hallucinations about recent cases or specialized topics
not in the static corpus.
"""

import os
from typing import List, Tuple, Optional
from pathlib import Path

import requests


class DynamicEvidenceRetriever:
    """
    Evidence retriever with CourtListener API fallback.
    
    Uses static corpus first, then queries API if needed.
    """
    
    def __init__(
        self,
        static_retriever,  # DocumentRetriever instance
        api_key: Optional[str] = None,
        fallback_threshold: float = 0.6,  # Min similarity to skip API
        fallback_top_k: int = 3,
    ):
        """
        Initialize dynamic retriever.
        
        Args:
            static_retriever: Existing static corpus retriever
            api_key: CourtListener API key (or load from env)
            fallback_threshold: If max static similarity < this, query API
            fallback_top_k: Number of API results to fetch
        """
        self.static_retriever = static_retriever
        self.api_key = api_key or self._load_api_key()
        self.fallback_threshold = fallback_threshold
        self.fallback_top_k = fallback_top_k
        
        self.base_url = "https://www.courtlistener.com/api/rest/v4"
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Token {self.api_key}",
                "User-Agent": "LegalLies-HallucinationDetector/1.0",
            })
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment."""
        # Try environment variable
        api_key = os.getenv("COURTLISTENER_API_KEY")
        if api_key:
            return api_key
        
        # Try .env file
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("COURTLISTENER_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"\'')
        
        return None
    
    def retrieve_evidence(
        self,
        query: str,
        top_k: int = 5,
        force_api: bool = False,
    ) -> Tuple[List[str], dict]:
        """
        Retrieve evidence from static corpus + API fallback.
        
        Args:
            query: Search query
            top_k: Total results to return
            force_api: Force API query even if static corpus sufficient
        
        Returns:
            (evidence_list, metadata)
            metadata includes: {
                'static_count': int,
                'api_count': int,
                'fallback_triggered': bool,
                'max_static_similarity': float
            }
        """
        metadata = {
            'static_count': 0,
            'api_count': 0,
            'fallback_triggered': False,
            'max_static_similarity': 0.0,
        }
        
        # Step 1: Query static corpus
        static_results = self.static_retriever.retrieve_evidence(
            query=query,
            top_k=top_k,
        )
        
        # Get similarity scores (if available)
        static_scores = getattr(self.static_retriever, 'last_scores', [])
        max_similarity = max(static_scores) if static_scores else 0.0
        metadata['max_static_similarity'] = max_similarity
        metadata['static_count'] = len(static_results)
        
        # Step 2: Check if API fallback needed
        should_query_api = (
            force_api or
            (max_similarity < self.fallback_threshold and self.api_key is not None)
        )
        
        if should_query_api:
            metadata['fallback_triggered'] = True
            
            # Query CourtListener API
            api_results = self._query_courtlistener_api(query, self.fallback_top_k)
            metadata['api_count'] = len(api_results)
            
            # Combine results (static first, then API)
            combined_results = static_results + api_results
            
            # Return top-k from combined
            return combined_results[:top_k], metadata
        else:
            # Static corpus sufficient
            return static_results, metadata
    
    def _query_courtlistener_api(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[str]:
        """
        Query CourtListener API for relevant cases.
        
        Args:
            query: Search query
            top_k: Number of results to fetch
        
        Returns:
            List of opinion texts
        """
        if not self.api_key:
            print("[WARNING] No CourtListener API key - skipping API fallback")
            return []
        
        try:
            # Search API endpoint
            response = self.session.get(
                f"{self.base_url}/search/",
                params={
                    "q": query,
                    "type": "o",  # Opinions
                    "order_by": "score desc",
                    "stat_Precedential": "on",  # Only precedential
                },
                timeout=5,
            )
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])[:top_k]
            
            # Extract opinion texts
            evidence = []
            for result in results:
                # Get snippet or plain_text
                text = result.get("snippet", result.get("plain_text", ""))
                if text and len(text) > 100:
                    # Limit to 1000 chars
                    evidence.append(text[:1000])
            
            print(f"[API] Retrieved {len(evidence)} opinions from CourtListener")
            return evidence
            
        except Exception as e:
            print(f"[WARNING] CourtListener API query failed: {e}")
            return []


# Example usage
if __name__ == "__main__":
    from src.retrieval.document_retriever import DocumentRetriever
    
    # Initialize static retriever
    static_retriever = DocumentRetriever(
        index_path="data/indexes/legal_corpus",
        embedding_model="sentence-transformers/all-mpnet-base-v2",
    )
    
    # Initialize dynamic retriever with API fallback
    dynamic_retriever = DynamicEvidenceRetriever(
        static_retriever=static_retriever,
        fallback_threshold=0.6,  # Query API if max similarity < 0.6
    )
    
    # Test query
    query = "What is the standard for qualified immunity in excessive force cases?"
    
    evidence, metadata = dynamic_retriever.retrieve_evidence(query, top_k=5)
    
    print(f"\nQuery: {query}")
    print(f"\nMetadata:")
    print(f"  Static corpus results: {metadata['static_count']}")
    print(f"  API results: {metadata['api_count']}")
    print(f"  Fallback triggered: {metadata['fallback_triggered']}")
    print(f"  Max static similarity: {metadata['max_static_similarity']:.2f}")
    
    print(f"\nEvidence ({len(evidence)} passages):")
    for i, passage in enumerate(evidence, 1):
        print(f"\n[{i}] {passage[:200]}...")
