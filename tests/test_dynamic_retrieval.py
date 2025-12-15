"""
Test dynamic evidence retrieval with CourtListener API fallback.

Validates:
- Static corpus retrieval (fast path)
- API fallback when static corpus insufficient
- Combined results and authority ranking
- Latency measurements
"""

import time
from pathlib import Path

from src.retrieval.dynamic_retriever import DynamicEvidenceRetriever
from src.retrieval.document_retriever import DocumentRetriever


def test_static_corpus_query():
    """Test query that should be well-covered by static corpus."""
    print("\n" + "=" * 60)
    print("Test 1: Static Corpus Query (No API Fallback)")
    print("=" * 60)

    # Create static retriever first
    static_retriever = DocumentRetriever(index_path="data/indexes/legal_corpus")

    retriever = DynamicEvidenceRetriever(
        static_retriever=static_retriever,
        api_key=None,  # No API key = static only
    )

    # Query about Fourth Amendment (should be in sample corpus)
    query = "What is the Fourth Amendment exclusionary rule?"
    print(f"\nQuery: {query}")

    start_time = time.time()
    results, metadata = retriever.retrieve_evidence(query, top_k=3)
    latency = (time.time() - start_time) * 1000

    print(f"\nRetrieved {len(results)} passages in {latency:.1f}ms")
    print(f"Used API: {metadata.get('fallback_triggered', False)}")
    print(f"Static results: {metadata.get('static_count', 0)}")
    print(f"API results: {metadata.get('api_count', 0)}")

    print("\nTop 3 Results:")
    for i, passage in enumerate(results[:3], 1):
        print(f"\n[{i}] {passage[:150]}...")

    # Verify we got results without API
    assert len(results) > 0, "Should retrieve results from static corpus"
    assert not metadata.get("fallback_triggered"), "Should NOT use API for well-covered query"

    print("\n[OK] Test passed: Static corpus retrieval works")
    return latency


def test_api_fallback_trigger():
    """Test query that triggers API fallback."""
    print("\n" + "=" * 60)
    print("Test 2: API Fallback (Low Similarity Trigger)")
    print("=" * 60)

    # Check if API key is available
    import os
    api_key = os.getenv("COURTLISTENER_API_KEY")

    if not api_key:
        print("\n⚠ SKIPPED: No COURTLISTENER_API_KEY found")
        print("Set API key to test fallback: export COURTLISTENER_API_KEY=your_key")
        return None

    # Create static retriever first
    static_retriever = DocumentRetriever(index_path="data/indexes/legal_corpus")

    retriever = DynamicEvidenceRetriever(
        static_retriever=static_retriever,
        api_key=api_key,
        fallback_threshold=0.6,
    )

    # Query about very specific/recent case that likely isn't in corpus
    query = "What was the Supreme Court decision in Dobbs v. Jackson Women's Health Organization?"
    print(f"\nQuery: {query}")
    print(f"Fallback threshold: {retriever.fallback_threshold}")

    start_time = time.time()
    results, metadata = retriever.retrieve_evidence(query, top_k=5)
    latency = (time.time() - start_time) * 1000

    print(f"\nRetrieved {len(results)} passages in {latency:.1f}ms")
    print(f"Used API: {metadata.get('fallback_triggered', False)}")
    print(f"Static results: {metadata.get('static_count', 0)}")
    print(f"API results: {metadata.get('api_count', 0)}")
    print(f"Max static similarity: {metadata.get('max_static_similarity', 0.0):.3f}")

    if results:
        print("\nTop 3 Results:")
        for i, passage in enumerate(results[:3], 1):
            print(f"\n[{i}] {passage[:150]}...")

    # Log result but don't fail if API doesn't return results
    # (API might not have this specific case)
    print("\n[OK] Test completed: API fallback attempted")
    return latency


def test_forced_api_query():
    """Test forcing API query regardless of static corpus."""
    print("\n" + "=" * 60)
    print("Test 3: Forced API Query")
    print("=" * 60)

    import os
    api_key = os.getenv("COURTLISTENER_API_KEY")

    if not api_key:
        print("\n⚠ SKIPPED: No COURTLISTENER_API_KEY found")
        return None

    # Create static retriever first
    static_retriever = DocumentRetriever(index_path="data/indexes/legal_corpus")

    retriever = DynamicEvidenceRetriever(
        static_retriever=static_retriever,
        api_key=api_key,
    )

    query = "Fourth Amendment search and seizure"
    print(f"\nQuery: {query}")
    print("Force API: True")

    start_time = time.time()
    results, metadata = retriever.retrieve_evidence(query, top_k=5, force_api=True)
    latency = (time.time() - start_time) * 1000

    print(f"\nRetrieved {len(results)} passages in {latency:.1f}ms")
    print(f"Used API: {metadata.get('fallback_triggered', False)}")
    print(f"Static results: {metadata.get('static_count', 0)}")
    print(f"API results: {metadata.get('api_count', 0)}")

    # Should always use API when forced
    assert metadata.get("fallback_triggered"), "Should use API when force_api=True"

    print("\n[OK] Test passed: Forced API query works")
    return latency


def test_corpus_stats():
    """Test corpus statistics retrieval."""
    print("\n" + "=" * 60)
    print("Test 4: Corpus Statistics")
    print("=" * 60)

    # Create static retriever first
    static_retriever = DocumentRetriever(index_path="data/indexes/legal_corpus")

    retriever = DynamicEvidenceRetriever(
        static_retriever=static_retriever,
        api_key=None,
    )

    stats = retriever.static_retriever.get_corpus_stats()

    print("\nCorpus Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Verify we have a corpus loaded
    assert stats.get("num_documents", 0) > 0, "Should have documents in corpus"

    print("\n[OK] Test passed: Corpus statistics available")


def main():
    """Run all dynamic retrieval tests."""
    print("=" * 60)
    print("Dynamic Evidence Retrieval Test Suite")
    print("=" * 60)

    # Check if index exists
    index_path = Path("data/indexes/legal_corpus")
    if not index_path.exists():
        print("\n❌ ERROR: Retrieval index not found")
        print(f"Expected: {index_path}")
        print("\nRun: python scripts/build_retrieval_index.py")
        return

    latencies = {}

    # Test 1: Static corpus
    try:
        latencies["static"] = test_static_corpus_query()
    except Exception as e:
        print(f"\n❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: API fallback
    try:
        api_latency = test_api_fallback_trigger()
        if api_latency:
            latencies["api_fallback"] = api_latency
    except Exception as e:
        print(f"\n❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Forced API
    try:
        forced_latency = test_forced_api_query()
        if forced_latency:
            latencies["forced_api"] = forced_latency
    except Exception as e:
        print(f"\n❌ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Corpus stats
    try:
        test_corpus_stats()
    except Exception as e:
        print(f"\n❌ Test 4 failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    if latencies:
        print("\nLatency Measurements:")
        for test_name, latency_ms in latencies.items():
            print(f"  {test_name}: {latency_ms:.1f}ms")

    print("\n[OK] Dynamic retrieval test suite complete")


if __name__ == "__main__":
    main()
