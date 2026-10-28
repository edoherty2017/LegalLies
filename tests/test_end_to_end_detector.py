"""
End-to-end hallucination detection pipeline test.

Tests the complete unified detector with all layers:
- Layer 0: Universal Retrieval (dynamic with API fallback)
- Layer 1: Citation Verification
- Layer 2: NLI Entailment
- Layer 3: Semantic Similarity
- Layer 4: Aggregate Verdict
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import DetectorConfig
from src.detection.unified_detector import UnifiedHallucinationDetector


def test_baseline_detection():
    """Test hallucination detection for baseline (no sources) response."""
    print("\n" + "=" * 60)
    print("Test 1: Baseline Detection (No Sources)")
    print("=" * 60)

    # Initialize detector
    config = DetectorConfig()
    detector = UnifiedHallucinationDetector(config)

    # Test case: Question + Answer (no sources)
    question = "What is the Fourth Amendment exclusionary rule?"
    answer = (
        "The exclusionary rule prevents evidence obtained in violation of the "
        "Fourth Amendment from being used in criminal prosecutions. This rule "
        "was established to deter police misconduct and protect constitutional rights."
    )

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer[:150]}...")

    # Run detection
    print("\nRunning hallucination detection...")
    result = detector.detect(
        query=question,
        generated_answer=answer,
        retrieved_context=None,  # Baseline mode
    )

    # Display results
    print("\n" + "=" * 60)
    print("Detection Results")
    print("=" * 60)
    print(f"\nHallucination Detected: {result.is_hallucinated}")
    print(f"Hallucination Risk: {result.hallucination_risk:.3f}")
    print(f"Confidence Score: {result.confidence:.3f}")
    if result.hallucination_type:
        print(f"Hallucination Type: {result.hallucination_type}")

    print(f"\nClaims Analyzed: {len(result.claims)}")
    print(f"Structural Verification: {'PASSED' if result.structural_passed else 'FAILED'}")

    if hasattr(result, 'layer_latencies') and result.layer_latencies:
        print("\nLayer Latencies:")
        for layer_name, latency in result.layer_latencies.items():
            print(f"  {layer_name}: {latency:.1f}ms")

    print(f"\nTotal Processing Time: {result.latency_ms:.1f}ms")

    # Validate result structure
    assert isinstance(result.is_hallucinated, bool)
    assert 0 <= result.confidence <= 1
    assert len(result.claims) > 0

    print("\n[OK] Baseline detection test passed")
    return result


def test_rag_detection_with_sources():
    """Test hallucination detection for RAG response with sources."""
    print("\n" + "=" * 60)
    print("Test 2: RAG Detection (With Sources)")
    print("=" * 60)

    # Initialize detector
    config = DetectorConfig()
    detector = UnifiedHallucinationDetector(config)

    # Test case: Question + Answer + Sources
    question = "What federal statute governs the Telephone Consumer Protection Act?"
    answer = (
        "The Telephone Consumer Protection Act (TCPA) is governed by 47 U.S.C. § 227. "
        "Federal district courts have jurisdiction under 28 U.S.C. § 1331 over civil "
        "actions arising under federal law, including TCPA claims."
    )
    sources = [
        "28 U.S.C. Section 1331 grants federal courts original jurisdiction of all civil actions arising under federal law.",
        "The Telephone Consumer Protection Act (TCPA) restricts telemarketing calls and provides for a private right of action.",
    ]

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer[:150]}...")
    print(f"\nSources provided: {len(sources)}")

    # Run detection
    print("\nRunning hallucination detection...")
    result = detector.detect(
        query=question,
        generated_answer=answer,
        retrieved_context=sources,  # RAG mode
    )

    # Display results
    print("\n" + "=" * 60)
    print("Detection Results")
    print("=" * 60)
    print(f"\nHallucination Detected: {result.is_hallucinated}")
    print(f"Hallucination Risk: {result.hallucination_risk:.3f}")
    print(f"Confidence Score: {result.confidence:.3f}")
    if result.hallucination_type:
        print(f"Hallucination Type: {result.hallucination_type}")

    print(f"\nClaims Analyzed: {len(result.claims)}")
    print(f"Structural Verification: {'PASSED' if result.structural_passed else 'FAILED'}")

    print(f"\nTotal Processing Time: {result.latency_ms:.1f}ms")

    # Validate result structure
    assert isinstance(result.is_hallucinated, bool)
    assert 0 <= result.confidence <= 1

    print("\n[OK] RAG detection test passed")
    return result


def test_hallucinated_answer():
    """Test detection of clearly hallucinated answer."""
    print("\n" + "=" * 60)
    print("Test 3: Hallucinated Answer Detection")
    print("=" * 60)

    # Initialize detector
    config = DetectorConfig()
    detector = UnifiedHallucinationDetector(config)

    # Test case: Clearly false legal claim
    question = "What is the Supreme Court case that established qualified immunity?"
    answer = (
        "The Supreme Court case of Johnson v. Smith (1965) established the doctrine "
        "of qualified immunity. This landmark decision was written by Justice Roberts "
        "and has been cited over 10,000 times in subsequent cases."
    )
    # Note: This is fabricated - the real case is Harlow v. Fitzgerald (1982)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer[:150]}...")
    print("\nNote: This answer contains fabricated citations")

    # Run detection
    print("\nRunning hallucination detection...")
    result = detector.detect(
        query=question,
        generated_answer=answer,
        retrieved_context=None,
    )

    # Display results
    print("\n" + "=" * 60)
    print("Detection Results")
    print("=" * 60)
    print(f"\nHallucination Detected: {result.is_hallucinated}")
    print(f"Hallucination Risk: {result.hallucination_risk:.3f}")
    print(f"Confidence Score: {result.confidence:.3f}")
    if result.hallucination_type:
        print(f"Hallucination Type: {result.hallucination_type}")

    print(f"\nClaims Analyzed: {len(result.claims)}")
    print(f"\nTotal Processing Time: {result.latency_ms:.1f}ms")

    # We expect this to be detected as hallucinated or high risk
    # (since the case doesn't exist in our corpus)
    assert result.is_hallucinated or result.hallucination_risk > 0.5

    print("\n[OK] Hallucination detection test passed")
    return result


def test_retrieval_with_authority_ranking():
    """Test that authority ranking is being used in retrieval."""
    print("\n" + "=" * 60)
    print("Test 4: Authority-Weighted Retrieval")
    print("=" * 60)

    # Initialize detector
    config = DetectorConfig()
    detector = UnifiedHallucinationDetector(config)

    # Access the retriever
    retriever = detector.retriever

    # Check if authority ranking is enabled
    print(f"\nAuthority ranking enabled: {retriever.use_authority_ranking}")
    print(f"Citation graph loaded: {retriever.citation_graph is not None}")
    print(f"Hybrid ranker loaded: {retriever.hybrid_ranker is not None}")

    if retriever.citation_graph:
        print(f"Citation graph documents: {len(retriever.citation_graph.documents)}")

    # Test retrieval
    query = "What is qualified immunity?"
    print(f"\nTest query: {query}")

    results = retriever.retrieve_evidence(query, top_k=5)
    print(f"\nRetrieved {len(results)} passages")

    if results:
        print("\nTop 3 results:")
        for i, passage in enumerate(results[:3], 1):
            print(f"\n  [{i}] {passage[:100]}...")

    assert len(results) > 0

    print("\n[OK] Authority ranking test passed")


def main():
    """Run all end-to-end tests."""
    print("=" * 60)
    print("End-to-End Hallucination Detection Test Suite")
    print("=" * 60)

    # Check if index exists
    index_path = Path("data/indexes/legal_corpus")
    if not index_path.exists():
        print("\n[ERROR] Retrieval index not found")
        print(f"Expected: {index_path}")
        print("\nRun: python scripts/build_retrieval_index.py")
        return

    results = {}

    # Test 1: Baseline detection
    try:
        results['baseline'] = test_baseline_detection()
    except Exception as e:
        print(f"\n[ERROR] Test 1 failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: RAG detection
    try:
        results['rag'] = test_rag_detection_with_sources()
    except Exception as e:
        print(f"\n[ERROR] Test 2 failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Hallucination detection
    try:
        results['hallucination'] = test_hallucinated_answer()
    except Exception as e:
        print(f"\n[ERROR] Test 3 failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Authority ranking
    try:
        test_retrieval_with_authority_ranking()
    except Exception as e:
        print(f"\n[ERROR] Test 4 failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("Test Suite Summary")
    print("=" * 60)

    if results:
        print("\nPerformance Metrics:")
        for test_name, result in results.items():
            if result:
                status = "HALLUCINATED" if result.is_hallucinated else "SUPPORTED"
                print(f"  {test_name}: {result.latency_ms:.1f}ms "
                      f"({status}, risk={result.hallucination_risk:.3f})")

    print("\n[OK] End-to-end test suite complete")


if __name__ == "__main__":
    main()
