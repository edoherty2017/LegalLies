"""
Test unified detector with automatic retrieval.

Tests both scenarios:
1. With provided sources (RAG+Web variant)
2. Without sources (Baseline variant with auto-retrieval)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import DetectorConfig
from src.detection.unified_detector import UnifiedHallucinationDetector


def test_detector_with_retrieval():
    """Test detector with automatic retrieval enabled."""
    print("=" * 60)
    print("Testing Unified Detector with Automatic Retrieval")
    print("=" * 60)

    # Initialize detector with retrieval enabled
    config = DetectorConfig(
        nli_entailment_threshold=0.70,
        semantic_similarity_threshold=0.50,
        device="cpu",
        retrieval_enabled=True,
        retrieval_top_k=3,
    )

    print("\nInitializing detector...")
    detector = UnifiedHallucinationDetector(config)
    print()

    # Test Case 1: Baseline variant (no sources provided)
    print("=" * 60)
    print("Test 1: Baseline Variant (Auto-Retrieval)")
    print("=" * 60)
    query1 = "Does the inevitable discovery doctrine create a per se exception to the exclusionary rule?"
    answer1 = "The inevitable discovery doctrine does not automatically create a per se exception to the exclusionary rule."

    result1 = detector.detect(
        query=query1,
        generated_answer=answer1,
        retrieved_context=None,  # No sources - should trigger auto-retrieval
    )

    print(f"\nQuery: {query1}")
    print(f"Answer: {answer1[:100]}...")
    print(f"\nResult: {'HALLUCINATION' if result1.is_hallucinated else 'GROUNDED'}")
    print(f"Risk Score: {result1.hallucination_risk:.2f}")
    print(f"Confidence: {result1.confidence:.2f}")
    print(f"Failure Layer: {result1.hallucination_type or 'None'}")
    print(f"Latency: {result1.latency_ms:.0f}ms")
    if 'retrieval' in result1.layer_latencies:
        print(f"  - Retrieval: {result1.layer_latencies['retrieval']:.0f}ms")

    # Test Case 2: RAG+Web variant (sources provided)
    print("\n" + "=" * 60)
    print("Test 2: RAG+Web Variant (Provided Sources)")
    print("=" * 60)
    query2 = "What is the Fourth Amendment exclusionary rule?"
    answer2 = "The exclusionary rule prevents illegally obtained evidence from being used in court."
    context2 = [
        "The Fourth Amendment exclusionary rule prohibits the use of evidence obtained in violation of constitutional rights.",
        "Evidence obtained through illegal searches cannot be admitted in criminal prosecutions.",
    ]

    result2 = detector.detect(
        query=query2,
        generated_answer=answer2,
        retrieved_context=context2,  # Sources provided
    )

    print(f"\nQuery: {query2}")
    print(f"Answer: {answer2}")
    print(f"Sources: {len(context2)} provided")
    print(f"\nResult: {'HALLUCINATION' if result2.is_hallucinated else 'GROUNDED'}")
    print(f"Risk Score: {result2.hallucination_risk:.2f}")
    print(f"Confidence: {result2.confidence:.2f}")
    print(f"Failure Layer: {result2.hallucination_type or 'None'}")
    print(f"Latency: {result2.latency_ms:.0f}ms")

    # Test Case 3: Force retrieval (augment provided sources)
    print("\n" + "=" * 60)
    print("Test 3: Force Retrieval (Augment Sources)")
    print("=" * 60)
    query3 = "What is federal question jurisdiction?"
    answer3 = "Federal question jurisdiction is granted under 28 U.S.C. Section 1331."
    context3 = ["Federal courts have jurisdiction over cases arising under federal law."]

    result3 = detector.detect(
        query=query3,
        generated_answer=answer3,
        retrieved_context=context3,
        force_retrieve=True,  # Augment provided sources with retrieved ones
    )

    print(f"\nQuery: {query3}")
    print(f"Answer: {answer3}")
    print(f"Sources: {len(context3)} provided (augmented with retrieval)")
    print(f"\nResult: {'HALLUCINATION' if result3.is_hallucinated else 'GROUNDED'}")
    print(f"Risk Score: {result3.hallucination_risk:.2f}")
    print(f"Confidence: {result3.confidence:.2f}")
    print(f"Failure Layer: {result3.hallucination_type or 'None'}")
    print(f"Latency: {result3.latency_ms:.0f}ms")
    if 'retrieval' in result3.layer_latencies:
        print(f"  - Retrieval: {result3.layer_latencies['retrieval']:.0f}ms")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Test 1 (Baseline Auto-Retrieval): {'PASS' if not result1.is_hallucinated else 'FAIL'}")
    print(f"Test 2 (RAG+Web Provided):        {'PASS' if not result2.is_hallucinated else 'FAIL'}")
    print(f"Test 3 (Force Retrieval):          {'PASS' if not result3.is_hallucinated else 'FAIL'}")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_detector_with_retrieval()
