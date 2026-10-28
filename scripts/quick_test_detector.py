"""
Quick detector test with sample legal questions.

Tests basic functionality before running full evaluation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import DetectorConfig
from src.detection.unified_detector import UnifiedHallucinationDetector


def test_detector():
    """Test detector with sample legal questions."""
    print("="*60)
    print("Quick Detector Test")
    print("="*60)

    # Initialize detector
    config = DetectorConfig(
        nli_entailment_threshold=0.70,
        semantic_similarity_threshold=0.50,
        device="cpu",
    )

    print("\nInitializing detector...")
    detector = UnifiedHallucinationDetector(config)
    print("Detector initialized OK\n")

    # Test Case 1: Grounded claim (should NOT be flagged)
    print("Test 1: Grounded Claim")
    print("-" * 60)
    query1 = "Is murder a crime under US law?"
    answer1 = "Yes, murder is a serious felony under US criminal law."
    context1 = ["Murder is classified as a felony in the United States Code."]

    result1 = detector.detect(
        query=query1,
        generated_answer=answer1,
        retrieved_context=context1
    )

    print(f"Query: {query1}")
    print(f"Answer: {answer1}")
    print(f"Result: {'HALLUCINATION' if result1.is_hallucinated else 'GROUNDED'}")
    print(f"Risk Score: {result1.hallucination_risk:.2f}")
    print(f"Failure Layer: {result1.hallucination_type or 'None'}\n")

    # Test Case 2: Contradictory claim (should BE flagged)
    print("Test 2: Contradictory Claim")
    print("-" * 60)
    query2 = "What is the speed limit on highways?"
    answer2 = "The speed limit on highways is 100 mph."
    context2 = ["The maximum speed limit on most US highways is 65-70 mph."]

    result2 = detector.detect(
        query=query2,
        generated_answer=answer2,
        retrieved_context=context2
    )

    print(f"Query: {query2}")
    print(f"Answer: {answer2}")
    print(f"Result: {'HALLUCINATION' if result2.is_hallucinated else 'GROUNDED'}")
    print(f"Risk Score: {result2.hallucination_risk:.2f}")
    print(f"Failure Layer: {result2.hallucination_type or 'None'}\n")

    # Test Case 3: Fabricated citation (should BE flagged)
    print("Test 3: Fabricated Citation")
    print("-" * 60)
    query3 = "What is the legal standard for search and seizure?"
    answer3 = "See 28 U.S.C. Section 9999, which establishes the standard."
    context3 = ["The Fourth Amendment protects against unreasonable searches."]

    result3 = detector.detect(
        query=query3,
        generated_answer=answer3,
        retrieved_context=context3
    )

    print(f"Query: {query3}")
    print(f"Answer: {answer3}")
    print(f"Result: {'HALLUCINATION' if result3.is_hallucinated else 'GROUNDED'}")
    print(f"Risk Score: {result3.hallucination_risk:.2f}")
    print(f"Failure Layer: {result3.hallucination_type or 'None'}\n")

    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    print(f"Test 1 (Grounded):       {'PASS' if not result1.is_hallucinated else 'FAIL'}")
    print(f"Test 2 (Contradiction):  {'PASS' if result2.is_hallucinated else 'FAIL'}")
    print(f"Test 3 (Fabrication):    {'PASS' if result3.is_hallucinated else 'FAIL'}")

    all_passed = (not result1.is_hallucinated) and result2.is_hallucinated and result3.is_hallucinated
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")


if __name__ == "__main__":
    test_detector()
