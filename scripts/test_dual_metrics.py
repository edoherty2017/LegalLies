#!/usr/bin/env python3
"""
Quick test script to verify the dual-metric hallucination rate implementation.

Tests the new metrics:
1. testability_rate = (Supported + Contradicted) / Total Claims
2. hallucination_rate = Contradicted / (Supported + Contradicted)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import DetectorConfig
from src.detection.unified_detector import FactualityDetector

# 5 sample questions from the test set
SAMPLE_QUESTIONS = [
    # Q1 - Gold
    'Does the inevitable discovery doctrine create a per se exception to the exclusionary rule for evidence seized after a Fourth Amendment "knock and announce" violation?',
    # Q3 - Gold
    'Did Congress divest the federal district courts of their federal-question jurisdiction under 28 U.S.C. Section 1331 over private actions brought under the Telephone Consumer Protection Act?',
    # Q4 - Silver
    'Does imposing failure-to-train liability on a district attorney\'s office for a single Brady violation contravene the rigorous culpability and causation standards of Canton and Bryan County?',
    # Q8 - Silver
    'Are the deadlines established by the bankruptcy rules for objecting to discharge jurisdictional?',
    # Q15 - Silver
    'Does a case become moot when the lone plaintiff receives an offer from the defendants to satisfy all of the plaintiff\'s claims?',
]

# Sample answers to test (simulated - in real use these would come from LLM)
SAMPLE_ANSWERS = [
    "The inevitable discovery doctrine does not create a per se exception. Evidence may still be suppressed if the Fourth Amendment violation was particularly egregious.",
    "Yes, Congress divested federal district courts of jurisdiction over private TCPA actions through the statutory text.",
    "Imposing failure-to-train liability for a single Brady violation does contravene the Canton and Bryan County standards, which require a pattern of violations.",
    "The deadlines in Bankruptcy Rule 4004 are not jurisdictional but rather procedural and subject to equitable tolling.",
    "A case does not automatically become moot when a defendant offers to satisfy all claims. The plaintiff must accept the offer for mootness to apply.",
]


def main():
    print("=" * 70)
    print("DUAL-METRIC HALLUCINATION RATE TEST")
    print("=" * 70)
    print("\nTesting new metrics:")
    print("  1. testability_rate = (Supported + Contradicted) / Total")
    print("  2. hallucination_rate = Contradicted / (Supported + Contradicted)")
    print()

    # Initialize detector
    print("Initializing detector...")
    config = DetectorConfig(device="cuda")
    detector = FactualityDetector(config)

    print("\n" + "=" * 70)
    print("RUNNING 5 SAMPLE QUESTIONS")
    print("=" * 70)

    all_results = []

    for i, (question, answer) in enumerate(zip(SAMPLE_QUESTIONS, SAMPLE_ANSWERS), 1):
        print(f"\n[{i}/5] {question[:80]}...")
        print(f"Answer: {answer[:100]}...")

        result = detector.detect(query=question, answer=answer)

        print(f"\n  DUAL METRICS:")
        print(f"    Testability:    {result.testability_rate:.1%} ({result.num_testable}/{result.num_claims} claims adjudicable)")
        print(f"    Hallucination:  {result.hallucination_rate:.1%} ({result.num_contradicted}/{result.num_testable} testable wrong)")
        print(f"  BREAKDOWN:")
        print(f"    Supported:      {result.num_supported}")
        print(f"    Contradicted:   {result.num_contradicted}")
        print(f"    Unsupported:    {result.num_unsupported}")
        print(f"  is_hallucinated: {result.is_hallucinated}")

        all_results.append(result)

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    total_claims = sum(r.num_claims for r in all_results)
    total_supported = sum(r.num_supported for r in all_results)
    total_contradicted = sum(r.num_contradicted for r in all_results)
    total_unsupported = sum(r.num_unsupported for r in all_results)
    total_testable = sum(r.num_testable for r in all_results)

    agg_testability = total_testable / total_claims if total_claims > 0 else 0
    agg_hallucination = total_contradicted / total_testable if total_testable > 0 else 0

    print(f"\nTotal claims:       {total_claims}")
    print(f"Adjudicable:        {total_testable}")
    print(f"Unadjudicable:      {total_unsupported}")
    print(f"---")
    print(f"Supported:          {total_supported}")
    print(f"Contradicted:       {total_contradicted}")
    print(f"---")
    print(f"TESTABILITY RATE:   {agg_testability:.1%}")
    print(f"HALLUCINATION RATE: {agg_hallucination:.1%}")

    # Verify the math
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    print(f"\nFormula check:")
    print(f"  testability_rate = {total_testable}/{total_claims} = {agg_testability:.4f}")
    print(f"  hallucination_rate = {total_contradicted}/{total_testable} = {agg_hallucination:.4f}")

    # Check that new fields exist
    print(f"\nNew fields present in FactualityResult:")
    print(f"  hallucination_rate: {hasattr(all_results[0], 'hallucination_rate')}")
    print(f"  testability_rate:   {hasattr(all_results[0], 'testability_rate')}")
    print(f"  num_testable:       {hasattr(all_results[0], 'num_testable')}")

    print("\n[OK] Test complete!")


if __name__ == "__main__":
    main()
