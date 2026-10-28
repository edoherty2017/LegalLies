#!/usr/bin/env python3
"""
Run sanity tests on synthetic data to verify detector behavior.

Tests the factuality detector on simple examples with provided evidence.
"""

import sys
import time
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import FactualityDetector
from src.core.config import DetectorConfig


def run_tier(detector, tier_name: str, input_path: Path) -> pd.DataFrame:
    """Run detection on a single tier of tests."""
    df = pd.read_csv(input_path)
    print(f"\n{'='*60}")
    print(f"Running {tier_name}: {len(df)} test cases")
    print(f"{'='*60}")

    results = []

    for idx, row in df.iterrows():
        test_id = row['id']
        evidence = row['evidence_text']
        question = row['question']
        answer = row['answer']
        expected = row['expected_fact_label']

        print(f"  [{idx+1}/{len(df)}] {test_id}...", end=" ")

        try:
            # Run detection with provided evidence
            result = detector.detect(
                query=question,
                answer=answer,
                evidence=[evidence],
            )

            # Map result to label
            if result.is_hallucinated:
                detected_label = "HALLUCINATED"
            else:
                detected_label = "CORRECT"

            correct = (detected_label == expected)
            print(f"{detected_label} (expected: {expected}) {'✓' if correct else '✗'}")

            results.append({
                "id": test_id,
                "expected_label": expected,
                "detected_label": detected_label,
                "correct": correct,
                "num_claims": result.num_claims,
                "num_contradicted": result.num_contradicted,
                "num_supported": result.num_supported,
                "num_unsupported": result.num_unsupported,
                "contradiction_rate": round(result.contradiction_rate, 4),
                "support_rate": round(result.support_rate, 4),
                "latency_ms": round(result.latency_ms, 1),
            })

        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")
            results.append({
                "id": test_id,
                "expected_label": expected,
                "detected_label": "ERROR",
                "correct": False,
                "num_claims": 0,
                "num_contradicted": 0,
                "num_supported": 0,
                "num_unsupported": 0,
                "contradiction_rate": 0.0,
                "support_rate": 0.0,
                "latency_ms": 0,
            })

    return pd.DataFrame(results)


def print_tier_summary(tier_name: str, results_df: pd.DataFrame):
    """Print summary for a tier."""
    total = len(results_df)
    correct = results_df['correct'].sum()
    accuracy = correct / total * 100

    print(f"\n{tier_name} Summary: {correct}/{total} ({accuracy:.1f}%)")

    # By expected label
    for label in results_df['expected_label'].unique():
        subset = results_df[results_df['expected_label'] == label]
        label_correct = subset['correct'].sum()
        label_total = len(subset)
        print(f"  {label}: {label_correct}/{label_total} ({label_correct/label_total*100:.1f}%)")

    # Show errors
    errors = results_df[~results_df['correct']]
    if len(errors) > 0:
        print(f"\n  Errors ({len(errors)}):")
        for _, row in errors.iterrows():
            print(f"    {row['id']}: expected {row['expected_label']}, got {row['detected_label']} (contradictions={row['num_contradicted']})")


def main():
    # Initialize detector (no retrieval needed - using provided evidence)
    print("Initializing FactualityDetector...")
    config = DetectorConfig(
        retrieval_enabled=False,
        device="cuda",
    )
    detector = FactualityDetector(config)
    print("[OK] Detector initialized\n")

    test_dir = Path("experiments/synthetic_sanity_tests")
    all_results = {}

    # Run each tier
    tiers = [
        ("Tier 0", test_dir / "tier0_synthetic.csv"),
        ("Tier 1", test_dir / "tier1_synthetic.csv"),
        ("Tier 2", test_dir / "tier2_synthetic.csv"),
    ]

    start_time = time.time()

    for tier_name, input_path in tiers:
        if input_path.exists():
            results_df = run_tier(detector, tier_name, input_path)
            all_results[tier_name] = results_df

            # Save results
            output_path = test_dir / f"{tier_name.lower().replace(' ', '')}_factuality_results.csv"
            results_df.to_csv(output_path, index=False)
            print(f"  Saved to: {output_path}")

    elapsed = time.time() - start_time

    # Print summaries
    print(f"\n{'='*60}")
    print("SANITY TEST RESULTS (Factuality Detector)")
    print(f"{'='*60}")
    print(f"Total time: {elapsed:.1f}s")

    for tier_name, results_df in all_results.items():
        print_tier_summary(tier_name, results_df)

    # Combined summary
    if all_results:
        combined = pd.concat(all_results.values())
        total = len(combined)
        correct = combined['correct'].sum()
        print(f"\n{'='*60}")
        print(f"COMBINED: {correct}/{total} ({correct/total*100:.1f}%)")

        # By expected label type
        print("\nBy Expected Label:")
        for label in ["CORRECT", "HALLUCINATED"]:
            subset = combined[combined['expected_label'] == label]
            if len(subset) > 0:
                label_correct = subset['correct'].sum()
                print(f"  {label}: {label_correct}/{len(subset)} ({label_correct/len(subset)*100:.1f}%)")

        # Confusion matrix
        print("\nConfusion Matrix:")
        print("                    Detected")
        print("                 CORRECT  HALLUCINATED  ERROR")
        for expected in ["CORRECT", "HALLUCINATED"]:
            row = combined[combined['expected_label'] == expected]
            correct_count = (row['detected_label'] == 'CORRECT').sum()
            halluc_count = (row['detected_label'] == 'HALLUCINATED').sum()
            error_count = (row['detected_label'] == 'ERROR').sum()
            print(f"Expected {expected:12} {correct_count:7}  {halluc_count:12}  {error_count:5}")


if __name__ == "__main__":
    main()
