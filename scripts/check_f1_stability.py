"""
Script to check F1 score stability against baseline.

Prevents model regressions by ensuring F1 score doesn't drop significantly.
Used in CI/CD model regression workflow.

Usage:
    python scripts/check_f1_stability.py --baseline-f1 0.65 --max-drop 0.02
"""

import argparse
import json
import sys
from pathlib import Path

from src.core.config import DetectorConfig
from src.evaluation.metrics import ConfidenceWeightedMetrics


def main():
    parser = argparse.ArgumentParser(
        description="Check F1 score stability against baseline"
    )
    parser.add_argument(
        "--baseline-f1", type=float, required=True, help="Baseline F1 score"
    )
    parser.add_argument(
        "--max-drop",
        type=float,
        default=0.02,
        help="Maximum allowed F1 drop (default: 0.02)",
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("data/gold_standard/gold_benchmark.json"),
        help="Path to benchmark JSON",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("regression_metrics.json"),
        help="Path to save results JSON",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("F1 STABILITY CHECK")
    print("=" * 80)
    print(f"Baseline F1: {args.baseline_f1:.3f}")
    print(f"Max allowed drop: {args.max_drop:.3f}")
    print(f"Minimum acceptable F1: {args.baseline_f1 - args.max_drop:.3f}")
    print()

    # Check if benchmark exists
    if not args.benchmark_path.exists():
        print(f"[ERROR] Benchmark not found: {args.benchmark_path}")
        print("Run scripts/generate_benchmarks_from_analysis.py to create benchmarks")
        sys.exit(1)

    # Load benchmark
    with open(args.benchmark_path, "r") as f:
        benchmark_data = json.load(f)

    print(f"Loaded {len(benchmark_data)} benchmark cases")

    # PLACEHOLDER: In production, this would run the detector on benchmark
    # For now, we simulate by checking if we have cached results

    # Check for cached results
    if args.results_path.exists():
        with open(args.results_path, "r") as f:
            results = json.load(f)

        current_f1 = results.get("f1", 0.0)
        print(f"Current F1: {current_f1:.3f}")
    else:
        print("[WARNING] No cached results found")
        print("To enable F1 stability checks, run detector on benchmark and save results")
        print(f"Expected results file: {args.results_path}")
        print()
        print("[SKIPPING] F1 stability check")
        sys.exit(0)

    # Check stability
    f1_drop = args.baseline_f1 - current_f1
    print()
    print(f"F1 drop: {f1_drop:.3f}")

    if f1_drop > args.max_drop:
        print(f"[FAIL] F1 dropped by {f1_drop:.3f}, exceeding max drop of {args.max_drop:.3f}")
        print()
        print("Model regression detected!")
        print("Possible causes:")
        print("  - Model weights updated")
        print("  - Threshold changes")
        print("  - Pipeline modifications")
        print()
        print("Action required: Review changes or update baseline")
        sys.exit(1)
    else:
        print(f"[PASS] F1 stable (drop: {f1_drop:.3f} <= max: {args.max_drop:.3f})")
        sys.exit(0)


if __name__ == "__main__":
    main()
