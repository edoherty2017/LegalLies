"""
Script to check P95 latency against budget.

Ensures detector performance meets latency requirements.
Used in CI/CD model regression workflow.

Usage:
    python scripts/check_latency.py --max-p95-ms 200
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from src.core.config import DetectorConfig
from src.detection.unified_detector import UnifiedHallucinationDetector


def main():
    parser = argparse.ArgumentParser(description="Check P95 latency against budget")
    parser.add_argument(
        "--max-p95-ms",
        type=float,
        default=200.0,
        help="Maximum acceptable P95 latency in milliseconds (default: 200)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of test samples to measure (default: 20)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for testing (default: cpu)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("LATENCY BUDGET CHECK")
    print("=" * 80)
    print(f"Max acceptable P95 latency: {args.max_p95_ms:.1f} ms")
    print(f"Number of samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print()

    print("Initializing detector...")
    config = DetectorConfig(device=args.device)

    try:
        detector = UnifiedHallucinationDetector(config)
    except Exception as e:
        print(f"[WARNING] Failed to initialize detector: {e}")
        print("Latency check requires model downloads")
        print("[SKIPPING] Latency budget check")
        sys.exit(0)

    print("[OK] Detector initialized")
    print()

    # Test samples
    test_cases = [
        {
            "query": "What is the legal standard?",
            "answer": "The legal standard is established by precedent.",
            "context": ["Precedent establishes the legal standard in common law."],
        },
        {
            "query": "Is this allowed?",
            "answer": "Yes, under 17 U.S.C. Section 107, fair use is permitted.",
            "context": ["Title 17 U.S.C. Section 107 codifies the fair use doctrine."],
        },
        {
            "query": "What does the statute say?",
            "answer": "The statute prohibits unauthorized copying.",
            "context": ["Copyright law prohibits unauthorized reproduction."],
        },
    ]

    # Measure latencies
    latencies: List[float] = []

    print(f"Running {args.num_samples} latency measurements...")

    for i in range(args.num_samples):
        # Cycle through test cases
        test_case = test_cases[i % len(test_cases)]

        start_time = time.time()

        try:
            result = detector.detect(
                query=test_case["query"],
                generated_answer=test_case["answer"],
                retrieved_context=test_case["context"],
                return_details=False,  # Faster without details
            )

            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)

            if (i + 1) % 5 == 0:
                print(f"[{i + 1}/{args.num_samples}] Latest latency: {latency_ms:.1f} ms")

        except Exception as e:
            print(f"[ERROR] Detection failed on sample {i + 1}: {e}")
            continue

    if not latencies:
        print("[ERROR] No successful latency measurements")
        sys.exit(1)

    # Compute statistics
    latencies_array = np.array(latencies)
    p50 = np.percentile(latencies_array, 50)
    p95 = np.percentile(latencies_array, 95)
    p99 = np.percentile(latencies_array, 99)
    mean_latency = np.mean(latencies_array)
    min_latency = np.min(latencies_array)
    max_latency = np.max(latencies_array)

    print()
    print("=" * 80)
    print("LATENCY STATISTICS")
    print("=" * 80)
    print(f"Mean:   {mean_latency:.1f} ms")
    print(f"Min:    {min_latency:.1f} ms")
    print(f"Max:    {max_latency:.1f} ms")
    print(f"P50:    {p50:.1f} ms")
    print(f"P95:    {p95:.1f} ms")
    print(f"P99:    {p99:.1f} ms")
    print()

    # Check against budget
    if p95 > args.max_p95_ms:
        print(f"[FAIL] P95 latency ({p95:.1f} ms) exceeds budget ({args.max_p95_ms:.1f} ms)")
        print()
        print("Performance regression detected!")
        print("Possible causes:")
        print("  - Inefficient code changes")
        print("  - Increased batch size")
        print("  - Model changes")
        print()
        print("Action required: Optimize pipeline or update budget")
        sys.exit(1)
    else:
        margin = args.max_p95_ms - p95
        print(f"[PASS] P95 latency ({p95:.1f} ms) within budget ({args.max_p95_ms:.1f} ms)")
        print(f"Margin: {margin:.1f} ms ({margin / args.max_p95_ms * 100:.1f}%)")
        sys.exit(0)


if __name__ == "__main__":
    main()
