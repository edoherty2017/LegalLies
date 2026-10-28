"""
Comprehensive detector evaluation script.

Runs the hallucination detector on:
1. Gold benchmark (Q1, Q3) - High confidence
2. Silver benchmark (Q2, Q4-Q11, Q14) - Medium confidence
3. Generates evaluation report with metrics

Note: This script requires ML models to be downloaded.
Run with: python -m poetry run python scripts/evaluate_detector.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import DetectorConfig
from src.detection.unified_detector import UnifiedHallucinationDetector
from src.evaluation.metrics import ConfidenceWeightedMetrics


def load_benchmark(filepath: Path) -> List[Dict]:
    """Load benchmark JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_on_benchmark(
    detector: UnifiedHallucinationDetector,
    benchmark_data: List[Dict],
    benchmark_name: str,
) -> Dict:
    """
    Evaluate detector on benchmark.

    Args:
        detector: Hallucination detector instance
        benchmark_data: List of benchmark cases
        benchmark_name: Name of benchmark (for reporting)

    Returns:
        Dict with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {benchmark_name} Benchmark")
    print(f"{'='*60}\n")

    results = []
    predictions = []
    ground_truth = []
    probabilities = []
    confidence_weights = []

    for i, case in enumerate(benchmark_data, 1):
        print(f"Case {i}/{len(benchmark_data)}: {case['query_id']}")

        # Extract case data
        query_text = case.get("query_text", "")
        answer_text = case.get("generated_answer", case.get("answer_text", ""))
        sources = case.get("retrieved_sources", [])

        # Get ground truth label
        if "gold_label" in case:
            # Gold benchmark: binary label
            true_label = 1 if case["gold_label"] == "Hallucination" else 0
        elif "majority_label" in case:
            # Silver benchmark: majority vote
            true_label = 1 if case["majority_label"] == "Hallucination" else 0
        else:
            print(f"  WARNING: No ground truth label found, skipping")
            continue

        try:
            # Run detector
            result = detector.detect(
                query=query_text,
                generated_answer=answer_text,
                retrieved_context=sources,
            )

            # Record results
            pred_label = 1 if result.is_hallucinated else 0
            predictions.append(pred_label)
            ground_truth.append(true_label)
            probabilities.append(result.hallucination_risk)
            confidence_weights.append(case.get("confidence", 3.0) / 5.0)

            # Print result
            match = "OK" if pred_label == true_label else "FAIL"
            print(f"  Predicted: {pred_label} | True: {true_label} | {match}")
            print(
                f"  Confidence: {result.hallucination_risk:.2f} | Layer: {result.hallucination_type or 'None'}"
            )

            results.append(
                {
                    "query_id": case["query_id"],
                    "predicted": pred_label,
                    "true_label": true_label,
                    "hallucination_risk": result.hallucination_risk,
                    "failure_layer": result.hallucination_type,
                    "failure_reason": result.failure_reason,
                    "confidence_weight": case.get("confidence", 3.0) / 5.0,
                }
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Compute metrics
    metrics_calc = ConfidenceWeightedMetrics()

    # Unweighted metrics
    unweighted = metrics_calc.compute_precision_recall_f1(predictions, ground_truth)

    # Weighted metrics
    weighted = metrics_calc.compute_precision_recall_f1(
        predictions, ground_truth, confidence_weights
    )

    # Brier score
    brier = metrics_calc.compute_brier_score(probabilities, ground_truth)

    # ECE
    ece = metrics_calc.compute_ece(probabilities, ground_truth, num_bins=10)

    # Bootstrap CIs
    ci_results = metrics_calc.bootstrap_confidence_intervals(
        predictions, ground_truth, n_iterations=1000
    )

    print(f"\n{benchmark_name} Benchmark Results:")
    print(f"{'='*60}")
    print(f"Unweighted Metrics:")
    print(f"  Precision: {unweighted['precision']:.3f}")
    print(f"  Recall:    {unweighted['recall']:.3f}")
    print(f"  F1:        {unweighted['f1']:.3f}")
    print(f"  Accuracy:  {unweighted['accuracy']:.3f}")

    print(f"\nWeighted Metrics (by lawyer confidence):")
    print(f"  Precision: {weighted['precision']:.3f}")
    print(f"  Recall:    {weighted['recall']:.3f}")
    print(f"  F1:        {weighted['f1']:.3f}")
    print(f"  Accuracy:  {weighted['accuracy']:.3f}")

    print(f"\nCalibration Metrics:")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  ECE:         {ece:.4f}")

    print(f"\nBootstrap 95% Confidence Intervals:")
    for metric_name, (mean, lower, upper) in ci_results.items():
        print(f"  {metric_name:10} {mean:.3f} [{lower:.3f}, {upper:.3f}]")

    return {
        "benchmark_name": benchmark_name,
        "num_cases": len(benchmark_data),
        "num_evaluated": len(predictions),
        "unweighted_metrics": unweighted,
        "weighted_metrics": weighted,
        "brier_score": brier,
        "ece": ece,
        "confidence_intervals": ci_results,
        "detailed_results": results,
    }


def main():
    """Main evaluation function."""
    print("="*60)
    print("Legal Hallucination Detector Evaluation")
    print("="*60)

    # Initialize detector
    print("\nInitializing detector...")
    config = DetectorConfig(
        nli_entailment_threshold=0.70,
        semantic_similarity_threshold=0.50,
        device="cpu",  # Use CPU for testing (change to "cuda" if GPU available)
        max_source_chars=1500,
        max_claim_chars=500,
    )

    print("\nNOTE: This will download ML models (~1-2GB) on first run.")
    print("Models will be cached for future runs.")
    print("\nDownloading models (this may take 5-10 minutes)...")

    detector = UnifiedHallucinationDetector(config)
    print("Detector initialized OK")

    # Load benchmarks
    gold_path = Path("data/gold_standard/gold_benchmark.json")
    silver_path = Path("data/gold_standard/silver_benchmark.json")

    gold_data = load_benchmark(gold_path)
    silver_data = load_benchmark(silver_path)

    print(f"\nLoaded benchmarks:")
    print(f"  Gold:   {len(gold_data)} cases")
    print(f"  Silver: {len(silver_data)} cases")

    # Evaluate on Gold benchmark
    gold_results = evaluate_on_benchmark(detector, gold_data, "Gold")

    # Evaluate on Silver benchmark
    silver_results = evaluate_on_benchmark(detector, silver_data, "Silver")

    # Generate final report
    print(f"\n{'='*60}")
    print("FINAL EVALUATION SUMMARY")
    print(f"{'='*60}\n")

    print(f"Gold Benchmark (High Confidence):")
    print(f"  F1: {gold_results['unweighted_metrics']['f1']:.3f}")
    print(f"  ECE: {gold_results['ece']:.4f}")
    print(f"  Brier: {gold_results['brier_score']:.4f}")

    print(f"\nSilver Benchmark (Medium Confidence):")
    print(f"  F1: {silver_results['unweighted_metrics']['f1']:.3f}")
    print(f"  ECE: {silver_results['ece']:.4f}")
    print(f"  Brier: {silver_results['brier_score']:.4f}")

    # Save results
    output_path = Path("evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {"gold": gold_results, "silver": silver_results}, f, indent=2
        )

    print(f"\nResults saved to: {output_path}")

    # Success criteria
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*60}\n")

    gold_f1 = gold_results["unweighted_metrics"]["f1"]
    silver_f1 = silver_results["unweighted_metrics"]["f1"]
    gold_ece = gold_results["ece"]

    checks = [
        ("Gold F1 >= 0.85", gold_f1 >= 0.85, gold_f1),
        ("Silver F1 >= 0.70", silver_f1 >= 0.70, silver_f1),
        ("Gold ECE <= 0.10", gold_ece <= 0.10, gold_ece),
    ]

    for check_name, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{status:8} {check_name:20} (actual: {value:.3f})")

    print()


if __name__ == "__main__":
    main()
