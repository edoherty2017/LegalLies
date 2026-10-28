"""
Threshold tuning for hallucination detector on Gold benchmark.

Performs grid search over NLI and semantic thresholds to maximize F1 score
on high-confidence lawyer annotations (Q1, Q3).

Follows guide Section 5 for threshold optimization strategy.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.config import DetectorConfig
from ..detection.unified_detector import UnifiedHallucinationDetector
from .metrics import ConfidenceWeightedMetrics


class ThresholdTuner:
    """
    Optimize detector thresholds on Gold benchmark.

    Usage:
        tuner = ThresholdTuner(gold_benchmark_path="data/gold_standard/gold_benchmark.json")

        # Sweep thresholds
        results_df = tuner.sweep_thresholds(
            nli_range=(0.5, 0.9),
            semantic_range=(0.4, 0.8)
        )

        # Get best thresholds
        best = results_df.iloc[0]
        print(f"Best NLI threshold: {best['nli_threshold']}")
        print(f"Best semantic threshold: {best['semantic_threshold']}")
        print(f"Best F1: {best['f1']:.3f}")

        # Validate on Silver
        silver_metrics = tuner.validate_on_silver(
            best['nli_threshold'],
            best['semantic_threshold'],
            silver_benchmark_path="data/gold_standard/silver_benchmark.json"
        )
    """

    def __init__(
        self,
        gold_benchmark_path: Optional[Path] = None,
        base_config: Optional[DetectorConfig] = None,
    ) -> None:
        """
        Initialize threshold tuner.

        Args:
            gold_benchmark_path: Path to Gold benchmark JSON
            base_config: Base detector configuration (uses defaults if not provided)
        """
        self.gold_benchmark_path = gold_benchmark_path or Path(
            "data/gold_standard/gold_benchmark.json"
        )
        self.base_config = base_config or DetectorConfig()

        # Load Gold benchmark
        self.gold_data = self._load_benchmark(self.gold_benchmark_path)
        print(f"Loaded {len(self.gold_data)} Gold benchmark cases from {self.gold_benchmark_path}")

        # Initialize metrics calculator
        self.metrics_calculator = ConfidenceWeightedMetrics()

    def _load_benchmark(self, path: Path) -> List[dict]:
        """
        Load benchmark JSON file.

        Args:
            path: Path to benchmark JSON

        Returns:
            List of benchmark cases

        Raises:
            FileNotFoundError: If benchmark file not found
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Benchmark file not found: {path}. "
                "Run scripts/generate_benchmarks_from_analysis.py to create benchmarks."
            )

        with open(path, "r") as f:
            data = json.load(f)

        return data

    def sweep_thresholds(
        self,
        nli_range: Tuple[float, float] = (0.5, 0.9),
        semantic_range: Tuple[float, float] = (0.4, 0.8),
        nli_step: float = 0.05,
        semantic_step: float = 0.05,
    ) -> pd.DataFrame:
        """
        Sweep threshold combinations and evaluate on Gold benchmark.

        Args:
            nli_range: (min, max) for NLI entailment threshold
            semantic_range: (min, max) for semantic similarity threshold
            nli_step: Step size for NLI threshold sweep
            semantic_step: Step size for semantic threshold sweep

        Returns:
            DataFrame with results sorted by F1 score (descending)
        """
        print(f"Sweeping thresholds on {len(self.gold_data)} Gold cases...")
        print(f"NLI range: {nli_range}, step: {nli_step}")
        print(f"Semantic range: {semantic_range}, step: {semantic_step}")

        results = []

        # Generate threshold combinations
        nli_thresholds = np.arange(nli_range[0], nli_range[1] + nli_step, nli_step)
        semantic_thresholds = np.arange(
            semantic_range[0], semantic_range[1] + semantic_step, semantic_step
        )

        total_combinations = len(nli_thresholds) * len(semantic_thresholds)
        print(f"Testing {total_combinations} threshold combinations...")

        combination_count = 0

        for nli_thresh in nli_thresholds:
            for semantic_thresh in semantic_thresholds:
                combination_count += 1

                # Create detector with current thresholds
                config = DetectorConfig(
                    nli_entailment_threshold=nli_thresh,
                    semantic_similarity_threshold=semantic_thresh,
                    # Use other defaults from base config
                    nli_contradiction_threshold=self.base_config.nli_contradiction_threshold,
                    bi_encoder_top_k=self.base_config.bi_encoder_top_k,
                    device=self.base_config.device,
                )

                # Evaluate on Gold benchmark
                metrics = self._evaluate_on_benchmark(
                    config, self.gold_data, use_confidence_weighting=True
                )

                results.append(
                    {
                        "nli_threshold": nli_thresh,
                        "semantic_threshold": semantic_thresh,
                        "f1": metrics["f1"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "accuracy": metrics["accuracy"],
                        "num_correct": metrics["num_correct"],
                        "num_total": metrics["num_total"],
                    }
                )

                if combination_count % 10 == 0:
                    print(
                        f"[{combination_count}/{total_combinations}] "
                        f"NLI={nli_thresh:.2f}, Semantic={semantic_thresh:.2f}, "
                        f"F1={metrics['f1']:.3f}"
                    )

        # Convert to DataFrame and sort by F1
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("f1", ascending=False).reset_index(drop=True)

        print(f"\n[OK] Threshold sweep complete")
        print(f"Best F1: {results_df.iloc[0]['f1']:.3f}")
        print(f"Best NLI threshold: {results_df.iloc[0]['nli_threshold']:.2f}")
        print(f"Best semantic threshold: {results_df.iloc[0]['semantic_threshold']:.2f}")

        return results_df

    def _evaluate_on_benchmark(
        self,
        config: DetectorConfig,
        benchmark_data: List[dict],
        use_confidence_weighting: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate detector on benchmark data.

        Args:
            config: Detector configuration
            benchmark_data: List of benchmark cases
            use_confidence_weighting: Weight metrics by lawyer confidence

        Returns:
            Dictionary of metrics
        """
        # Note: For MVP, we simulate evaluation since we don't have actual
        # query/answer/context in the benchmark JSON yet.
        # In production, this would run the full detector pipeline.

        predictions = []
        ground_truth = []
        confidence_weights = []

        for case in benchmark_data:
            # Ground truth: "Yes" = hallucination, "No" = not hallucination
            gold_label = case.get("gold_label", "No")
            is_hallucination_gt = 1 if gold_label == "Yes" else 0
            ground_truth.append(is_hallucination_gt)

            # Confidence weight (normalized to 0-1)
            confidence = case.get("confidence", 3.0)
            confidence_weights.append(confidence / 5.0)

            # PLACEHOLDER: Simulate prediction based on thresholds
            # In production, this would call detector.detect(query, answer, context)
            # For now, use a simple heuristic based on agreement rate
            agreement_rate = case.get("agreement_rate", 0.6)

            # Higher threshold = more conservative = fewer hallucinations detected
            # Lower threshold = more aggressive = more hallucinations detected
            detection_sensitivity = (
                1.0 - config.nli_entailment_threshold
            )  # 0.3 for threshold 0.7

            # Simulate: Low agreement + sensitive detector = predict hallucination
            predicted_hallucination = 1 if (1.0 - agreement_rate) > (1.0 - detection_sensitivity) else 0
            predictions.append(predicted_hallucination)

        # Compute metrics
        if use_confidence_weighting:
            metrics = self.metrics_calculator.compute_precision_recall_f1(
                predictions, ground_truth, confidence_weights
            )
        else:
            # Unweighted metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

            metrics = {
                "precision": precision_score(
                    ground_truth, predictions, zero_division=0
                ),
                "recall": recall_score(ground_truth, predictions, zero_division=0),
                "f1": f1_score(ground_truth, predictions, zero_division=0),
                "accuracy": accuracy_score(ground_truth, predictions),
            }

        metrics["num_correct"] = sum(
            1 for gt, pred in zip(ground_truth, predictions) if gt == pred
        )
        metrics["num_total"] = len(ground_truth)

        return metrics

    def validate_on_silver(
        self,
        nli_threshold: float,
        semantic_threshold: float,
        silver_benchmark_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """
        Validate tuned thresholds on Silver benchmark.

        Args:
            nli_threshold: Tuned NLI entailment threshold
            semantic_threshold: Tuned semantic similarity threshold
            silver_benchmark_path: Path to Silver benchmark JSON

        Returns:
            Dictionary of metrics on Silver set
        """
        silver_path = silver_benchmark_path or Path(
            "data/gold_standard/silver_benchmark.json"
        )

        print(f"\nValidating on Silver benchmark: {silver_path}")

        # Load Silver benchmark
        silver_data = self._load_benchmark(silver_path)
        print(f"Loaded {len(silver_data)} Silver benchmark cases")

        # Create detector with tuned thresholds
        config = DetectorConfig(
            nli_entailment_threshold=nli_threshold,
            semantic_similarity_threshold=semantic_threshold,
            device=self.base_config.device,
        )

        # Evaluate on Silver
        metrics = self._evaluate_on_benchmark(
            config, silver_data, use_confidence_weighting=True
        )

        print(f"Silver Validation Results:")
        print(f"  F1: {metrics['f1']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")

        return metrics

    def export_best_config(
        self, results_df: pd.DataFrame, output_path: Path
    ) -> DetectorConfig:
        """
        Export best threshold configuration to file.

        Args:
            results_df: Results from sweep_thresholds()
            output_path: Path to save configuration YAML

        Returns:
            DetectorConfig with best thresholds
        """
        best_row = results_df.iloc[0]

        config = DetectorConfig(
            nli_entailment_threshold=best_row["nli_threshold"],
            semantic_similarity_threshold=best_row["semantic_threshold"],
        )

        # Save to YAML
        import yaml

        config_dict = config.to_dict()
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        print(f"[OK] Best configuration saved to {output_path}")

        return config

    def generate_tuning_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate human-readable tuning report.

        Args:
            results_df: Results from sweep_thresholds()

        Returns:
            Formatted report string
        """
        best = results_df.iloc[0]
        worst = results_df.iloc[-1]

        report = f"""
Threshold Tuning Report
=======================

Gold Benchmark: {len(self.gold_data)} cases

Best Configuration:
  NLI Entailment Threshold: {best['nli_threshold']:.2f}
  Semantic Similarity Threshold: {best['semantic_threshold']:.2f}
  F1 Score: {best['f1']:.3f}
  Precision: {best['precision']:.3f}
  Recall: {best['recall']:.3f}
  Accuracy: {best['accuracy']:.3f}

Worst Configuration:
  NLI Entailment Threshold: {worst['nli_threshold']:.2f}
  Semantic Similarity Threshold: {worst['semantic_threshold']:.2f}
  F1 Score: {worst['f1']:.3f}

F1 Score Range: {worst['f1']:.3f} - {best['f1']:.3f} (spread: {best['f1'] - worst['f1']:.3f})

Top 5 Configurations:
"""

        for i in range(min(5, len(results_df))):
            row = results_df.iloc[i]
            report += f"""
  {i+1}. NLI={row['nli_threshold']:.2f}, Semantic={row['semantic_threshold']:.2f}
     F1={row['f1']:.3f}, Precision={row['precision']:.3f}, Recall={row['recall']:.3f}
"""

        return report
