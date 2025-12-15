"""
Stratified reporting by evaluability tier (Gold vs Silver vs Overall).

Generates separate metrics for:
- Gold tier (high confidence, Q1/Q3)
- Silver tier (medium confidence, Q2/Q4-Q11/Q14-Q15)
- Overall (confidence-weighted combination)

Follows guide Section 4 for stratified evaluation strategy.
"""

from typing import Dict, List, Optional

import pandas as pd

from .metrics import ConfidenceWeightedMetrics


class StratifiedReporter:
    """
    Generate stratified evaluation reports by confidence tier.

    Usage:
        reporter = StratifiedReporter()

        # Prepare results DataFrame
        results = pd.DataFrame([
            {"query_id": "Q1", "tier": "gold", "prediction": 0, "ground_truth": 0, "probability": 0.1, "confidence": 4.0},
            {"query_id": "Q3", "tier": "gold", "prediction": 1, "ground_truth": 1, "probability": 0.9, "confidence": 4.4},
            {"query_id": "Q2", "tier": "silver", "prediction": 0, "ground_truth": 1, "probability": 0.4, "confidence": 3.2},
            # ...
        ])

        # Generate report
        report = reporter.generate_report(results)
        print(report)
    """

    def __init__(self) -> None:
        """Initialize stratified reporter."""
        self.metrics_calculator = ConfidenceWeightedMetrics()

    def generate_report(
        self, results: pd.DataFrame, include_ci: bool = True
    ) -> str:
        """
        Generate stratified report by tier.

        Args:
            results: DataFrame with columns:
                - query_id: Query identifier
                - tier: "gold", "silver", or "excluded"
                - prediction: Binary prediction (0/1)
                - ground_truth: Binary ground truth (0/1)
                - probability: Predicted probability (0-1)
                - confidence: Lawyer confidence (1-5)
            include_ci: Include bootstrap confidence intervals

        Returns:
            Formatted stratified report
        """
        # Separate by tier
        gold_results = results[results["tier"] == "gold"]
        silver_results = results[results["tier"] == "silver"]

        # Overall (all tiers except excluded)
        overall_results = results[results["tier"].isin(["gold", "silver"])]

        report = "=" * 80 + "\n"
        report += "STRATIFIED EVALUATION REPORT\n"
        report += "=" * 80 + "\n\n"

        # Gold tier metrics
        if len(gold_results) > 0:
            report += self._format_tier_section(
                "GOLD TIER (High Confidence)", gold_results, include_ci
            )
            report += "\n"

        # Silver tier metrics
        if len(silver_results) > 0:
            report += self._format_tier_section(
                "SILVER TIER (Medium Confidence)", silver_results, include_ci
            )
            report += "\n"

        # Overall metrics (confidence-weighted)
        if len(overall_results) > 0:
            report += self._format_tier_section(
                "OVERALL (Confidence-Weighted)", overall_results, include_ci, weighted=True
            )
            report += "\n"

        report += "=" * 80 + "\n"

        return report

    def _format_tier_section(
        self, title: str, tier_results: pd.DataFrame, include_ci: bool, weighted: bool = True
    ) -> str:
        """
        Format metrics section for a specific tier.

        Args:
            title: Section title
            tier_results: Results for this tier
            include_ci: Include confidence intervals
            weighted: Use confidence weighting

        Returns:
            Formatted section string
        """
        section = f"{title}\n"
        section += "-" * 80 + "\n"

        # Extract data
        predictions = tier_results["prediction"].tolist()
        ground_truth = tier_results["ground_truth"].tolist()
        probabilities = tier_results["probability"].tolist()
        confidence_weights = None

        if weighted and "confidence" in tier_results.columns:
            # Normalize confidence to 0-1
            confidence_weights = (tier_results["confidence"] / 5.0).tolist()

        # Compute metrics
        metrics = self.metrics_calculator.compute_all_metrics(
            predictions=predictions,
            probabilities=probabilities,
            ground_truth=ground_truth,
            confidence_weights=confidence_weights,
            compute_ci=include_ci,
        )

        # Format metrics
        section += f"Number of cases: {len(tier_results)}\n"

        if "confidence" in tier_results.columns:
            section += f"Average confidence: {tier_results['confidence'].mean():.2f} +/- {tier_results['confidence'].std():.2f}\n"

        section += "\n"
        section += f"F1 Score:   {metrics['f1']:.3f}\n"
        section += f"Precision:  {metrics['precision']:.3f}\n"
        section += f"Recall:     {metrics['recall']:.3f}\n"
        section += f"Accuracy:   {metrics['accuracy']:.3f}\n"
        section += "\n"
        section += f"Brier Score: {metrics['brier_score']:.3f}\n"
        section += f"ECE:         {metrics['ece']:.3f}\n"
        section += "\n"

        # Confusion matrix
        cm = metrics["confusion_matrix"]
        section += "Confusion Matrix:\n"
        section += f"  TN={cm[0][0]:.1f}  FP={cm[0][1]:.1f}\n"
        section += f"  FN={cm[1][0]:.1f}  TP={cm[1][1]:.1f}\n"

        # Confidence intervals
        if include_ci and "confidence_intervals" in metrics:
            section += "\n95% Confidence Intervals:\n"
            ci = metrics["confidence_intervals"]
            for metric_name, (mean, lower, upper) in ci.items():
                section += f"  {metric_name.capitalize()}: {mean:.3f} [{lower:.3f}, {upper:.3f}]\n"

        return section

    def export_to_csv(
        self, results: pd.DataFrame, output_path: str
    ) -> None:
        """
        Export stratified metrics to CSV.

        Args:
            results: Results DataFrame
            output_path: Path to save CSV
        """
        # Compute metrics for each tier
        tiers_summary = []

        for tier_name in ["gold", "silver"]:
            tier_results = results[results["tier"] == tier_name]

            if len(tier_results) == 0:
                continue

            predictions = tier_results["prediction"].tolist()
            ground_truth = tier_results["ground_truth"].tolist()
            probabilities = tier_results["probability"].tolist()
            confidence_weights = (tier_results["confidence"] / 5.0).tolist()

            metrics = self.metrics_calculator.compute_all_metrics(
                predictions=predictions,
                probabilities=probabilities,
                ground_truth=ground_truth,
                confidence_weights=confidence_weights,
                compute_ci=False,  # Skip CI for CSV export
            )

            tiers_summary.append(
                {
                    "tier": tier_name,
                    "num_cases": len(tier_results),
                    "avg_confidence": tier_results["confidence"].mean(),
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "accuracy": metrics["accuracy"],
                    "brier_score": metrics["brier_score"],
                    "ece": metrics["ece"],
                }
            )

        # Overall
        overall_results = results[results["tier"].isin(["gold", "silver"])]
        if len(overall_results) > 0:
            predictions = overall_results["prediction"].tolist()
            ground_truth = overall_results["ground_truth"].tolist()
            probabilities = overall_results["probability"].tolist()
            confidence_weights = (overall_results["confidence"] / 5.0).tolist()

            metrics = self.metrics_calculator.compute_all_metrics(
                predictions=predictions,
                probabilities=probabilities,
                ground_truth=ground_truth,
                confidence_weights=confidence_weights,
                compute_ci=False,
            )

            tiers_summary.append(
                {
                    "tier": "overall",
                    "num_cases": len(overall_results),
                    "avg_confidence": overall_results["confidence"].mean(),
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "accuracy": metrics["accuracy"],
                    "brier_score": metrics["brier_score"],
                    "ece": metrics["ece"],
                }
            )

        # Export to CSV
        summary_df = pd.DataFrame(tiers_summary)
        summary_df.to_csv(output_path, index=False)
        print(f"[OK] Stratified metrics exported to {output_path}")

    def compare_configurations(
        self, results_list: List[pd.DataFrame], config_names: List[str]
    ) -> str:
        """
        Compare multiple detector configurations side-by-side.

        Args:
            results_list: List of results DataFrames (one per configuration)
            config_names: List of configuration names

        Returns:
            Comparison report
        """
        report = "=" * 80 + "\n"
        report += "CONFIGURATION COMPARISON\n"
        report += "=" * 80 + "\n\n"

        # Compute metrics for each configuration
        configs_metrics = []

        for results, config_name in zip(results_list, config_names):
            overall_results = results[results["tier"].isin(["gold", "silver"])]

            predictions = overall_results["prediction"].tolist()
            ground_truth = overall_results["ground_truth"].tolist()
            probabilities = overall_results["probability"].tolist()
            confidence_weights = (overall_results["confidence"] / 5.0).tolist()

            metrics = self.metrics_calculator.compute_all_metrics(
                predictions=predictions,
                probabilities=probabilities,
                ground_truth=ground_truth,
                confidence_weights=confidence_weights,
                compute_ci=False,
            )

            configs_metrics.append(
                {
                    "config": config_name,
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "brier": metrics["brier_score"],
                    "ece": metrics["ece"],
                }
            )

        # Format as table
        df = pd.DataFrame(configs_metrics)
        report += df.to_string(index=False)
        report += "\n\n"

        # Highlight best configuration
        best_f1_idx = df["f1"].argmax()
        report += f"Best F1: {config_names[best_f1_idx]} (F1={df.iloc[best_f1_idx]['f1']:.3f})\n"

        return report
