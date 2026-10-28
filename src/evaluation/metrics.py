"""
Confidence-weighted metrics for evaluating hallucination detector.

Implements metrics that account for varying lawyer confidence levels:
- Weighted Precision/Recall/F1
- Brier Score (calibration quality)
- Expected Calibration Error (ECE)
- Bootstrap confidence intervals

Follows guide Section 4 for evaluation strategy.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ConfidenceWeightedMetrics:
    """
    Compute evaluation metrics with confidence weighting.

    Lawyer annotations have varying confidence (1.53 - 4.20).
    High-confidence annotations should be weighted more heavily.
    """

    def compute_precision_recall_f1(
        self,
        predictions: List[int],
        ground_truth: List[int],
        confidence_weights: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Compute weighted precision, recall, and F1 score.

        Args:
            predictions: Binary predictions (0 = no hallucination, 1 = hallucination)
            ground_truth: Binary ground truth labels
            confidence_weights: Confidence weights (0-1), default: uniform weights

        Returns:
            Dictionary with precision, recall, f1, accuracy
        """
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)

        if confidence_weights is None:
            confidence_weights = np.ones(len(predictions))
        else:
            confidence_weights = np.array(confidence_weights)

        # Normalize weights to sum to 1
        confidence_weights = confidence_weights / confidence_weights.sum()

        # Weighted TP, FP, FN, TN
        tp = np.sum(
            confidence_weights[(predictions == 1) & (ground_truth == 1)]
        )
        fp = np.sum(
            confidence_weights[(predictions == 1) & (ground_truth == 0)]
        )
        fn = np.sum(
            confidence_weights[(predictions == 0) & (ground_truth == 1)]
        )
        tn = np.sum(
            confidence_weights[(predictions == 0) & (ground_truth == 0)]
        )

        # Weighted metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def compute_brier_score(
        self,
        probabilities: List[float],
        ground_truth: List[int],
        confidence_weights: Optional[List[float]] = None,
    ) -> float:
        """
        Compute Brier score (mean squared error of probabilities).

        Lower is better. Perfect predictions have Brier score = 0.

        Args:
            probabilities: Predicted probabilities (0-1)
            ground_truth: Binary ground truth labels
            confidence_weights: Confidence weights (0-1)

        Returns:
            Brier score (0-1)
        """
        probabilities = np.array(probabilities)
        ground_truth = np.array(ground_truth)

        if confidence_weights is None:
            # Unweighted Brier score
            return brier_score_loss(ground_truth, probabilities)

        confidence_weights = np.array(confidence_weights)
        confidence_weights = confidence_weights / confidence_weights.sum()

        # Weighted Brier score
        squared_errors = (probabilities - ground_truth) ** 2
        weighted_brier = np.sum(confidence_weights * squared_errors)

        return float(weighted_brier)

    def compute_ece(
        self,
        probabilities: List[float],
        ground_truth: List[int],
        num_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE measures how well predicted probabilities match actual outcomes.
        Lower is better. Well-calibrated models have ECE ~ 0.

        Args:
            probabilities: Predicted probabilities (0-1)
            ground_truth: Binary ground truth labels
            num_bins: Number of bins for calibration curve

        Returns:
            ECE value (0-1)
        """
        probabilities = np.array(probabilities)
        ground_truth = np.array(ground_truth)

        if len(probabilities) == 0:
            return 0.0

        # Use sklearn's calibration_curve
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore empty bin warnings
            try:
                prob_true, prob_pred = calibration_curve(
                    ground_truth, probabilities, n_bins=num_bins, strategy="uniform"
                )
            except ValueError:
                # Not enough data for calibration
                return 0.0

        # Compute ECE as weighted average of |accuracy - confidence| per bin
        bin_edges = np.linspace(0, 1, num_bins + 1)
        ece = 0.0

        for i in range(len(prob_pred)):
            # Find samples in this bin
            bin_mask = (probabilities >= bin_edges[i]) & (
                probabilities < bin_edges[i + 1]
            )
            if i == len(prob_pred) - 1:  # Last bin includes upper edge
                bin_mask = (probabilities >= bin_edges[i]) & (
                    probabilities <= bin_edges[i + 1]
                )

            bin_count = np.sum(bin_mask)
            if bin_count == 0:
                continue

            # |accuracy - confidence| for this bin
            bin_accuracy = prob_true[i]
            bin_confidence = prob_pred[i]
            bin_weight = bin_count / len(probabilities)

            ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return float(ece)

    def bootstrap_confidence_intervals(
        self,
        predictions: List[int],
        ground_truth: List[int],
        confidence_weights: Optional[List[float]] = None,
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Compute bootstrap confidence intervals for metrics.

        Args:
            predictions: Binary predictions
            ground_truth: Binary ground truth labels
            confidence_weights: Confidence weights (0-1)
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            Dictionary mapping metric name to (mean, lower_ci, upper_ci)
        """
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)

        if confidence_weights is None:
            confidence_weights = np.ones(len(predictions))
        else:
            confidence_weights = np.array(confidence_weights)

        n_samples = len(predictions)
        bootstrap_metrics = {
            "precision": [],
            "recall": [],
            "f1": [],
            "accuracy": [],
        }

        # Bootstrap resampling
        rng = np.random.default_rng(seed=42)
        for _ in range(n_iterations):
            # Sample with replacement
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            boot_pred = predictions[indices]
            boot_gt = ground_truth[indices]
            boot_weights = confidence_weights[indices]

            # Compute metrics for this bootstrap sample
            metrics = self.compute_precision_recall_f1(
                boot_pred.tolist(), boot_gt.tolist(), boot_weights.tolist()
            )

            bootstrap_metrics["precision"].append(metrics["precision"])
            bootstrap_metrics["recall"].append(metrics["recall"])
            bootstrap_metrics["f1"].append(metrics["f1"])
            bootstrap_metrics["accuracy"].append(metrics["accuracy"])

        # Compute confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        results = {}
        for metric_name, values in bootstrap_metrics.items():
            values = np.array(values)
            mean = np.mean(values)
            lower_ci = np.percentile(values, lower_percentile)
            upper_ci = np.percentile(values, upper_percentile)
            results[metric_name] = (mean, lower_ci, upper_ci)

        return results

    def compute_confusion_matrix(
        self, predictions: List[int], ground_truth: List[int]
    ) -> np.ndarray:
        """
        Compute confusion matrix.

        Args:
            predictions: Binary predictions
            ground_truth: Binary ground truth labels

        Returns:
            2x2 confusion matrix [[TN, FP], [FN, TP]]
        """
        return confusion_matrix(ground_truth, predictions)

    def compute_all_metrics(
        self,
        predictions: List[int],
        probabilities: List[float],
        ground_truth: List[int],
        confidence_weights: Optional[List[float]] = None,
        compute_ci: bool = True,
        num_bins: int = 10,
    ) -> Dict:
        """
        Compute all evaluation metrics.

        Args:
            predictions: Binary predictions
            probabilities: Predicted probabilities (0-1)
            ground_truth: Binary ground truth labels
            confidence_weights: Confidence weights (0-1)
            compute_ci: Compute bootstrap confidence intervals
            num_bins: Number of bins for ECE

        Returns:
            Dictionary with all metrics
        """
        # Basic metrics
        basic_metrics = self.compute_precision_recall_f1(
            predictions, ground_truth, confidence_weights
        )

        # Brier score
        brier = self.compute_brier_score(probabilities, ground_truth, confidence_weights)

        # ECE
        ece = self.compute_ece(probabilities, ground_truth, num_bins)

        # Confusion matrix
        cm = self.compute_confusion_matrix(predictions, ground_truth)

        results = {
            **basic_metrics,
            "brier_score": brier,
            "ece": ece,
            "confusion_matrix": cm.tolist(),
        }

        # Bootstrap confidence intervals (optional, computationally expensive)
        if compute_ci:
            ci_results = self.bootstrap_confidence_intervals(
                predictions, ground_truth, confidence_weights
            )
            results["confidence_intervals"] = ci_results

        return results

    def format_metrics_report(self, metrics: Dict) -> str:
        """
        Format metrics as human-readable report.

        Args:
            metrics: Dictionary from compute_all_metrics()

        Returns:
            Formatted report string
        """
        report = "Evaluation Metrics\n"
        report += "==================\n\n"

        # Basic metrics
        report += f"Precision: {metrics['precision']:.3f}\n"
        report += f"Recall:    {metrics['recall']:.3f}\n"
        report += f"F1 Score:  {metrics['f1']:.3f}\n"
        report += f"Accuracy:  {metrics['accuracy']:.3f}\n\n"

        # Calibration metrics
        report += f"Brier Score: {metrics['brier_score']:.3f}\n"
        report += f"ECE:         {metrics['ece']:.3f}\n\n"

        # Confusion matrix
        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            report += "Confusion Matrix:\n"
            report += f"  TN={cm[0,0]:.1f}  FP={cm[0,1]:.1f}\n"
            report += f"  FN={cm[1,0]:.1f}  TP={cm[1,1]:.1f}\n\n"

        # Confidence intervals
        if "confidence_intervals" in metrics:
            report += "95% Confidence Intervals:\n"
            ci = metrics["confidence_intervals"]
            for metric_name, (mean, lower, upper) in ci.items():
                report += f"  {metric_name.capitalize()}: {mean:.3f} [{lower:.3f}, {upper:.3f}]\n"

        return report
