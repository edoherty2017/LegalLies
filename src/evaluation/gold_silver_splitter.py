"""
Gold/Silver benchmark split generator based on evaluability confidence.

Creates stratified benchmark sets:
- Gold: Unanimous OR avg_confidence >= 4.0 (Q1, Q3 expected)
- Silver: Majority (3+/5) + avg_confidence >= 3.0 (Q2, Q4-Q11, Q14-Q15 expected)
- Excluded: avg_confidence < 2.5 (Q12, Q13 - too research-intensive)

These splits enable:
- Threshold tuning on Gold set (high-confidence, reliable labels)
- Validation on Silver set (medium-confidence, stratified metrics)
- Avoiding contamination from low-confidence unreliable labels
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class GoldSilverSplitter:
    """
    Splits lawyer annotations into Gold, Silver, and Excluded tiers.

    Uses confidence stratification to ensure detector evaluation reflects
    the heterogeneous, medium-confidence nature of human legal evaluation.
    """

    def __init__(
        self,
        gold_confidence_threshold: float = 4.0,
        silver_confidence_threshold: float = 3.0,
        excluded_confidence_threshold: float = 2.5,
        gold_agreement_threshold: float = 0.8,  # 4/5 or 5/5 agreement
        silver_agreement_threshold: float = 0.6,  # 3/5 agreement
    ) -> None:
        """
        Initialize splitter with confidence and agreement thresholds.

        Args:
            gold_confidence_threshold: Min avg confidence for Gold tier (default: 4.0)
            silver_confidence_threshold: Min avg confidence for Silver tier (default: 3.0)
            excluded_confidence_threshold: Max avg confidence for Excluded tier (default: 2.5)
            gold_agreement_threshold: Min agreement rate for Gold tier (default: 0.8 = 4/5)
            silver_agreement_threshold: Min agreement rate for Silver tier (default: 0.6 = 3/5)
        """
        self.gold_confidence_threshold = gold_confidence_threshold
        self.silver_confidence_threshold = silver_confidence_threshold
        self.excluded_confidence_threshold = excluded_confidence_threshold
        self.gold_agreement_threshold = gold_agreement_threshold
        self.silver_agreement_threshold = silver_agreement_threshold

    def create_gold_split(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """
        Create Gold benchmark split.

        Criteria:
        - Unanimous OR near-unanimous (agreement >= 0.8, i.e., 4+/5 lawyers agree)
        - Average confidence >= 4.0
        - Expected questions: Q1, Q3

        Args:
            annotations: Aggregated annotations DataFrame

        Returns:
            DataFrame with Gold tier annotations
        """
        # Filter by confidence and agreement
        gold_df = annotations[
            (annotations["Avg_Confidence"] >= self.gold_confidence_threshold)
            & (annotations["Agreement_Rate"] >= self.gold_agreement_threshold)
        ].copy()

        gold_df["Tier"] = "gold"

        return gold_df

    def create_silver_split(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """
        Create Silver benchmark split.

        Criteria:
        - Majority agreement (agreement >= 0.6, i.e., 3+/5 lawyers agree)
        - Average confidence >= 3.0 (but < 4.0 to avoid overlap with Gold)
        - Expected questions: Q2, Q4-Q11, Q14-Q15

        Args:
            annotations: Aggregated annotations DataFrame

        Returns:
            DataFrame with Silver tier annotations
        """
        # Filter by confidence and agreement (exclude Gold tier)
        silver_df = annotations[
            (annotations["Avg_Confidence"] >= self.silver_confidence_threshold)
            & (annotations["Avg_Confidence"] < self.gold_confidence_threshold)
            & (annotations["Agreement_Rate"] >= self.silver_agreement_threshold)
        ].copy()

        silver_df["Tier"] = "silver"

        return silver_df

    def exclude_low_confidence(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """
        Identify low-confidence questions to exclude from benchmarks.

        Criteria:
        - Average confidence < 2.5
        - Expected questions: Q12, Q13 (too research-intensive)

        Args:
            annotations: Aggregated annotations DataFrame

        Returns:
            DataFrame with Excluded tier annotations
        """
        excluded_df = annotations[
            annotations["Avg_Confidence"] < self.excluded_confidence_threshold
        ].copy()

        excluded_df["Tier"] = "excluded"

        return excluded_df

    def create_all_splits(
        self, annotations: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Create all splits (Gold, Silver, Excluded) in one call.

        Args:
            annotations: Aggregated annotations DataFrame

        Returns:
            Dictionary with keys "gold", "silver", "excluded" and corresponding DataFrames
        """
        gold = self.create_gold_split(annotations)
        silver = self.create_silver_split(annotations)
        excluded = self.exclude_low_confidence(annotations)

        # Also create "all" split with tiers marked
        all_with_tiers = pd.concat([gold, silver, excluded], ignore_index=True)

        return {
            "gold": gold,
            "silver": silver,
            "excluded": excluded,
            "all": all_with_tiers,
        }

    def generate_split_report(
        self, splits: Dict[str, pd.DataFrame]
    ) -> str:
        """
        Generate human-readable report of split statistics.

        Args:
            splits: Dictionary of splits from create_all_splits()

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("GOLD/SILVER SPLIT REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall statistics
        total = len(splits["all"])
        report.append(f"Total Annotations: {total}")
        report.append("")

        # Per-tier statistics
        for tier in ["gold", "silver", "excluded"]:
            df = splits[tier]
            if len(df) == 0:
                continue

            report.append(f"--- {tier.upper()} TIER ---")
            report.append(f"Count: {len(df)} ({len(df)/total*100:.1f}%)")
            report.append(
                f"Avg Confidence: {df['Avg_Confidence'].mean():.2f} (range: {df['Avg_Confidence'].min():.2f}-{df['Avg_Confidence'].max():.2f})"
            )
            report.append(
                f"Avg Agreement: {df['Agreement_Rate'].mean():.2%} (range: {df['Agreement_Rate'].min():.2%}-{df['Agreement_Rate'].max():.2%})"
            )

            # Show question IDs in this tier
            question_ids = df["Query_ID"].unique().tolist()
            report.append(f"Questions: {', '.join(map(str, sorted(question_ids)))}")

            # Hallucination distribution
            hall_dist = df["Majority_Label"].value_counts().to_dict()
            report.append(
                f"Hallucination Distribution: Yes={hall_dist.get('Yes', 0)}, No={hall_dist.get('No', 0)}"
            )

            report.append("")

        # Recommendations
        report.append("--- RECOMMENDATIONS ---")
        report.append(
            f"- Use GOLD tier ({len(splits['gold'])} items) for threshold tuning and calibration"
        )
        report.append(
            f"- Use SILVER tier ({len(splits['silver'])} items) for validation with confidence weighting"
        )
        report.append(
            f"- EXCLUDE {len(splits['excluded'])} low-confidence items from headline metrics"
        )

        return "\n".join(report)

    def export_splits_to_json(
        self,
        splits: Dict[str, pd.DataFrame],
        output_dir: Path,
        include_excluded: bool = False,
    ) -> None:
        """
        Export splits to JSON format for benchmarking.

        Args:
            splits: Dictionary of splits from create_all_splits()
            output_dir: Directory to save JSON files
            include_excluded: If True, also export excluded tier
        """
        import json

        output_dir.mkdir(parents=True, exist_ok=True)

        tiers_to_export = ["gold", "silver"]
        if include_excluded:
            tiers_to_export.append("excluded")

        for tier in tiers_to_export:
            df = splits[tier]
            if len(df) == 0:
                print(f"Skipping {tier} tier (no data)")
                continue

            # Convert to benchmark format
            benchmark_data = []

            for _, row in df.iterrows():
                entry = {
                    "query_id": row["Query_ID"],
                    "variant": row["Variant"],
                    "gold_label": row["Majority_Label"],
                    "confidence": float(row["Avg_Confidence"]),
                    "agreement_rate": float(row["Agreement_Rate"]),
                    "votes": {"Yes": int(row["Votes_Yes"]), "No": int(row["Votes_No"])},
                    "evaluability_tier": tier,
                    "total_annotators": int(row["Total_Annotators"]),
                }

                # Add weighted hallucination if available
                if "Weighted_Hallucination_Avg" in row and pd.notna(
                    row["Weighted_Hallucination_Avg"]
                ):
                    entry["weighted_hallucination_avg"] = float(
                        row["Weighted_Hallucination_Avg"]
                    )

                benchmark_data.append(entry)

            # Save to JSON
            output_path = output_dir / f"{tier}_benchmark.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(benchmark_data, f, indent=2)

            print(f"Exported {len(benchmark_data)} {tier} annotations to {output_path}")

        # Also save summary statistics
        summary_path = output_dir / "split_summary.json"
        summary = {
            "gold": {
                "count": len(splits["gold"]),
                "avg_confidence": float(splits["gold"]["Avg_Confidence"].mean())
                if len(splits["gold"]) > 0
                else 0,
                "questions": splits["gold"]["Query_ID"].unique().tolist()
                if len(splits["gold"]) > 0
                else [],
            },
            "silver": {
                "count": len(splits["silver"]),
                "avg_confidence": float(splits["silver"]["Avg_Confidence"].mean())
                if len(splits["silver"]) > 0
                else 0,
                "questions": splits["silver"]["Query_ID"].unique().tolist()
                if len(splits["silver"]) > 0
                else [],
            },
            "excluded": {
                "count": len(splits["excluded"]),
                "avg_confidence": float(splits["excluded"]["Avg_Confidence"].mean())
                if len(splits["excluded"]) > 0
                else 0,
                "questions": splits["excluded"]["Query_ID"].unique().tolist()
                if len(splits["excluded"]) > 0
                else [],
            },
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved split summary to {summary_path}")
