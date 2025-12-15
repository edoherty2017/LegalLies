"""
Annotation loader for lawyer-annotated hallucination data.

Loads and processes annotations from Excel files (Annotation_Round1_AB.xlsx,
Annotation_Round2_C.xlsx) and derives hallucination labels based on the formula:

    Hallucination = Incorrect AND (Ungrounded OR Misgrounded)

Supports confidence-weighted label computation for evaluability-aware metrics.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class AnnotationLoader:
    """
    Loads and processes lawyer annotations from Excel files.

    Handles:
    - Loading annotations from multiple rounds (Round1_AB, Round2_C)
    - Deriving hallucination labels from correctness + groundedness
    - Computing confidence weights per annotator
    - Aggregating multi-annotator judgments
    """

    def __init__(self, data_dir: Path = Path("lawyer_data_package/raw_annotations")) -> None:
        """
        Initialize loader with path to annotation data.

        Args:
            data_dir: Directory containing annotation Excel files
        """
        self.data_dir = data_dir

    def load_annotations(
        self, filepath: str, derive_hallucination: bool = True
    ) -> pd.DataFrame:
        """
        Load annotations from Excel file and optionally derive hallucination labels.

        Args:
            filepath: Path to annotation Excel file
            derive_hallucination: If True, compute hallucination labels from
                                 correctness + groundedness dimensions

        Returns:
            DataFrame with annotations
        """
        # Load Excel file
        df = pd.read_excel(filepath)

        # Derive hallucination labels if requested
        if derive_hallucination:
            df = self.derive_hallucination_labels(df)

        return df

    def derive_hallucination_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive hallucination labels based on correctness and groundedness.

        Formula: Hallucination = Incorrect AND (Ungrounded OR Misgrounded)

        Args:
            df: DataFrame with Correctness and Groundedness columns

        Returns:
            DataFrame with added Hallucination_Label column
        """
        # Ensure required columns exist
        required_cols = ["Correctness", "Groundedness"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Derive hallucination label
        def _is_hallucination(row: pd.Series) -> str:
            correctness = row["Correctness"]
            groundedness = row["Groundedness"]

            # Hallucination: Incorrect AND (Ungrounded OR Misgrounded)
            is_incorrect = correctness == "Incorrect"
            is_ungrounded = groundedness in ["Ungrounded", "Misgrounded"]

            if is_incorrect and is_ungrounded:
                return "Yes"
            else:
                return "No"

        df["Hallucination_Label"] = df.apply(_is_hallucination, axis=1)

        return df

    def compute_confidence_weights(
        self, df: pd.DataFrame, confidence_col: str = "Confidence"
    ) -> pd.DataFrame:
        """
        Compute confidence-weighted labels for each annotation.

        Weight formula: weighted_label = (confidence / 5.0) x rating

        Args:
            df: DataFrame with annotations
            confidence_col: Name of confidence column (default: "Confidence")

        Returns:
            DataFrame with added confidence weight columns
        """
        if confidence_col not in df.columns:
            raise ValueError(f"Confidence column '{confidence_col}' not found")

        # Normalize confidence to 0-1 scale (assuming 1-5 scale)
        df["Confidence_Weight"] = df[confidence_col] / 5.0

        # Compute weighted hallucination label (for binary: 1=Yes, 0=No)
        df["Weighted_Hallucination"] = (
            df["Hallucination_Label"].map({"Yes": 1, "No": 0}) * df["Confidence_Weight"]
        )

        return df

    def aggregate_multi_annotator(
        self,
        df: pd.DataFrame,
        group_by: List[str] = ["Query_ID", "Variant"],
        confidence_weighted: bool = True,
    ) -> pd.DataFrame:
        """
        Aggregate annotations from multiple lawyers per question.

        Args:
            df: DataFrame with annotations
            group_by: Columns to group by (default: Query_ID, Variant)
            confidence_weighted: If True, use confidence-weighted aggregation

        Returns:
            Aggregated DataFrame with majority vote and confidence stats
        """
        # Ensure confidence weights exist
        if confidence_weighted and "Confidence_Weight" not in df.columns:
            df = self.compute_confidence_weights(df)

        # Group by question and variant
        grouped = df.groupby(group_by)

        # Aggregate per group
        aggregated_data = []

        for name, group in grouped:
            query_id, variant = name

            # Count votes
            hallucination_votes = group["Hallucination_Label"].value_counts().to_dict()
            num_yes = hallucination_votes.get("Yes", 0)
            num_no = hallucination_votes.get("No", 0)

            # Majority vote (unweighted)
            majority_label = "Yes" if num_yes > num_no else "No"

            # Confidence-weighted average (if using weights)
            if confidence_weighted:
                weighted_avg = group["Weighted_Hallucination"].mean()
            else:
                weighted_avg = None

            # Confidence statistics
            avg_confidence = group["Confidence"].mean()
            min_confidence = group["Confidence"].min()
            max_confidence = group["Confidence"].max()

            # Agreement metrics
            total_annotators = len(group)
            agreement_rate = max(num_yes, num_no) / total_annotators

            aggregated_data.append(
                {
                    "Query_ID": query_id,
                    "Variant": variant,
                    "Majority_Label": majority_label,
                    "Votes_Yes": num_yes,
                    "Votes_No": num_no,
                    "Total_Annotators": total_annotators,
                    "Agreement_Rate": agreement_rate,
                    "Weighted_Hallucination_Avg": weighted_avg,
                    "Avg_Confidence": avg_confidence,
                    "Min_Confidence": min_confidence,
                    "Max_Confidence": max_confidence,
                }
            )

        return pd.DataFrame(aggregated_data)

    def load_all_rounds(
        self, derive_hallucination: bool = True, aggregate: bool = True
    ) -> pd.DataFrame:
        """
        Load annotations from all rounds and optionally aggregate.

        Args:
            derive_hallucination: If True, derive hallucination labels
            aggregate: If True, aggregate multi-annotator judgments

        Returns:
            Combined DataFrame from all rounds
        """
        dfs = []

        # Load Round 1 (Baseline + RAG variants)
        round1_path = self.data_dir / "Annotation_Round1_AB.xlsx"
        if round1_path.exists():
            df1 = self.load_annotations(str(round1_path), derive_hallucination)
            dfs.append(df1)

        # Load Round 2 (RAG+Steering variant)
        round2_path = self.data_dir / "Annotation_Round2_C.xlsx"
        if round2_path.exists():
            df2 = self.load_annotations(str(round2_path), derive_hallucination)
            dfs.append(df2)

        # Combine all rounds
        if not dfs:
            raise FileNotFoundError(f"No annotation files found in {self.data_dir}")

        combined_df = pd.concat(dfs, ignore_index=True)

        # Aggregate if requested
        if aggregate:
            combined_df = self.aggregate_multi_annotator(
                combined_df, confidence_weighted=True
            )

        return combined_df

    def get_evaluability_tier(self, avg_confidence: float) -> str:
        """
        Classify question into evaluability tier based on average confidence.

        Tiers:
        - Gold: avg_confidence >= 4.0 (high confidence, unanimous/near-unanimous)
        - Silver: 2.5 <= avg_confidence < 4.0 (medium confidence, majority agreement)
        - Excluded: avg_confidence < 2.5 (low confidence, research-intensive)

        Args:
            avg_confidence: Average confidence rating (1-5 scale)

        Returns:
            Evaluability tier: "gold", "silver", or "excluded"
        """
        if avg_confidence >= 4.0:
            return "gold"
        elif avg_confidence >= 2.5:
            return "silver"
        else:
            return "excluded"

    def add_evaluability_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add evaluability tier classification to aggregated annotations.

        Args:
            df: Aggregated DataFrame (must have Avg_Confidence column)

        Returns:
            DataFrame with added Evaluability_Tier column
        """
        if "Avg_Confidence" not in df.columns:
            raise ValueError("DataFrame must have Avg_Confidence column (aggregate first)")

        df["Evaluability_Tier"] = df["Avg_Confidence"].apply(self.get_evaluability_tier)

        return df

    def export_for_benchmark(
        self,
        df: pd.DataFrame,
        output_path: Path,
        tier_filter: Optional[str] = None,
    ) -> None:
        """
        Export annotations to JSON format for benchmarking.

        Args:
            df: Aggregated annotations DataFrame
            output_path: Path to save JSON file
            tier_filter: Optional tier to filter by ("gold", "silver", or None for all)
        """
        if tier_filter:
            if "Evaluability_Tier" not in df.columns:
                df = self.add_evaluability_tiers(df)
            df = df[df["Evaluability_Tier"] == tier_filter]

        # Convert to benchmark format
        benchmark_data = []

        for _, row in df.iterrows():
            entry = {
                "query_id": row["Query_ID"],
                "variant": row["Variant"],
                "gold_label": row["Majority_Label"],
                "confidence": row["Avg_Confidence"],
                "agreement_rate": row["Agreement_Rate"],
                "votes": {"Yes": int(row["Votes_Yes"]), "No": int(row["Votes_No"])},
                "evaluability_tier": row.get("Evaluability_Tier", "unknown"),
                "weighted_hallucination_avg": row.get("Weighted_Hallucination_Avg"),
            }
            benchmark_data.append(entry)

        # Save to JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, indent=2)

        print(f"Exported {len(benchmark_data)} annotations to {output_path}")
