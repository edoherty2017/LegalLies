"""
Generate Gold/Silver benchmarks from existing analysis outputs.

Uses the pre-computed analysis files instead of re-processing raw annotations:
- evaluability_assessment.csv: Confidence and evaluability tiers
- disagreements.csv: Individual lawyer votes on hallucination labels

This is more reliable than re-parsing stub Excel files.
"""

import ast
import json
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_rating_array(rating_str: str) -> list:
    """Parse string representation of Python list into actual list."""
    try:
        return ast.literal_eval(rating_str)
    except:
        return []


def compute_majority_vote(ratings: list) -> tuple[str, dict]:
    """
    Compute majority vote from list of ratings.

    Returns:
        Tuple of (majority_label, vote_counts)
    """
    from collections import Counter

    counter = Counter(ratings)
    majority_label = counter.most_common(1)[0][0] if counter else "Unknown"
    vote_counts = dict(counter)

    return majority_label, vote_counts


def main() -> None:
    """Generate benchmarks from analysis outputs."""
    print("=" * 80)
    print("GOLD/SILVER BENCHMARK GENERATOR (from analysis outputs)")
    print("=" * 80)
    print()

    # Paths
    analysis_dir = Path("lawyer_data_package/analysis_outputs")
    output_dir = Path("data/gold_standard")

    # Load evaluability assessment (has confidence data)
    eval_path = analysis_dir / "evaluability_assessment.csv"
    print(f"Loading evaluability data from: {eval_path}")
    eval_df = pd.read_csv(eval_path)
    print(f"[OK] Loaded {len(eval_df)} evaluability assessments")
    print()

    # Load disagreements (has actual votes)
    disagree_path = analysis_dir / "disagreements.csv"
    print(f"Loading disagreement data from: {disagree_path}")
    disagree_df = pd.read_csv(disagree_path)

    # Filter to Hallucination_Label dimension only
    hall_df = disagree_df[disagree_df["Dimension"] == "Hallucination_Label"].copy()
    print(f"[OK] Loaded {len(hall_df)} hallucination label sets")
    print()

    # Parse ratings arrays and compute majority votes
    print("Computing majority votes...")
    hall_df["Ratings_List"] = hall_df["Ratings"].apply(parse_rating_array)
    hall_df["Majority_Label"] = hall_df["Ratings_List"].apply(
        lambda x: compute_majority_vote(x)[0]
    )
    hall_df["Vote_Counts"] = hall_df["Ratings_List"].apply(
        lambda x: compute_majority_vote(x)[1]
    )

    # Compute agreement rate (max votes / total votes)
    hall_df["Agreement_Rate"] = hall_df.apply(
        lambda row: max(row["Vote_Counts"].values()) / row["Num_Raters"]
        if row["Vote_Counts"]
        else 0.0,
        axis=1,
    )

    print("[OK] Majority votes computed")
    print()

    # Merge with evaluability data
    print("Merging with evaluability data...")
    merged_df = hall_df.merge(
        eval_df, on=["Query_ID", "Variant"], how="inner", suffixes=("", "_eval")
    )
    print(f"[OK] Merged data: {len(merged_df)} entries")
    print()

    # Map evaluability tier to our Gold/Silver/Excluded categories
    def map_tier(tier: str) -> str:
        if tier == "high_confidence_evaluable":
            return "gold"
        elif tier == "medium_confidence_evaluable":
            return "silver"
        else:  # low_confidence_evaluable
            return "excluded"

    merged_df["Tier"] = merged_df["Evaluability_Tier"].apply(map_tier)

    # Show distribution
    print("Tier Distribution:")
    tier_counts = merged_df["Tier"].value_counts()
    for tier, count in tier_counts.items():
        questions = merged_df[merged_df["Tier"] == tier]["Query_ID"].unique()
        print(f"  {tier}: {count} entries (Questions: {sorted(questions)})")
    print()

    # Export each tier to JSON
    output_dir.mkdir(parents=True, exist_ok=True)

    for tier in ["gold", "silver", "excluded"]:
        tier_df = merged_df[merged_df["Tier"] == tier]

        if len(tier_df) == 0:
            print(f"Skipping {tier} tier (no data)")
            continue

        # Convert to benchmark format
        benchmark_data = []

        for _, row in tier_df.iterrows():
            entry = {
                "query_id": str(row["Query_ID"]),
                "variant": row["Variant"],
                "gold_label": row["Majority_Label"],
                "confidence": float(row["Avg_Confidence"]),
                "agreement_rate": float(row["Agreement_Rate"]),
                "votes": row["Vote_Counts"],
                "total_annotators": int(row["Num_Raters"]),
                "evaluability_tier": tier,
            }
            benchmark_data.append(entry)

        # Save to JSON
        output_path = output_dir / f"{tier}_benchmark.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, indent=2)

        print(f"[OK] Exported {len(benchmark_data)} {tier} annotations to {output_path}")

    # Save summary
    summary = {
        "gold": {
            "count": len(merged_df[merged_df["Tier"] == "gold"]),
            "avg_confidence": float(
                merged_df[merged_df["Tier"] == "gold"]["Avg_Confidence"].mean()
            )
            if len(merged_df[merged_df["Tier"] == "gold"]) > 0
            else 0,
            "questions": sorted(
                merged_df[merged_df["Tier"] == "gold"]["Query_ID"].unique().tolist()
            ),
        },
        "silver": {
            "count": len(merged_df[merged_df["Tier"] == "silver"]),
            "avg_confidence": float(
                merged_df[merged_df["Tier"] == "silver"]["Avg_Confidence"].mean()
            )
            if len(merged_df[merged_df["Tier"] == "silver"]) > 0
            else 0,
            "questions": sorted(
                merged_df[merged_df["Tier"] == "silver"]["Query_ID"].unique().tolist()
            ),
        },
        "excluded": {
            "count": len(merged_df[merged_df["Tier"] == "excluded"]),
            "avg_confidence": float(
                merged_df[merged_df["Tier"] == "excluded"]["Avg_Confidence"].mean()
            )
            if len(merged_df[merged_df["Tier"] == "excluded"]) > 0
            else 0,
            "questions": sorted(
                merged_df[merged_df["Tier"] == "excluded"]["Query_ID"].unique().tolist()
            ),
        },
    }

    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Saved split summary to {summary_path}")
    print()

    print("=" * 80)
    print("BENCHMARK GENERATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Gold tier: {summary['gold']['count']} annotations (Avg conf: {summary['gold']['avg_confidence']:.2f})")
    print(f"  Questions: {summary['gold']['questions']}")
    print()
    print(f"Silver tier: {summary['silver']['count']} annotations (Avg conf: {summary['silver']['avg_confidence']:.2f})")
    print(f"  Questions: {summary['silver']['questions']}")
    print()
    print(f"Excluded tier: {summary['excluded']['count']} annotations (Avg conf: {summary['excluded']['avg_confidence']:.2f})")
    print(f"  Questions: {summary['excluded']['questions']}")
    print()
    print("Next steps:")
    print("1. Review benchmarks in data/gold_standard/")
    print("2. Use Gold benchmark for threshold tuning")
    print("3. Use Silver benchmark for validation")


if __name__ == "__main__":
    main()
