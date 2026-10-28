"""
Generate Gold/Silver benchmarks from actual raw lawyer annotation Excel files.

Reads 6 lawyer annotation files:
- Kenneth, Eileen, Kenny, Liza, Adi, John
- Each file: 15 questions x 2 variants = 30 annotations
- Note: Adi may have incomplete coverage

Aggregates by:
- Majority vote for hallucination labels
- Average confidence rating
- Agreement rate
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def load_single_annotator(filepath: Path, annotator_name: str) -> pd.DataFrame:
    """Load annotations from a single lawyer's Excel file."""
    # Read with header in row 1 (row 0 has annotator name)
    df = pd.read_excel(filepath, header=1)

    # Add annotator column
    df["Annotator"] = annotator_name

    # Filter to non-null Query_IDs
    df = df[df["Query_ID"].notna()].copy()

    # Ensure proper column names
    required_cols = [
        "Query_ID",
        "Variant",
        "Correctness",
        "Groundedness",
        "Hallucination_Label",
        "Confidence_Rating",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  Warning: {annotator_name} missing columns: {missing_cols}")

    return df[required_cols + ["Annotator"]]


def main() -> None:
    """Generate benchmarks from raw lawyer annotations."""
    print("=" * 80)
    print("GOLD/SILVER BENCHMARK GENERATOR (from raw annotations)")
    print("=" * 80)
    print()

    # Annotation files
    annotations_dir = Path("lawyer_data_package/raw_annotations")
    annotator_files = {
        "Kenneth": "round_1_annotation_A_Kenneth_Doherty.xlsx",
        "Eileen": "round_1_annotation_B_Eileen_Doherty.xlsx",
        "Kenny": "round_1_annotation_C_Kenny_Doherty.xlsx",
        "Liza": "round_1_annotation_D_Liza_Jameson.xlsx",
        "Adi": "round_1_annotation_E_Adi_Kennedy.xlsx",
        "John": "round_1_annotation_F_John_Jameson.xlsx",
    }

    # Load all annotators
    print("Loading annotations from 6 lawyers...")
    all_annotations = []

    for annotator_name, filename in annotator_files.items():
        filepath = annotations_dir / filename
        if not filepath.exists():
            print(f"  [SKIP] {annotator_name}: file not found")
            continue

        try:
            df = load_single_annotator(filepath, annotator_name)
            all_annotations.append(df)
            print(f"  [OK] {annotator_name}: {len(df)} annotations")
        except Exception as e:
            print(f"  [ERROR] {annotator_name}: {e}")

    if not all_annotations:
        print("\n[ERROR] No annotation files loaded!")
        return

    # Combine all annotations
    combined_df = pd.concat(all_annotations, ignore_index=True)
    print(f"\n[OK] Total annotations: {len(combined_df)}")
    print()

    # Aggregate by Query_ID and Variant
    print("Aggregating multi-annotator judgments...")
    grouped = combined_df.groupby(["Query_ID", "Variant"])

    aggregated_data = []

    for (query_id, variant), group in grouped:
        # Hallucination votes
        hall_votes = group["Hallucination_Label"].value_counts().to_dict()
        total_annotators = len(group)

        # Majority label
        if hall_votes:
            majority_label = max(hall_votes.items(), key=lambda x: x[1])[0]
            max_votes = hall_votes[majority_label]
            agreement_rate = max_votes / total_annotators
        else:
            majority_label = "Unknown"
            max_votes = 0
            agreement_rate = 0.0

        # Confidence statistics (convert to 1-5 scale if needed)
        conf_ratings = group["Confidence_Rating"].dropna()
        if len(conf_ratings) > 0:
            avg_confidence = conf_ratings.mean()
            min_confidence = conf_ratings.min()
            max_confidence = conf_ratings.max()
        else:
            avg_confidence = 0.0
            min_confidence = 0.0
            max_confidence = 0.0

        # Evaluability tier based on confidence
        if avg_confidence >= 4.0:
            tier = "gold"
        elif avg_confidence >= 2.5:
            tier = "silver"
        else:
            tier = "excluded"

        aggregated_data.append(
            {
                "Query_ID": int(query_id),
                "Variant": variant,
                "Majority_Label": majority_label,
                "Votes": hall_votes,
                "Total_Annotators": total_annotators,
                "Agreement_Rate": agreement_rate,
                "Avg_Confidence": avg_confidence,
                "Min_Confidence": min_confidence,
                "Max_Confidence": max_confidence,
                "Tier": tier,
            }
        )

    aggregated_df = pd.DataFrame(aggregated_data)
    print(f"[OK] Aggregated {len(aggregated_df)} question-variant pairs")
    print()

    # Show tier distribution
    print("Tier Distribution:")
    tier_counts = aggregated_df["Tier"].value_counts()
    for tier in ["gold", "silver", "excluded"]:
        if tier in tier_counts.index:
            count = tier_counts[tier]
            questions = sorted(
                aggregated_df[aggregated_df["Tier"] == tier]["Query_ID"].unique()
            )
            avg_conf = aggregated_df[aggregated_df["Tier"] == tier][
                "Avg_Confidence"
            ].mean()
            print(f"  {tier}: {count} entries (Questions: {questions}, Avg conf: {avg_conf:.2f})")
    print()

    # Export benchmarks
    output_dir = Path("data/gold_standard")
    output_dir.mkdir(parents=True, exist_ok=True)

    for tier in ["gold", "silver", "excluded"]:
        tier_df = aggregated_df[aggregated_df["Tier"] == tier]

        if len(tier_df) == 0:
            print(f"[SKIP] {tier} tier (no data)")
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
                "votes": row["Votes"],
                "total_annotators": int(row["Total_Annotators"]),
                "evaluability_tier": tier,
            }
            benchmark_data.append(entry)

        # Save to JSON
        output_path = output_dir / f"{tier}_benchmark.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, indent=2)

        print(f"[OK] Exported {len(benchmark_data)} {tier} annotations to {output_path}")

    # Save summary
    summary = {}
    for tier in ["gold", "silver", "excluded"]:
        tier_df = aggregated_df[aggregated_df["Tier"] == tier]
        summary[tier] = {
            "count": len(tier_df),
            "avg_confidence": float(tier_df["Avg_Confidence"].mean())
            if len(tier_df) > 0
            else 0.0,
            "questions": sorted(tier_df["Query_ID"].unique().tolist())
            if len(tier_df) > 0
            else [],
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
    print(
        f"Gold tier: {summary['gold']['count']} annotations (Avg conf: {summary['gold']['avg_confidence']:.2f})"
    )
    print(f"  Questions: {summary['gold']['questions']}")
    print()
    print(
        f"Silver tier: {summary['silver']['count']} annotations (Avg conf: {summary['silver']['avg_confidence']:.2f})"
    )
    print(f"  Questions: {summary['silver']['questions']}")
    print()
    print(
        f"Excluded tier: {summary['excluded']['count']} annotations (Avg conf: {summary['excluded']['avg_confidence']:.2f})"
    )
    print(f"  Questions: {summary['excluded']['questions']}")
    print()
    print("Next steps:")
    print("1. Review benchmarks in data/gold_standard/")
    print("2. Use Gold benchmark for threshold tuning")
    print("3. Use Silver benchmark for validation")


if __name__ == "__main__":
    main()
