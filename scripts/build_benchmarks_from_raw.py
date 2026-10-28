"""
Build Gold and Silver benchmark datasets from raw lawyer annotations.

Uses the source-of-truth raw annotation Excel files from individual lawyers.
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_all_annotator_files(raw_dir: Path) -> pd.DataFrame:
    """Load and consolidate all individual annotator Excel files."""
    all_annotations = []

    annotator_files = list(raw_dir.glob("round_1_annotation_*.xlsx"))
    print(f"Found {len(annotator_files)} annotator files")

    for file in annotator_files:
        # Extract annotator name from filename
        annotator_name = file.stem.replace("round_1_annotation_", "").replace("_", " ")

        # Read with row 1 as header (row 0 contains column descriptions)
        df = pd.read_excel(file, header=1)

        # Add annotator name column
        df['Annotator'] = annotator_name

        all_annotations.append(df)
        print(f"  Loaded {len(df)} annotations from {annotator_name}")

    # Consolidate all annotations
    consolidated = pd.concat(all_annotations, ignore_index=True)
    print(f"\nTotal annotations: {len(consolidated)}")

    return consolidated


def compute_majority_vote(annotations: pd.DataFrame) -> Dict:
    """Compute majority vote for hallucination label."""
    hall_labels = annotations['Hallucination_Label'].dropna().tolist()

    yes_count = hall_labels.count('Yes')
    no_count = hall_labels.count('No')
    total = len(hall_labels)

    if yes_count > no_count:
        majority = "Hallucination"
        agreement_rate = yes_count / total
    elif no_count > yes_count:
        majority = "No_Hallucination"
        agreement_rate = no_count / total
    else:
        majority = "Tie"
        agreement_rate = 0.5

    return {
        "majority_label": majority,
        "agreement_rate": agreement_rate,
        "votes": {"Yes": yes_count, "No": no_count},
        "total_annotators": total
    }


def build_benchmark_from_annotations(
    annotations_df: pd.DataFrame,
    evaluability_df: pd.DataFrame,
    tier: str
) -> List[Dict]:
    """Build benchmark cases for a specific evaluability tier."""
    cases = []

    # Filter by tier
    tier_queries = evaluability_df[evaluability_df["Evaluability_Tier"] == tier]

    for _, tier_row in tier_queries.iterrows():
        query_id = int(tier_row["Query_ID"])
        variant = tier_row["Variant"]
        avg_confidence = float(tier_row["Avg_Confidence"])

        # Get all annotations for this query + variant
        query_annotations = annotations_df[
            (annotations_df["Query_ID"] == query_id) &
            (annotations_df["Variant"] == variant)
        ]

        if query_annotations.empty:
            print(f"WARNING: No annotations for Q{query_id} {variant}")
            continue

        # Extract query text and response (should be same across annotators)
        first_anno = query_annotations.iloc[0]
        query_text = str(first_anno["Query_Text"])
        generated_answer = str(first_anno["Response"])

        # Skip if no response (NaN/empty)
        if pd.isna(generated_answer) or generated_answer == "nan":
            print(f"WARNING: No response for Q{query_id} {variant}")
            continue

        # Compute majority vote
        vote_result = compute_majority_vote(query_annotations)

        case = {
            "query_id": str(query_id),
            "variant": variant,
            "query_text": query_text,
            "generated_answer": generated_answer,
            "retrieved_sources": [],  # Will need to extract from detector_package if available
            "gold_label": vote_result["majority_label"],
            "confidence": avg_confidence,
            "agreement_rate": vote_result["agreement_rate"],
            "votes": vote_result["votes"],
            "total_annotators": vote_result["total_annotators"],
            "evaluability_tier": "gold" if tier == "high_confidence_evaluable" else "silver"
        }

        cases.append(case)

    return cases


def load_sources_from_detector_package(detector_package_dir: Path) -> Dict:
    """Load retrieved sources from detector package results if available."""
    sources_by_query = {}

    results_file = detector_package_dir / "output_samples" / "round1_detector_results.json"
    if not results_file.exists():
        print("WARNING: No detector results file found, sources will be empty")
        return sources_by_query

    with open(results_file) as f:
        data = json.load(f)

    # Extract sources from each variant
    for variant_key, variant_data in data.items():
        if "reports" not in variant_data:
            continue

        for report in variant_data["reports"]:
            query_id = str(report["query_id"])
            # Note: sources might not be in the report, would need to check original data

    return sources_by_query


def main():
    """Build Gold and Silver benchmarks from raw lawyer annotations."""
    print("=" * 60)
    print("Building Gold/Silver Benchmarks from Raw Annotations")
    print("=" * 60)

    # Paths
    lawyer_pkg = Path("lawyer_data_package")
    raw_dir = lawyer_pkg / "raw_annotations"
    evaluability_path = lawyer_pkg / "analysis_outputs" / "evaluability_assessment.csv"
    detector_pkg = Path("detector_package")

    output_dir = Path("data/gold_standard")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all raw annotations
    print("\nLoading raw annotations...")
    annotations_df = load_all_annotator_files(raw_dir)

    # Load evaluability tiers
    print("\nLoading evaluability tiers...")
    evaluability_df = pd.read_csv(evaluability_path)
    print(f"Loaded {len(evaluability_df)} evaluability assessments")

    # Build Gold benchmark (high confidence: Q1, Q3)
    print("\n" + "=" * 60)
    print("Building Gold Benchmark (High Confidence)")
    print("=" * 60)
    gold_cases = build_benchmark_from_annotations(
        annotations_df, evaluability_df, "high_confidence_evaluable"
    )

    gold_path = output_dir / "gold_benchmark.json"
    with open(gold_path, "w", encoding="utf-8") as f:
        json.dump(gold_cases, f, indent=2, ensure_ascii=False)

    print(f"\n[PASS] Gold benchmark: {len(gold_cases)} cases")
    print(f"[PASS] Saved to: {gold_path}")
    for case in gold_cases:
        print(f"  Q{case['query_id']}: {case['gold_label']} (conf={case['confidence']:.1f}, agree={case['agreement_rate']:.1%})")

    # Build Silver benchmark (medium confidence)
    print("\n" + "=" * 60)
    print("Building Silver Benchmark (Medium Confidence)")
    print("=" * 60)
    silver_cases = build_benchmark_from_annotations(
        annotations_df, evaluability_df, "medium_confidence_evaluable"
    )

    silver_path = output_dir / "silver_benchmark.json"
    with open(silver_path, "w", encoding="utf-8") as f:
        json.dump(silver_cases, f, indent=2, ensure_ascii=False)

    print(f"\n[PASS] Silver benchmark: {len(silver_cases)} cases")
    print(f"[PASS] Saved to: {silver_path}")
    for case in silver_cases:
        print(f"  Q{case['query_id']}: {case['gold_label']} (conf={case['confidence']:.1f}, agree={case['agreement_rate']:.1%})")

    print("\n" + "=" * 60)
    print("Benchmark Creation Complete")
    print("=" * 60)
    print(f"Total Gold cases: {len(gold_cases)}")
    print(f"Total Silver cases: {len(silver_cases)}")
    print(f"Total cases: {len(gold_cases) + len(silver_cases)}")
    print("\nNOTE: 'retrieved_sources' are empty placeholders.")
    print("For full evaluation, sources would need to be extracted from original data.")


if __name__ == "__main__":
    main()
