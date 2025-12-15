"""
Build Gold and Silver benchmark datasets from lawyer annotations.

Loads:
- Query_Collection.xlsx: Question texts
- Annotation_Round1_AB.xlsx: Lawyer annotations with answers and sources
- evaluability_assessment.csv: Gold/Silver tier assignments

Outputs:
- data/gold_standard/gold_benchmark.json: High-confidence cases (Q1, Q3)
- data/gold_standard/silver_benchmark.json: Medium-confidence cases (Q2, Q4-Q11, Q14-Q15)
"""

import json
from pathlib import Path

import pandas as pd


def load_query_collection(path: Path) -> pd.DataFrame:
    """Load query collection with question texts."""
    df = pd.read_excel(path)
    print(f"Loaded {len(df)} queries from {path.name}")
    return df


def load_annotations(path: Path) -> pd.DataFrame:
    """Load lawyer annotations."""
    df = pd.read_excel(path)
    print(f"Loaded {len(df)} annotations from {path.name}")
    return df


def load_evaluability_tiers(path: Path) -> pd.DataFrame:
    """Load evaluability tier assignments."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} evaluability assessments from {path.name}")
    return df


def derive_hallucination_label(row) -> str:
    """
    Derive hallucination label from annotations.

    Hallucination = Incorrect AND (Ungrounded OR Misgrounded)
    """
    correctness = row.get("Correctness", "")
    groundedness = row.get("Groundedness", "")

    is_incorrect = "Incorrect" in str(correctness)
    is_ungrounded = str(groundedness) in ["Ungrounded", "Misgrounded"]

    if is_incorrect and is_ungrounded:
        return "Yes"  # Hallucination
    else:
        return "No"  # Not a hallucination


def compute_majority_label(annotations: list) -> tuple:
    """
    Compute majority vote and confidence for hallucination labels.

    Returns:
        (majority_label, confidence, votes_dict, agreement_rate)
    """
    labels = [anno["hallucination_label"] for anno in annotations]

    yes_count = labels.count("Yes")
    no_count = labels.count("No")
    total = len(labels)

    if yes_count > no_count:
        majority = "Yes"
        confidence = yes_count / total
    elif no_count > yes_count:
        majority = "No"
        confidence = no_count / total
    else:
        majority = "Tie"
        confidence = 0.5

    votes = {"Yes": yes_count, "No": no_count}
    agreement_rate = max(yes_count, no_count) / total if total > 0 else 0.0

    return majority, confidence, votes, agreement_rate


def build_benchmark_cases(
    queries_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    evaluability_df: pd.DataFrame,
    tier: str
) -> list:
    """
    Build benchmark cases for a specific evaluability tier.

    Args:
        tier: "high_confidence_evaluable" or "medium_confidence_evaluable"
    """
    cases = []

    # Filter evaluability by tier
    tier_queries = evaluability_df[evaluability_df["Evaluability_Tier"] == tier]

    for _, tier_row in tier_queries.iterrows():
        query_id = str(tier_row["Query_ID"])
        variant = tier_row["Variant"]
        avg_confidence = tier_row["Avg_Confidence"]

        # Get query text
        query_row = queries_df[queries_df["Query_ID"] == int(query_id)]
        if query_row.empty:
            print(f"WARNING: No query found for Query_ID {query_id}")
            continue

        query_text = query_row.iloc[0].get("Query_Text", "")

        # Get annotations for this query + variant
        query_annotations = annotations_df[
            (annotations_df["Query_ID"] == int(query_id)) &
            (annotations_df["Variant"] == variant)
        ]

        if query_annotations.empty:
            print(f"WARNING: No annotations for Q{query_id} {variant}")
            continue

        # Extract answer and sources (should be same across annotators)
        first_anno = query_annotations.iloc[0]
        generated_answer = first_anno.get("Generated_Answer", "")

        # Sources might be in multiple columns
        sources = []
        for col in ["Retrieved_Source_1", "Retrieved_Source_2", "Retrieved_Source_3", "Retrieved_Source_4", "Retrieved_Source_5"]:
            if col in first_anno and pd.notna(first_anno[col]):
                sources.append(str(first_anno[col]))

        # Derive hallucination labels from all annotators
        annotator_labels = []
        for _, anno_row in query_annotations.iterrows():
            hall_label = derive_hallucination_label(anno_row)
            annotator_labels.append({
                "annotator": anno_row.get("Annotator_ID", "Unknown"),
                "hallucination_label": hall_label,
                "correctness": anno_row.get("Correctness", ""),
                "groundedness": anno_row.get("Groundedness", ""),
                "confidence": anno_row.get("Confidence", 3.0)
            })

        # Compute majority vote
        majority_label, vote_confidence, votes, agreement_rate = compute_majority_label(annotator_labels)

        case = {
            "query_id": query_id,
            "variant": variant,
            "query_text": query_text,
            "generated_answer": generated_answer,
            "retrieved_sources": sources,
            "gold_label": majority_label,  # "Yes" or "No" for hallucination
            "confidence": float(avg_confidence),
            "agreement_rate": float(agreement_rate),
            "votes": votes,
            "total_annotators": len(annotator_labels),
            "evaluability_tier": "gold" if tier == "high_confidence_evaluable" else "silver",
            "annotator_details": annotator_labels
        }

        cases.append(case)

    return cases


def main():
    """Build Gold and Silver benchmarks from lawyer annotations."""
    print("=" * 60)
    print("Building Gold and Silver Benchmarks")
    print("=" * 60)

    # Paths
    lawyer_pkg = Path("lawyer_data_package")
    queries_path = lawyer_pkg / "raw_annotations" / "Query_Collection.xlsx"
    annotations_path = lawyer_pkg / "raw_annotations" / "Annotation_Round1_AB.xlsx"
    evaluability_path = lawyer_pkg / "analysis_outputs" / "evaluability_assessment.csv"

    output_dir = Path("data/gold_standard")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    queries_df = load_query_collection(queries_path)
    annotations_df = load_annotations(annotations_path)
    evaluability_df = load_evaluability_tiers(evaluability_path)

    # Build Gold benchmark (high confidence)
    print("\n" + "=" * 60)
    print("Building Gold Benchmark (High Confidence)")
    print("=" * 60)
    gold_cases = build_benchmark_cases(
        queries_df, annotations_df, evaluability_df, "high_confidence_evaluable"
    )

    gold_path = output_dir / "gold_benchmark.json"
    with open(gold_path, "w") as f:
        json.dump(gold_cases, f, indent=2)

    print(f"\nGold benchmark: {len(gold_cases)} cases")
    print(f"Saved to: {gold_path}")

    for case in gold_cases:
        print(f"  Q{case['query_id']}: {case['gold_label']} (confidence={case['confidence']:.1f}, agreement={case['agreement_rate']:.1%})")

    # Build Silver benchmark (medium confidence)
    print("\n" + "=" * 60)
    print("Building Silver Benchmark (Medium Confidence)")
    print("=" * 60)
    silver_cases = build_benchmark_cases(
        queries_df, annotations_df, evaluability_df, "medium_confidence_evaluable"
    )

    silver_path = output_dir / "silver_benchmark.json"
    with open(silver_path, "w") as f:
        json.dump(silver_cases, f, indent=2)

    print(f"\nSilver benchmark: {len(silver_cases)} cases")
    print(f"Saved to: {silver_path}")

    for case in silver_cases:
        print(f"  Q{case['query_id']}: {case['gold_label']} (confidence={case['confidence']:.1f}, agreement={case['agreement_rate']:.1%})")

    print("\n" + "=" * 60)
    print("Benchmark Creation Complete")
    print("=" * 60)
    print(f"Total Gold cases: {len(gold_cases)}")
    print(f"Total Silver cases: {len(silver_cases)}")
    print(f"Total cases: {len(gold_cases) + len(silver_cases)}")


if __name__ == "__main__":
    main()
