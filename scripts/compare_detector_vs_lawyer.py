#!/usr/bin/env python3
"""
EXPERIMENT 2: Compare Detector vs Lawyer Labels

Core thesis:
- Lawyers judge REASONABLENESS (does this answer make sense legally?)
- Detector judges FACTUAL ACCURACY (does this contradict the oracle corpus?)
→ They WILL disagree by design

Outputs:
- Agreement rate
- Confusion matrix
- Per-question disagreement patterns
- Publishable figure
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_lawyer_annotations() -> pd.DataFrame:
    """Load lawyer annotations from all annotators."""
    results_dir = Path("experiments/lawyer_annotation_evaluation/results")

    all_annotations = []

    for lawyer_dir in results_dir.iterdir():
        if not lawyer_dir.is_dir() or lawyer_dir.name in ['visualizations']:
            continue

        rag_file = lawyer_dir / "rag_detector_results.xlsx"
        if rag_file.exists():
            df = pd.read_excel(rag_file)

            # Skip if no annotations
            if df['Hallucination_Label'].isna().all():
                continue

            df['lawyer'] = lawyer_dir.name
            all_annotations.append(df)

    if not all_annotations:
        raise ValueError("No lawyer annotations found")

    combined = pd.concat(all_annotations, ignore_index=True)
    return combined


def prepare_comparison_data(lawyer_df: pd.DataFrame, detector_df: pd.DataFrame) -> pd.DataFrame:
    """Merge lawyer and detector results."""

    # Clean lawyer data
    lawyer_clean = lawyer_df.copy()

    # Forward-fill Query_ID (it's only on the first row of each pair)
    lawyer_clean['Query_ID'] = lawyer_clean['Query_ID'].ffill()

    # Map Variant to system_type
    variant_map = {
        'Baseline (LLaMA 3.1 - No Web)': 'non_rag',
        'RAG+Web (LLaMA 3.1 + Retrieval + Web)': 'rag'
    }
    lawyer_clean['system_type'] = lawyer_clean['Variant'].map(variant_map)

    # Map lawyer labels to binary
    lawyer_clean['lawyer_hallucinated'] = lawyer_clean['Hallucination_Label'].map({'Yes': True, 'No': False})

    # Keep relevant columns
    lawyer_subset = lawyer_clean[['Query_ID', 'system_type', 'Correctness', 'Groundedness',
                                   'Hallucination_Label', 'lawyer_hallucinated', 'lawyer']].copy()
    lawyer_subset = lawyer_subset.rename(columns={'Query_ID': 'question_id'})
    lawyer_subset['question_id'] = lawyer_subset['question_id'].astype(int)

    # Merge with detector results
    merged = pd.merge(
        detector_df,
        lawyer_subset,
        on=['question_id', 'system_type'],
        how='inner'
    )

    # Create detector binary label
    merged['detector_hallucinated'] = merged['fact_label'] == 'HALLUCINATED'

    return merged


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute agreement metrics."""

    # Overall agreement
    agreement = (df['detector_hallucinated'] == df['lawyer_hallucinated']).mean()

    # Confusion matrix counts
    # Rows: Detector prediction, Cols: Lawyer label
    tp = ((df['detector_hallucinated'] == True) & (df['lawyer_hallucinated'] == True)).sum()
    tn = ((df['detector_hallucinated'] == False) & (df['lawyer_hallucinated'] == False)).sum()
    fp = ((df['detector_hallucinated'] == True) & (df['lawyer_hallucinated'] == False)).sum()
    fn = ((df['detector_hallucinated'] == False) & (df['lawyer_hallucinated'] == True)).sum()

    total = len(df)

    return {
        'agreement_rate': agreement,
        'total_samples': total,
        'confusion_matrix': {
            'tp': int(tp),  # Both say hallucinated
            'tn': int(tn),  # Both say not hallucinated
            'fp': int(fp),  # Detector says hallucinated, lawyer says no
            'fn': int(fn),  # Detector says not hallucinated, lawyer says yes
        },
        'detector_positive_rate': (tp + fp) / total,
        'lawyer_positive_rate': (tp + fn) / total,
    }


def analyze_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze patterns in disagreements."""

    # Flag disagreements
    df['disagree'] = df['detector_hallucinated'] != df['lawyer_hallucinated']
    df['disagree_type'] = 'AGREE'
    df.loc[(df['detector_hallucinated'] == True) & (df['lawyer_hallucinated'] == False), 'disagree_type'] = 'DETECTOR_STRICTER'
    df.loc[(df['detector_hallucinated'] == False) & (df['lawyer_hallucinated'] == True), 'disagree_type'] = 'LAWYER_STRICTER'

    return df


def print_report(merged: pd.DataFrame, metrics: dict):
    """Print analysis report."""

    print("=" * 70)
    print("EXPERIMENT 2: DETECTOR vs LAWYER COMPARISON")
    print("=" * 70)
    print()
    print("Core Thesis:")
    print("  - Lawyers judge REASONABLENESS (legal sensibility)")
    print("  - Detector judges FACTUAL ACCURACY (contradiction with corpus)")
    print("  → They WILL disagree (by design)")
    print()

    print("-" * 70)
    print("OVERALL METRICS")
    print("-" * 70)
    print(f"Total comparisons: {metrics['total_samples']}")
    print(f"Agreement rate: {metrics['agreement_rate']:.1%}")
    print(f"Disagreement rate: {1 - metrics['agreement_rate']:.1%}")
    print()

    print("-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)
    cm = metrics['confusion_matrix']
    print()
    print("                          LAWYER LABEL")
    print("                    Hallucinated    Not Halluc.")
    print("                    ───────────────────────────")
    print(f"DETECTOR  Halluc.  │    {cm['tp']:3d}            {cm['fp']:3d}      │")
    print(f"          Not H.   │    {cm['fn']:3d}            {cm['tn']:3d}      │")
    print("                    ───────────────────────────")
    print()
    print(f"  TP (Both hallucinated):     {cm['tp']:3d}")
    print(f"  TN (Both not hallucinated): {cm['tn']:3d}")
    print(f"  FP (Detector stricter):     {cm['fp']:3d}  ← Detector finds contradiction, lawyer says OK")
    print(f"  FN (Lawyer stricter):       {cm['fn']:3d}  ← Lawyer finds issue, detector finds no contradiction")
    print()

    print("-" * 70)
    print("POSITIVE RATES")
    print("-" * 70)
    print(f"Detector hallucination rate: {metrics['detector_positive_rate']:.1%}")
    print(f"Lawyer hallucination rate:   {metrics['lawyer_positive_rate']:.1%}")
    print()

    # Per-system breakdown
    print("-" * 70)
    print("BREAKDOWN BY SYSTEM TYPE")
    print("-" * 70)
    for system in ['non_rag', 'rag']:
        subset = merged[merged['system_type'] == system]
        agree = (subset['detector_hallucinated'] == subset['lawyer_hallucinated']).mean()
        det_rate = subset['detector_hallucinated'].mean()
        law_rate = subset['lawyer_hallucinated'].mean()

        print(f"\n{system.upper()}:")
        print(f"  Samples: {len(subset)}")
        print(f"  Agreement: {agree:.1%}")
        print(f"  Detector halluc rate: {det_rate:.1%}")
        print(f"  Lawyer halluc rate:   {law_rate:.1%}")

    # Per-question analysis
    print()
    print("-" * 70)
    print("PER-QUESTION DISAGREEMENT ANALYSIS")
    print("-" * 70)

    question_stats = merged.groupby('question_id').agg({
        'disagree': 'mean',
        'detector_hallucinated': 'mean',
        'lawyer_hallucinated': 'mean',
    }).round(2)
    question_stats.columns = ['disagree_rate', 'detector_rate', 'lawyer_rate']
    question_stats = question_stats.sort_values('disagree_rate', ascending=False)

    print("\nQuestions sorted by disagreement rate:")
    print(question_stats.to_string())

    # Disagreement type breakdown
    print()
    print("-" * 70)
    print("DISAGREEMENT TYPES")
    print("-" * 70)
    type_counts = merged['disagree_type'].value_counts()
    for dtype, count in type_counts.items():
        pct = count / len(merged) * 100
        print(f"  {dtype}: {count} ({pct:.1f}%)")

    # Key insight
    print()
    print("=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    fp_rate = cm['fp'] / (cm['fp'] + cm['tn']) if (cm['fp'] + cm['tn']) > 0 else 0
    fn_rate = cm['fn'] / (cm['fn'] + cm['tp']) if (cm['fn'] + cm['tp']) > 0 else 0

    print(f"""
The detector finds contradictions in {metrics['detector_positive_rate']:.0%} of responses.
Lawyers flag hallucinations in {metrics['lawyer_positive_rate']:.0%} of responses.

When lawyers say "not hallucinated", detector disagrees {fp_rate:.0%} of the time.
When lawyers say "hallucinated", detector disagrees {fn_rate:.0%} of the time.

This demonstrates the fundamental difference:
- Detector: "Does this CONTRADICT known legal facts?"
- Lawyer: "Is this a REASONABLE legal answer?"

A response can be reasonable but contain factual contradictions.
A response can be unreasonable but not directly contradict sources.
""")


def save_results(merged: pd.DataFrame, metrics: dict, output_dir: Path):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save merged comparison data
    comparison_cols = [
        'question_id', 'system_type', 'fact_label', 'detector_hallucinated',
        'Correctness', 'Groundedness', 'Hallucination_Label', 'lawyer_hallucinated',
        'num_contradicted', 'num_supported', 'num_unsupported', 'contradiction_rate',
        'disagree', 'disagree_type', 'lawyer'
    ]
    merged[comparison_cols].to_csv(output_dir / "detector_vs_lawyer_comparison.csv", index=False)

    # Save metrics
    import json
    with open(output_dir / "comparison_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - detector_vs_lawyer_comparison.csv")
    print(f"  - comparison_metrics.json")


def main():
    # Load data
    print("Loading lawyer annotations...")
    lawyer_df = load_lawyer_annotations()
    print(f"  Loaded {len(lawyer_df)} annotations from {lawyer_df['lawyer'].nunique()} lawyers")

    print("\nLoading detector results...")
    detector_df = pd.read_csv("experiments/factuality_evaluation_results.csv")
    print(f"  Loaded {len(detector_df)} detector results")

    # Merge and prepare
    print("\nMerging data...")
    merged = prepare_comparison_data(lawyer_df, detector_df)
    print(f"  Matched {len(merged)} comparisons")

    # Analyze disagreements
    merged = analyze_disagreements(merged)

    # Compute metrics
    metrics = compute_metrics(merged)

    # Print report
    print_report(merged, metrics)

    # Save results
    output_dir = Path("experiments/detector_vs_lawyer")
    save_results(merged, metrics, output_dir)


if __name__ == "__main__":
    main()
