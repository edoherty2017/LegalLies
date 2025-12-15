"""
Rebuild Gold and Silver benchmarks from raw lawyer annotations.

Uses fresh vote counting directly from the 6 annotator Excel files.
Extracts query text, generated answers, and computes majority labels.
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def derive_hallucination(correctness, groundedness):
    """
    Derive hallucination label from annotations.

    Hallucination = Incorrect AND (Ungrounded OR Misgrounded)
    """
    is_incorrect = 'Incorrect' in str(correctness)
    is_bad_ground = str(groundedness) in ['Ungrounded', 'Misgrounded']
    return 'Yes' if (is_incorrect and is_bad_ground) else 'No'


def load_all_annotations():
    """Load and consolidate all 6 annotator files."""
    files = {
        'A_Kenneth': 'lawyer_data_package/raw_annotations/round_1_annotation_A_Kenneth_Doherty.xlsx',
        'B_Eileen': 'lawyer_data_package/raw_annotations/round_1_annotation_B_Eileen_Doherty.xlsx',
        'C_Kenny': 'lawyer_data_package/raw_annotations/round_1_annotation_C_Kenny_Doherty.xlsx',
        'D_Liza': 'lawyer_data_package/raw_annotations/round_1_annotation_D_Liza_Jameson.xlsx',
        'E_Adi': 'lawyer_data_package/raw_annotations/round_1_annotation_E_Adi_Kennedy.xlsx',
        'F_John': 'lawyer_data_package/raw_annotations/round_1_annotation_F_John_Jameson.xlsx',
    }

    all_data = []

    for name, filepath in files.items():
        df = pd.read_excel(filepath, header=1)
        df['Annotator'] = name
        all_data.append(df)
        print(f"  Loaded {len(df)} rows from {name}")

    consolidated = pd.concat(all_data, ignore_index=True)
    return consolidated


def analyze_query_variant(annotations_df: pd.DataFrame, query_id: int, variant: str) -> Dict:
    """Analyze a single query+variant combination."""

    # Filter for this query+variant
    subset = annotations_df[
        (annotations_df['Query_ID'] == query_id) &
        (annotations_df['Variant'] == variant)
    ]

    if subset.empty:
        return None

    # Extract query text and generated answer (should be same across annotators)
    first_row = subset.iloc[0]
    query_text = str(first_row['Query_Text'])
    generated_answer = str(first_row['Response'])

    # Skip if no response
    if pd.isna(generated_answer) or generated_answer == 'nan':
        return None

    # Count votes
    votes = []
    confidences = []

    for _, row in subset.iterrows():
        correctness = row.get('Correctness', None)
        groundedness = row.get('Groundedness', None)
        confidence = row.get('Confidence_Rating', None)

        hall_vote = derive_hallucination(correctness, groundedness)
        votes.append(hall_vote)

        if pd.notna(confidence):
            confidences.append(float(confidence))

    # Compute majority
    yes_count = votes.count('Yes')
    no_count = votes.count('No')
    total = len(votes)

    if yes_count > no_count:
        majority_label = 'Hallucination'
    elif no_count > yes_count:
        majority_label = 'No_Hallucination'
    else:
        majority_label = 'Tie'

    agreement_rate = max(yes_count, no_count) / total if total > 0 else 0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 3.0

    return {
        'query_id': str(query_id),
        'variant': variant,
        'query_text': query_text,
        'generated_answer': generated_answer,
        'retrieved_sources': [],  # Will be populated separately if available
        'majority_label': majority_label,
        'confidence': float(avg_confidence),
        'agreement_rate': float(agreement_rate),
        'votes': {'Yes': yes_count, 'No': no_count},
        'total_annotators': total,
    }


def build_gold_benchmark(annotations_df: pd.DataFrame) -> List[Dict]:
    """
    Build Gold benchmark: High confidence cases (Q1, Q3).

    Criteria: Average confidence >= 4.0
    """
    gold_cases = []

    # Q1 and Q3 are the high-confidence queries
    for qid in [1, 3]:
        case = analyze_query_variant(
            annotations_df,
            qid,
            'Baseline (LLaMA 3.1 - No Web)'
        )

        if case:
            case['evaluability_tier'] = 'gold'
            gold_cases.append(case)
            print(f"  Q{qid}: {case['majority_label']} (conf={case['confidence']:.2f}, agree={case['agreement_rate']:.1%})")

    return gold_cases


def build_silver_benchmark(annotations_df: pd.DataFrame) -> List[Dict]:
    """
    Build Silver benchmark: Medium confidence cases.

    Criteria: Q2, Q4-Q11, Q14-Q15 with confidence 2.6-3.8
    """
    silver_cases = []

    # Medium confidence queries
    silver_qids = [2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15]

    for qid in silver_qids:
        case = analyze_query_variant(
            annotations_df,
            qid,
            'Baseline (LLaMA 3.1 - No Web)'
        )

        if case:
            case['evaluability_tier'] = 'silver'
            silver_cases.append(case)
            print(f"  Q{qid}: {case['majority_label']} (conf={case['confidence']:.2f}, agree={case['agreement_rate']:.1%})")

    return silver_cases


def build_excluded_benchmark(annotations_df: pd.DataFrame) -> List[Dict]:
    """
    Build Excluded benchmark: Low confidence cases.

    Criteria: Q12-Q13 with confidence < 2.5
    """
    excluded_cases = []

    for qid in [12, 13]:
        case = analyze_query_variant(
            annotations_df,
            qid,
            'Baseline (LLaMA 3.1 - No Web)'
        )

        if case:
            case['evaluability_tier'] = 'excluded'
            excluded_cases.append(case)
            print(f"  Q{qid}: {case['majority_label']} (conf={case['confidence']:.2f}, agree={case['agreement_rate']:.1%})")

    return excluded_cases


def main():
    """Rebuild all benchmark files from raw annotations."""
    print("=" * 80)
    print("Rebuilding Benchmarks from Raw Lawyer Annotations")
    print("=" * 80)

    # Load all raw annotations
    print("\nLoading raw annotations...")
    annotations_df = load_all_annotations()
    print(f"Total rows: {len(annotations_df)}")

    output_dir = Path('data/gold_standard')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build Gold benchmark
    print("\n" + "=" * 80)
    print("Gold Benchmark (High Confidence: Q1, Q3)")
    print("=" * 80)
    gold_cases = build_gold_benchmark(annotations_df)

    gold_path = output_dir / 'gold_benchmark.json'
    with open(gold_path, 'w', encoding='utf-8') as f:
        json.dump(gold_cases, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(gold_cases)} cases to {gold_path}")

    # Build Silver benchmark
    print("\n" + "=" * 80)
    print("Silver Benchmark (Medium Confidence: Q2, Q4-Q11, Q14-Q15)")
    print("=" * 80)
    silver_cases = build_silver_benchmark(annotations_df)

    silver_path = output_dir / 'silver_benchmark.json'
    with open(silver_path, 'w', encoding='utf-8') as f:
        json.dump(silver_cases, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(silver_cases)} cases to {silver_path}")

    # Build Excluded benchmark
    print("\n" + "=" * 80)
    print("Excluded Benchmark (Low Confidence: Q12-Q13)")
    print("=" * 80)
    excluded_cases = build_excluded_benchmark(annotations_df)

    excluded_path = output_dir / 'excluded_benchmark.json'
    with open(excluded_path, 'w', encoding='utf-8') as f:
        json.dump(excluded_cases, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(excluded_cases)} cases to {excluded_path}")

    # Summary
    print("\n" + "=" * 80)
    print("Benchmark Rebuild Complete")
    print("=" * 80)
    print(f"Gold cases:     {len(gold_cases)}")
    print(f"Silver cases:   {len(silver_cases)}")
    print(f"Excluded cases: {len(excluded_cases)}")
    print(f"Total:          {len(gold_cases) + len(silver_cases) + len(excluded_cases)}")

    # Save summary
    summary = {
        'gold_count': len(gold_cases),
        'silver_count': len(silver_cases),
        'excluded_count': len(excluded_cases),
        'total_count': len(gold_cases) + len(silver_cases) + len(excluded_cases),
        'source': 'raw_lawyer_annotations',
        'derivation_rule': 'Hallucination = Incorrect AND (Ungrounded OR Misgrounded)'
    }

    summary_path = output_dir / 'split_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to {summary_path}")
    print("\nNOTE: 'retrieved_sources' arrays are empty.")
    print("Sources would need to be extracted from RAG system outputs if available.")


if __name__ == '__main__':
    main()
