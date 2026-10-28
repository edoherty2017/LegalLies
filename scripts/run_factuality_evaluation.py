#!/usr/bin/env python3
"""
Run factuality evaluation on the 15 Q × 2 systems (30 answers) using oracle corpus.

Uses the new factuality-only detector with contradiction-based hallucination detection.
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import FactualityDetector
from src.core.config import DetectorConfig


def main():
    # Load existing responses
    input_path = Path("experiments/task1_fact_detection_results.csv")
    df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} responses from {input_path}")
    print(f"Questions: {df['question_id'].nunique()}, Systems: {df['system_type'].unique().tolist()}")

    # Initialize detector with oracle corpus
    print("\nInitializing FactualityDetector with oracle corpus...")
    config = DetectorConfig(
        retrieval_enabled=True,
        device="cuda",
    )
    print(f"Retrieval index: {config.retrieval_index_path}")

    detector = FactualityDetector(config)
    print("[OK] Detector initialized\n")

    # Process each response
    results = []
    total = len(df)
    start_time = time.time()

    for idx, row in df.iterrows():
        question_id = row['question_id']
        system_type = row['system_type']
        answer_text = row['answer_text']

        print(f"[{idx+1}/{total}] Q{question_id} ({system_type})...", end=" ")

        try:
            # Run factuality detection (auto-retrieves from oracle corpus)
            result = detector.detect(
                query=f"Legal question {question_id}",
                answer=answer_text,
                evidence=None,  # Auto-retrieve from oracle corpus
            )

            # Determine fact_label based on new logic
            if result.is_hallucinated:
                fact_label = "HALLUCINATED"
            elif result.num_supported > 0 and result.num_unsupported == 0:
                fact_label = "SUPPORTED"
            elif result.num_unsupported > 0 and result.num_contradicted == 0:
                fact_label = "UNCERTAIN"
            else:
                fact_label = "UNCERTAIN"

            # Compute confidence (average from claim verdicts or default)
            avg_confidence = 0.0
            if result.claims:
                confidences = [c.confidence for c in result.claims if c.confidence > 0]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)

            print(f"{fact_label} (C:{result.num_contradicted}/S:{result.num_supported}/U:{result.num_unsupported})")

            results.append({
                "question_id": question_id,
                "system_type": system_type,
                "answer_text": answer_text,
                "fact_label": fact_label,
                "num_contradicted": result.num_contradicted,
                "num_supported": result.num_supported,
                "num_unsupported": result.num_unsupported,
                "num_claims": result.num_claims,
                "contradiction_rate": round(result.contradiction_rate, 4),
                "support_rate": round(result.support_rate, 4),
                "unsupported_rate": round(result.unsupported_rate, 4),
                "detector_confidence": round(avg_confidence, 4),
                "latency_ms": round(result.latency_ms, 1),
            })

        except Exception as e:
            print(f"ERROR: {str(e)[:80]}")
            results.append({
                "question_id": question_id,
                "system_type": system_type,
                "answer_text": answer_text,
                "fact_label": "ERROR",
                "num_contradicted": 0,
                "num_supported": 0,
                "num_unsupported": 0,
                "num_claims": 0,
                "contradiction_rate": 0.0,
                "support_rate": 0.0,
                "unsupported_rate": 0.0,
                "detector_confidence": 0.0,
                "latency_ms": 0,
            })

    elapsed = time.time() - start_time

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = Path("experiments/factuality_evaluation_results.csv")
    results_df.to_csv(output_path, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print("FACTUALITY EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/total:.1f}s per response)")
    print(f"Results saved to: {output_path}")

    # Summary by system type
    print(f"\n--- Results by System Type ---")
    for system in ["non_rag", "rag"]:
        subset = results_df[results_df['system_type'] == system]
        halluc = (subset['fact_label'] == 'HALLUCINATED').sum()
        supported = (subset['fact_label'] == 'SUPPORTED').sum()
        uncertain = (subset['fact_label'] == 'UNCERTAIN').sum()
        errors = (subset['fact_label'] == 'ERROR').sum()

        print(f"\n{system.upper()} ({len(subset)} responses):")
        print(f"  HALLUCINATED: {halluc} ({halluc/len(subset)*100:.1f}%)")
        print(f"  SUPPORTED:    {supported} ({supported/len(subset)*100:.1f}%)")
        print(f"  UNCERTAIN:    {uncertain} ({uncertain/len(subset)*100:.1f}%)")
        if errors > 0:
            print(f"  ERROR:        {errors}")

        print(f"  Avg contradiction rate: {subset['contradiction_rate'].mean():.3f}")
        print(f"  Avg support rate:       {subset['support_rate'].mean():.3f}")
        print(f"  Avg unsupported rate:   {subset['unsupported_rate'].mean():.3f}")

    # Overall verdict distribution
    print(f"\n--- Overall Distribution ---")
    for label in ["HALLUCINATED", "SUPPORTED", "UNCERTAIN", "ERROR"]:
        count = (results_df['fact_label'] == label).sum()
        if count > 0:
            print(f"  {label}: {count}/{total} ({count/total*100:.1f}%)")


if __name__ == "__main__":
    main()
