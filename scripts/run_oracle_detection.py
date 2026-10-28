#!/usr/bin/env python3
"""
Run hallucination detection on existing 30 responses using the oracle corpus index.

Outputs CSV with: question_id, system_type, fact_label, detector_confidence
"""

import sys
import time
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import UnifiedHallucinationDetector
from src.core.config import DetectorConfig


def main():
    # Load the existing responses
    input_path = Path("experiments/task1_fact_detection_results.csv")
    df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} responses from {input_path}")
    print(f"Questions: {df['question_id'].nunique()}, Systems: {df['system_type'].unique().tolist()}")

    # Initialize detector with oracle corpus
    print("\nInitializing UnifiedHallucinationDetector with oracle corpus...")
    config = DetectorConfig(
        retrieval_enabled=True,
        device="cuda",  # Use GPU
    )

    # Verify oracle corpus is being used
    print(f"Retrieval index path: {config.retrieval_index_path}")

    detector = UnifiedHallucinationDetector(config)
    print("[OK] Detector initialized")

    # Prepare results
    results = []

    # Process each response
    total = len(df)
    start_time = time.time()

    for idx, row in df.iterrows():
        question_id = row['question_id']
        system_type = row['system_type']
        answer_text = row['answer_text']
        original_label = row['fact_label']
        original_confidence = row['detector_confidence']

        print(f"\n[{idx+1}/{total}] Q{question_id} ({system_type})...")

        try:
            # Create a generic query from the question_id
            # Note: We don't have the original questions, so use the answer as query
            # The detector will use auto-retrieval based on the answer text
            query = f"Legal question {question_id}"

            # Run detection
            result = detector.detect(
                query=query,
                generated_answer=answer_text,
            )

            # Map hallucination to label
            if result.is_hallucinated:
                new_label = "HALLUCINATED"
            else:
                new_label = "SUPPORTED"

            new_confidence = result.confidence
            hallucination_risk = result.hallucination_risk
            num_claims = len(result.claims) if result.claims else 0
            supported_claims = sum(1 for c in result.claim_results if c.get("grounded", False)) if result.claim_results else 0
            latency_ms = result.latency_ms

            print(f"    Result: {new_label} (confidence={new_confidence:.3f}, risk={hallucination_risk:.3f})")
            print(f"    Claims: {supported_claims}/{num_claims} supported, latency={latency_ms:.0f}ms")

        except Exception as e:
            print(f"    ERROR: {str(e)[:100]}")
            new_label = "ERROR"
            new_confidence = 0.0
            hallucination_risk = 0.0
            num_claims = 0
            supported_claims = 0
            latency_ms = 0

        results.append({
            "question_id": question_id,
            "system_type": system_type,
            "fact_label": new_label,
            "detector_confidence": round(new_confidence, 4),
            "hallucination_risk": round(hallucination_risk, 4),
            "num_claims": num_claims,
            "supported_claims": supported_claims,
            "latency_ms": round(latency_ms, 1),
            "original_label": original_label,
            "original_confidence": original_confidence,
        })

    elapsed = time.time() - start_time

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = Path("experiments/oracle_corpus_detection_results.csv")
    results_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("DETECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/total:.1f}s per response)")
    print(f"Results saved to: {output_path}")

    # Summary statistics
    print(f"\n--- Summary ---")
    print(f"Total responses: {len(results_df)}")

    for system in results_df['system_type'].unique():
        subset = results_df[results_df['system_type'] == system]
        halluc_count = (subset['fact_label'] == 'HALLUCINATED').sum()
        supported_count = (subset['fact_label'] == 'SUPPORTED').sum()
        error_count = (subset['fact_label'] == 'ERROR').sum()

        print(f"\n{system.upper()}:")
        print(f"  Hallucinated: {halluc_count}/{len(subset)} ({halluc_count/len(subset)*100:.1f}%)")
        print(f"  Supported: {supported_count}/{len(subset)} ({supported_count/len(subset)*100:.1f}%)")
        if error_count > 0:
            print(f"  Errors: {error_count}")
        print(f"  Avg confidence: {subset['detector_confidence'].mean():.3f}")
        print(f"  Avg latency: {subset['latency_ms'].mean():.0f}ms")

    # Comparison with original
    print(f"\n--- Comparison with Original Results ---")
    agreement = (results_df['fact_label'] == results_df['original_label']).sum()
    print(f"Agreement with original: {agreement}/{len(results_df)} ({agreement/len(results_df)*100:.1f}%)")

    # Show changes
    changes = results_df[results_df['fact_label'] != results_df['original_label']]
    if len(changes) > 0:
        print(f"\nChanges from original ({len(changes)} total):")
        for _, row in changes.iterrows():
            print(f"  Q{row['question_id']} ({row['system_type']}): {row['original_label']} -> {row['fact_label']}")


if __name__ == "__main__":
    main()
