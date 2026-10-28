#!/usr/bin/env python3
"""
Generate new RAG and non-RAG answers using the updated retrieval pipeline,
then evaluate factuality using the oracle corpus.

This gives us the real comparison: same model, same questions,
with and without retrieval augmentation.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_questions() -> list[dict]:
    """Load the 15 legal questions from lawyer annotation data."""
    df = pd.read_excel(
        'experiments/lawyer_annotation_evaluation/results/B_Eileen_Doherty/rag_detector_results.xlsx'
    )

    questions = df[['Query_ID', 'Query_Text']].drop_duplicates()
    questions = questions.dropna(subset=['Query_ID'])
    questions['Query_ID'] = questions['Query_ID'].astype(int)
    questions = questions.sort_values('Query_ID')

    return [
        {"id": int(row['Query_ID']), "text": row['Query_Text']}
        for _, row in questions.iterrows()
    ]


def generate_nonrag_answers(questions: list[dict], generator) -> list[dict]:
    """Generate answers without retrieval augmentation."""
    results = []

    for q in questions:
        print(f"  Q{q['id']}: Generating...", end=" ", flush=True)
        start = time.time()

        try:
            result = generator.generate(q['text'])
            answer = result.generated_answer
            latency = (time.time() - start) * 1000
            print(f"Done ({latency:.0f}ms, {len(answer)} chars)")

            results.append({
                "question_id": q['id'],
                "question_text": q['text'],
                "system_type": "non_rag",
                "answer_text": answer,
                "generation_latency_ms": latency,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "question_id": q['id'],
                "question_text": q['text'],
                "system_type": "non_rag",
                "answer_text": f"ERROR: {str(e)}",
                "generation_latency_ms": 0,
            })

    return results


def generate_rag_answers(questions: list[dict], generator) -> list[dict]:
    """Generate answers with retrieval augmentation from oracle corpus."""
    results = []

    for q in questions:
        print(f"  Q{q['id']}: Retrieving + Generating...", end=" ", flush=True)
        start = time.time()

        try:
            result = generator.generate(q['text'])
            answer = result.generated_answer
            context = result.retrieved_context
            latency = (time.time() - start) * 1000
            print(f"Done ({latency:.0f}ms, {len(answer)} chars, {len(context)} docs)")

            results.append({
                "question_id": q['id'],
                "question_text": q['text'],
                "system_type": "rag",
                "answer_text": answer,
                "retrieved_context": context,
                "num_retrieved": len(context),
                "generation_latency_ms": latency,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "question_id": q['id'],
                "question_text": q['text'],
                "system_type": "rag",
                "answer_text": f"ERROR: {str(e)}",
                "retrieved_context": [],
                "num_retrieved": 0,
                "generation_latency_ms": 0,
            })

    return results


def evaluate_factuality(answers: list[dict], detector) -> list[dict]:
    """Run factuality detection on generated answers."""
    results = []

    for ans in answers:
        q_id = ans['question_id']
        system = ans['system_type']
        answer_text = ans['answer_text']

        print(f"  Q{q_id} ({system}): Evaluating...", end=" ", flush=True)

        if answer_text.startswith("ERROR:"):
            print("SKIPPED (generation error)")
            results.append({
                **ans,
                "fact_label": "ERROR",
                "num_contradicted": 0,
                "num_supported": 0,
                "num_unsupported": 0,
                "num_claims": 0,
                "contradiction_rate": 0.0,
                "support_rate": 0.0,
                "unsupported_rate": 0.0,
                "detection_latency_ms": 0,
            })
            continue

        try:
            # For RAG, use the retrieved context as evidence
            # For non-RAG, auto-retrieve from oracle corpus
            evidence = ans.get('retrieved_context', None)

            result = detector.detect(
                query=ans['question_text'],
                answer=answer_text,
                evidence=evidence,
            )

            # Determine fact_label
            if result.is_hallucinated:
                fact_label = "HALLUCINATED"
            elif result.num_supported > 0 and result.num_unsupported == 0:
                fact_label = "SUPPORTED"
            else:
                fact_label = "UNCERTAIN"

            print(f"{fact_label} (C:{result.num_contradicted}/S:{result.num_supported}/U:{result.num_unsupported})")

            results.append({
                **ans,
                "fact_label": fact_label,
                "num_contradicted": result.num_contradicted,
                "num_supported": result.num_supported,
                "num_unsupported": result.num_unsupported,
                "num_claims": result.num_claims,
                "contradiction_rate": round(result.contradiction_rate, 4),
                "support_rate": round(result.support_rate, 4),
                "unsupported_rate": round(result.unsupported_rate, 4),
                "detection_latency_ms": round(result.latency_ms, 1),
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                **ans,
                "fact_label": "ERROR",
                "num_contradicted": 0,
                "num_supported": 0,
                "num_unsupported": 0,
                "num_claims": 0,
                "contradiction_rate": 0.0,
                "support_rate": 0.0,
                "unsupported_rate": 0.0,
                "detection_latency_ms": 0,
            })

    return results


def print_summary(results_df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("GENERATION & EVALUATION SUMMARY")
    print("=" * 70)

    for system in ["non_rag", "rag"]:
        subset = results_df[results_df['system_type'] == system]
        total = len(subset)

        halluc = (subset['fact_label'] == 'HALLUCINATED').sum()
        supported = (subset['fact_label'] == 'SUPPORTED').sum()
        uncertain = (subset['fact_label'] == 'UNCERTAIN').sum()
        errors = (subset['fact_label'] == 'ERROR').sum()

        print(f"\n{system.upper()} ({total} responses):")
        print(f"  HALLUCINATED: {halluc:2d} ({halluc/total*100:5.1f}%)")
        print(f"  SUPPORTED:    {supported:2d} ({supported/total*100:5.1f}%)")
        print(f"  UNCERTAIN:    {uncertain:2d} ({uncertain/total*100:5.1f}%)")
        if errors > 0:
            print(f"  ERROR:        {errors:2d}")

        # Claim-level stats
        valid = subset[subset['fact_label'] != 'ERROR']
        if len(valid) > 0:
            print(f"\n  Claim-level averages:")
            print(f"    Contradiction rate: {valid['contradiction_rate'].mean():.1%}")
            print(f"    Support rate:       {valid['support_rate'].mean():.1%}")
            print(f"    Unsupported rate:   {valid['unsupported_rate'].mean():.1%}")
            print(f"    Avg claims/answer:  {valid['num_claims'].mean():.1f}")

    # Comparison
    print("\n" + "-" * 70)
    print("RAG vs NON-RAG COMPARISON")
    print("-" * 70)

    nonrag = results_df[results_df['system_type'] == 'non_rag']
    rag = results_df[results_df['system_type'] == 'rag']

    nonrag_halluc = (nonrag['fact_label'] == 'HALLUCINATED').mean()
    rag_halluc = (rag['fact_label'] == 'HALLUCINATED').mean()

    nonrag_contra = nonrag['contradiction_rate'].mean()
    rag_contra = rag['contradiction_rate'].mean()

    print(f"\nHallucination rate:")
    print(f"  Non-RAG: {nonrag_halluc:.1%}")
    print(f"  RAG:     {rag_halluc:.1%}")
    print(f"  Δ:       {rag_halluc - nonrag_halluc:+.1%}")

    print(f"\nContradiction rate:")
    print(f"  Non-RAG: {nonrag_contra:.1%}")
    print(f"  RAG:     {rag_contra:.1%}")
    print(f"  Δ:       {rag_contra - nonrag_contra:+.1%}")


def main():
    from src.generation import create_llama_generator, RAGGenerator
    from src.retrieval.document_retriever import DocumentRetriever
    from src.detection import FactualityDetector
    from src.core.config import DetectorConfig

    output_dir = Path("experiments/new_generation_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    print("Loading questions...")
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    # Initialize base generator
    print("\n" + "=" * 70)
    print("INITIALIZING LLAMA 3.1 8B GENERATOR")
    print("=" * 70)

    base_generator = create_llama_generator()

    # Generate non-RAG answers
    print("\n" + "=" * 70)
    print("GENERATING NON-RAG ANSWERS (15 questions)")
    print("=" * 70)

    nonrag_answers = generate_nonrag_answers(questions, base_generator)

    # Initialize retriever for RAG
    print("\n" + "=" * 70)
    print("INITIALIZING ORACLE CORPUS RETRIEVER")
    print("=" * 70)

    retriever = DocumentRetriever(
        index_path="data/indexes/oracle_corpus",
        device="cuda",
    )

    # Create RAG generator
    rag_generator = RAGGenerator(
        base_generator=base_generator,
        retriever=retriever,
        top_k=5,
        include_sources=True,
    )

    # Generate RAG answers
    print("\n" + "=" * 70)
    print("GENERATING RAG ANSWERS (15 questions)")
    print("=" * 70)

    rag_answers = generate_rag_answers(questions, rag_generator)

    # Combine all answers
    all_answers = nonrag_answers + rag_answers

    # Save raw generation results
    gen_df = pd.DataFrame(all_answers)
    gen_path = output_dir / "generated_answers.csv"
    # Don't save retrieved_context to CSV (too large)
    gen_df_save = gen_df.drop(columns=['retrieved_context'], errors='ignore')
    gen_df_save.to_csv(gen_path, index=False)
    print(f"\nGenerated answers saved to: {gen_path}")

    # Clear GPU memory before loading detector
    print("\nClearing GPU memory...")
    del base_generator
    del rag_generator
    torch.cuda.empty_cache()

    # Initialize factuality detector
    print("\n" + "=" * 70)
    print("INITIALIZING FACTUALITY DETECTOR")
    print("=" * 70)

    config = DetectorConfig(
        retrieval_enabled=True,
        device="cuda",
    )
    detector = FactualityDetector(config)

    # Run factuality evaluation
    print("\n" + "=" * 70)
    print("RUNNING FACTUALITY EVALUATION (30 responses)")
    print("=" * 70)

    evaluated = evaluate_factuality(all_answers, detector)

    # Create final results DataFrame
    results_df = pd.DataFrame(evaluated)

    # Save results
    results_path = output_dir / "factuality_results.csv"
    results_df_save = results_df.drop(columns=['retrieved_context'], errors='ignore')
    results_df_save.to_csv(results_path, index=False)
    print(f"\nFactuality results saved to: {results_path}")

    # Print summary
    print_summary(results_df)

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(questions),
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "retrieval_index": "data/indexes/oracle_corpus",
        "retrieval_top_k": 5,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
