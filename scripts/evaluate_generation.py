#!/usr/bin/env python3
"""
Evaluation script for the 4 experimental variants.

Variants:
1. Baseline - Llama3.1-8B only
2. RAG - Llama3.1-8B + retrieval
3. Steering - Llama3.1-8B + activation steering
4. RAG + Steering - Both interventions

Each variant is evaluated with the hallucination detector to measure
factual accuracy (contradiction rate, support rate).

REPRODUCIBILITY:
- All variants use the same random seed (default: 42)
- Temperature, top_p, top_k consistent across all variants
- Set SEED environment variable to override: SEED=123 python evaluate_generation.py
"""

from pathlib import Path
import os
import sys
from typing import List, Dict, Optional
import pandas as pd
import time
import torch
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reproducibility: Get seed from environment or use default
SEED = int(os.environ.get("SEED", 42))

from src.core.config import DetectorConfig
from src.detection.unified_detector import UnifiedHallucinationDetector
from src.generation import (
    create_llama_generator,
    create_llama_rag_generator,
    create_steered_llama_generator,
    create_steered_rag_generator,
)
from src.generation.base_generator import set_seed


# 15 Lawyer-Annotated Questions (from gold/silver/excluded benchmarks)
TEST_QUESTIONS = [
    # Q1 - Gold
    'Does the inevitable discovery doctrine create a per se exception to the exclusionary rule for evidence seized after a Fourth Amendment "knock and announce" violation or is evidence subject to suppression after such violations?',
    # Q2 - Silver
    'Is it the case that a claimed invention cannot be held "obvious", and thus unpatentable under 35 U.S.C. Section 103(a), in the absence of some specific, proven "teaching, suggestion, or motivation" that would have led a person of ordinary skill in the art to combine the relevant prior art teachings in the manner claimed?',
    # Q3 - Gold
    'Did Congress divest the federal district courts of their federal-question jurisdiction under 28 U.S.C. Section 1331 over private actions brought under the Telephone Consumer Protection Act?',
    # Q4 - Silver
    'Does imposing failure-to-train liability on a district attorney\'s office for a single Brady violation contravene the rigorous culpability and causation standards of Canton and Bryan County?',
    # Q5 - Silver
    'Should the Federal Communications Commission\'s determination that the broadcast of vulgar expletives may violate federal restrictions on the broadcast of "any obscene, indecent, or profane language," 18 U.S.C. 1464, when the expletives are not repeated, be stricken down?',
    # Q6 - Silver
    'Does the Equal Protection Clause preclude a local taxing authority from refusing to refund payments made by those who have paid their assessments in full, even though they forgave the obligations of identically situated taxpayers who chose to pay over a multi-year installment plan?',
    # Q7 - Silver
    'Is a habeas petitioner\'s claim considered "adjudicated on the merits" for purposes of 28 U.S.C. Section 2254(d) where the state court denied relief in an explained decision but did not expressly acknowledge a federal-law basis for the claim?',
    # Q8 - Silver
    'The deadlines for objecting to a debtor\'s discharge in bankruptcy are set by Bankruptcy Rule 4004. Are the deadlines established by the bankruptcy rules for objecting to discharge jurisdictional?',
    # Q9 - Silver
    'Bankruptcy Rules permit discharge of a student loan only through an adversary proceeding, commenced by filing a complaint and serving it and a summons on an appropriate agent of the creditor. Instead, debtor merely included a declaration of discharge in his Chapter 13 plan and mailed it to creditor\'s post office box. Does such procedure meet the rigorous demands of due process and entitle the resulting orders to respect under principles of res judicata?',
    # Q10 - Silver
    'Must a district court admit "me, too" evidence -- i.e., testimony, by nonparties, alleging discrimination at the hands of persons who played no role in the adverse employment decision challenged by the plaintiff?',
    # Q11 - Silver
    'In Phillips v. Washington Legal Found., 524 U.S. 156 (1998), the Supreme Court held that the interest on clients\' funds held in IOLTA accounts was the property of the clients. Does a regulatory scheme for funding state legal services by systematically seizing this property violate the Takings Clause of the Fifth Amendment?',
    # Q12 - Excluded
    'Do 28 U.S.C. Sections 1346(b) and 2680(h) waive the sovereign immunity of the United States for the intentional torts of prison guards when they are acting within the scope of their employment but are not exercising their authority to execute searches, to seize evidence, or to make arrests for violations of federal law?',
    # Q13 - Excluded
    'A department of corrections revised its prison visitation policy to prohibit visits by minor children unless immediate family, require accompanying parent/guardian, prohibit former inmates from visiting, and impose a two-year visitation ban for substance abuse violations. Do these policies violate prisoners\' First and Fourteenth Amendment rights?',
    # Q14 - Silver
    'Does a defendant who has fraudulently obtained a loan and thus owes restitution for the loan under 18 U.S.C. Section 3663A(b)(1)(B) return "any part" of the loan money by giving the lenders the collateral that secures the money?',
    # Q15 - Silver
    'Does a case become moot, and thus beyond the judicial power of Article III, when the lone plaintiff receives an offer from the defendants to satisfy all of the plaintiff\'s claims?',
]

# Steering vector path
STEERING_VECTOR_PATH = "steering_materials/Vectors/llama3.1-8b_hallu_steering_layer_15.pt"


def clear_gpu_memory():
    """Clear GPU memory between model loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("[OK] GPU memory cleared")


def evaluate_variant(
    variant_name: str,
    generator,
    detector: UnifiedHallucinationDetector,
    questions: List[str],
) -> List[Dict]:
    """
    Evaluate a single variant on all questions.

    Args:
        variant_name: Name of the variant (e.g., "Baseline", "RAG")
        generator: Generator instance
        detector: Hallucination detector
        questions: List of questions to evaluate

    Returns:
        List of result dictionaries
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {variant_name}")
    print("=" * 60)

    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question}")

        # Generate answer
        print("  Generating answer...")
        start_time = time.time()
        gen_result = generator.generate(question)
        gen_latency = (time.time() - start_time) * 1000

        answer = gen_result.generated_answer
        retrieved_context = gen_result.retrieved_context

        # Truncate for display
        display_answer = answer[:150] + "..." if len(answer) > 150 else answer
        print(f"  Answer: {display_answer}")
        print(f"  Generation time: {gen_latency:.1f}ms")

        # Detect hallucinations
        print("  Running hallucination detection...")
        detection_result = detector.detect(
            query=question,
            answer=answer,
            evidence=retrieved_context,
        )

        print(f"  Hallucination: {detection_result.is_hallucinated}")
        print(f"  Testability: {detection_result.testability_rate:.1%} ({detection_result.num_testable}/{detection_result.num_claims} claims adjudicable)")
        print(f"  Hallucination rate: {detection_result.hallucination_rate:.1%} ({detection_result.num_contradicted}/{detection_result.num_testable} testable wrong)")
        print(f"  Breakdown: {detection_result.num_supported} supported, {detection_result.num_contradicted} contradicted, {detection_result.num_unsupported} unsupported")
        print(f"  Detection time: {detection_result.latency_ms:.1f}ms")

        # Store results
        result = {
            "variant": variant_name,
            "question": question,
            "answer": answer,
            "has_sources": retrieved_context is not None,
            "num_sources": len(retrieved_context) if retrieved_context else 0,
            "generation_latency_ms": gen_latency,
            "is_hallucinated": detection_result.is_hallucinated,
            # DUAL METRICS (always report both per guidance)
            "hallucination_rate": detection_result.hallucination_rate,  # Contradicted / Testable
            "testability_rate": detection_result.testability_rate,  # Testable / Total
            "num_testable": detection_result.num_testable,
            # Claim counts
            "num_claims": detection_result.num_claims,
            "num_contradicted": detection_result.num_contradicted,
            "num_supported": detection_result.num_supported,
            "num_unsupported": detection_result.num_unsupported,
            # Raw rates over all claims (for detailed analysis)
            "contradiction_rate": detection_result.contradiction_rate,
            "support_rate": detection_result.support_rate,
            "unsupported_rate": detection_result.unsupported_rate,
            "detection_latency_ms": detection_result.latency_ms,
            "total_latency_ms": gen_latency + detection_result.latency_ms,
        }
        results.append(result)

    return results


def main():
    """Run evaluation on all 4 variants."""
    print("=" * 60)
    print("LegalLies: 4-Variant Hallucination Evaluation")
    print("=" * 60)
    print("\nVariants:")
    print("  1. Baseline - Llama3.1-8B only")
    print("  2. RAG - Llama3.1-8B + retrieval")
    print("  3. Steering - Llama3.1-8B + activation steering")
    print("  4. RAG + Steering - Both interventions")

    # Set global seed for reproducibility BEFORE any model loading
    print(f"\n[Reproducibility] Using seed: {SEED}")
    print("[Reproducibility] Generation params: temperature=0.7, top_p=0.9, top_k=50, do_sample=True")
    set_seed(SEED)

    # Check prerequisites
    index_path = Path("data/indexes/oracle_corpus")
    if not index_path.exists():
        print(f"\n[ERROR] Retrieval index not found at {index_path}")
        print("Run: python scripts/build_oracle_index.py")
        return

    steering_path = Path(STEERING_VECTOR_PATH)
    if not steering_path.exists():
        print(f"\n[ERROR] Steering vector not found at {steering_path}")
        return

    # Initialize detector with CUDA
    print("\nInitializing hallucination detector...")
    config = DetectorConfig(device="cuda")
    detector = UnifiedHallucinationDetector(config)

    all_results = []
    output_path = Path("data/evaluation_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save_results(results_list):
        """Save results incrementally."""
        if results_list:
            df = pd.DataFrame(results_list)
            df.to_csv(output_path, index=False)
            print(f"[OK] Results saved to: {output_path}")

    # ========================================
    # Variant 1: Baseline (No RAG, No Steering)
    # ========================================
    try:
        print("\n\n" + "=" * 60)
        print("VARIANT 1: Baseline")
        print("=" * 60)
        print("Loading Llama3.1-8B...")
        generator = create_llama_generator(seed=SEED)
        results = evaluate_variant(
            "Baseline",
            generator,
            detector,
            TEST_QUESTIONS,
        )
        all_results.extend(results)
        save_results(all_results)
        del generator
        clear_gpu_memory()
    except Exception as e:
        print(f"\n[ERROR] Baseline failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================
    # Variant 2: RAG (Retrieval, No Steering)
    # ========================================
    try:
        print("\n\n" + "=" * 60)
        print("VARIANT 2: RAG")
        print("=" * 60)
        print("Loading Llama3.1-8B + RAG...")
        generator = create_llama_rag_generator(
            index_path=str(index_path),
            top_k=3,
            seed=SEED,
        )
        results = evaluate_variant(
            "RAG",
            generator,
            detector,
            TEST_QUESTIONS,
        )
        all_results.extend(results)
        save_results(all_results)
        del generator
        clear_gpu_memory()
    except Exception as e:
        print(f"\n[ERROR] RAG failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================
    # Variant 3: Steering (No RAG, Steering)
    # ========================================
    try:
        print("\n\n" + "=" * 60)
        print("VARIANT 3: Steering")
        print("=" * 60)
        print("Loading Llama3.1-8B + Steering...")
        generator = create_steered_llama_generator(
            steering_vector_path=STEERING_VECTOR_PATH,
            multiplier=1.0,
            seed=SEED,
        )
        results = evaluate_variant(
            "Steering",
            generator,
            detector,
            TEST_QUESTIONS,
        )
        all_results.extend(results)
        save_results(all_results)
        del generator
        clear_gpu_memory()
    except Exception as e:
        print(f"\n[ERROR] Steering failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================
    # Variant 4: RAG + Steering (Both)
    # ========================================
    try:
        print("\n\n" + "=" * 60)
        print("VARIANT 4: RAG + Steering")
        print("=" * 60)
        print("Loading Llama3.1-8B + RAG + Steering...")
        generator = create_steered_rag_generator(
            steering_vector_path=STEERING_VECTOR_PATH,
            index_path=str(index_path),
            multiplier=1.0,
            top_k=3,
            seed=SEED,
        )
        results = evaluate_variant(
            "RAG + Steering",
            generator,
            detector,
            TEST_QUESTIONS,
        )
        all_results.extend(results)
        save_results(all_results)
        del generator
        clear_gpu_memory()
    except Exception as e:
        print(f"\n[ERROR] RAG + Steering failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================
    # Summary Statistics (Claim-Level, Testable Only)
    # ========================================
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_path, index=False)
        print(f"\n\n{'=' * 80}")
        print("FINAL RESULTS - DUAL METRICS (Claim-Level Analysis)")
        print("=" * 80)
        print("\nDUAL METRICS (per fact-verification framework guidance):")
        print("  1. Testability  = Testable / Total     ('What % of claims could we adjudicate?')")
        print("  2. Hallucination = Contradicted / Testable  ('Of verifiable claims, what % were wrong?')")

        print("\n" + "-" * 80)
        print(f"{'Variant':<18} {'Claims':>7} {'Testable':>9} {'Contra':>7} {'Supp':>6} {'Testabil%':>10} {'Halluc%':>9}")
        print("-" * 80)

        for variant in ["Baseline", "RAG", "Steering", "RAG + Steering"]:
            if variant in df["variant"].values:
                v = df[df["variant"] == variant]

                # Use direct counts from detector results
                total_claims = int(v["num_claims"].sum())
                total_contradicted = int(v["num_contradicted"].sum())
                total_supported = int(v["num_supported"].sum())
                total_testable = int(v["num_testable"].sum())

                # DUAL METRICS
                testability = (total_testable / total_claims * 100) if total_claims > 0 else 0
                hallucin_rate = (total_contradicted / total_testable * 100) if total_testable > 0 else 0

                print(f"{variant:<18} {total_claims:>7} {total_testable:>9} {total_contradicted:>7} {total_supported:>6} {testability:>9.1f}% {hallucin_rate:>8.1f}%")

        print("-" * 80)

        # Overall totals (using direct counts)
        total_all_claims = int(df["num_claims"].sum())
        total_all_contradicted = int(df["num_contradicted"].sum())
        total_all_supported = int(df["num_supported"].sum())
        total_all_unsupported = int(df["num_unsupported"].sum())
        total_all_testable = int(df["num_testable"].sum())

        overall_testability = (total_all_testable / total_all_claims * 100) if total_all_claims > 0 else 0
        overall_hallucination_rate = (total_all_contradicted / total_all_testable * 100) if total_all_testable > 0 else 0

        print(f"\nOVERALL TOTALS:")
        print(f"  Total claims made: {total_all_claims}")
        print(f"  Adjudicable (testable): {total_all_testable}")
        print(f"  Unadjudicable (no evidence): {total_all_unsupported}")
        print(f"  ---")
        print(f"  Contradicted (hallucinations): {total_all_contradicted}")
        print(f"  Supported (validated): {total_all_supported}")
        print(f"\n  >>> TESTABILITY RATE: {overall_testability:.1f}% of claims were adjudicable <<<")
        print(f"  >>> HALLUCINATION RATE: {overall_hallucination_rate:.1f}% of testable claims were wrong <<<")

        # Per-variant detailed breakdown
        print(f"\n\n{'=' * 70}")
        print("PER-VARIANT DETAILED BREAKDOWN")
        print("=" * 70)

        for variant in ["Baseline", "RAG", "Steering", "RAG + Steering"]:
            if variant in df["variant"].values:
                v = df[df["variant"] == variant]
                # Use direct counts from detector results
                total_claims = int(v["num_claims"].sum())
                total_contradicted = int(v["num_contradicted"].sum())
                total_supported = int(v["num_supported"].sum())
                total_unsupported = int(v["num_unsupported"].sum())
                total_testable = int(v["num_testable"].sum())
                testability = (total_testable / total_claims * 100) if total_claims > 0 else 0
                hallucin_rate = (total_contradicted / total_testable * 100) if total_testable > 0 else 0
                n_questions = len(v)

                print(f"\n{variant}:")
                print(f"  Questions answered: {n_questions}")
                print(f"  Total claims made: {total_claims}")
                print(f"  ---")
                print(f"  Adjudicable (testable): {total_testable}")
                print(f"  Unadjudicable (no evidence): {total_unsupported}")
                print(f"  ---")
                print(f"  Contradicted (hallucinations): {total_contradicted}")
                print(f"  Supported (validated): {total_supported}")
                print(f"  ---")
                print(f"  >>> TESTABILITY RATE: {testability:.1f}% <<<")
                print(f"  >>> HALLUCINATION RATE: {hallucin_rate:.1f}% <<<")

        print(f"\n\nResults saved to: {output_path}")

    print("\n[OK] Evaluation complete")


if __name__ == "__main__":
    main()
