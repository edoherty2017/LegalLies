#!/usr/bin/env python3
"""
Run ONLY the Steering and RAG + Steering variants.
Appends results to existing evaluation_results.csv.
"""

from pathlib import Path
import sys
from typing import List, Dict
import pandas as pd
import time
import torch
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import DetectorConfig
from src.detection.unified_detector import UnifiedHallucinationDetector
from src.generation import (
    create_steered_llama_generator,
    create_steered_rag_generator,
)

# Same 15 questions
TEST_QUESTIONS = [
    'Does the inevitable discovery doctrine create a per se exception to the exclusionary rule for evidence seized after a Fourth Amendment "knock and announce" violation or is evidence subject to suppression after such violations?',
    'Is it the case that a claimed invention cannot be held "obvious", and thus unpatentable under 35 U.S.C. Section 103(a), in the absence of some specific, proven "teaching, suggestion, or motivation" that would have led a person of ordinary skill in the art to combine the relevant prior art teachings in the manner claimed?',
    'Did Congress divest the federal district courts of their federal-question jurisdiction under 28 U.S.C. Section 1331 over private actions brought under the Telephone Consumer Protection Act?',
    'Does imposing failure-to-train liability on a district attorney\'s office for a single Brady violation contravene the rigorous culpability and causation standards of Canton and Bryan County?',
    'Should the Federal Communications Commission\'s determination that the broadcast of vulgar expletives may violate federal restrictions on the broadcast of "any obscene, indecent, or profane language," 18 U.S.C. 1464, when the expletives are not repeated, be stricken down?',
    'Does the Equal Protection Clause preclude a local taxing authority from refusing to refund payments made by those who have paid their assessments in full, even though they forgave the obligations of identically situated taxpayers who chose to pay over a multi-year installment plan?',
    'Is a habeas petitioner\'s claim considered "adjudicated on the merits" for purposes of 28 U.S.C. Section 2254(d) where the state court denied relief in an explained decision but did not expressly acknowledge a federal-law basis for the claim?',
    'The deadlines for objecting to a debtor\'s discharge in bankruptcy are set by Bankruptcy Rule 4004. Are the deadlines established by the bankruptcy rules for objecting to discharge jurisdictional?',
    'Bankruptcy Rules permit discharge of a student loan only through an adversary proceeding, commenced by filing a complaint and serving it and a summons on an appropriate agent of the creditor. Instead, debtor merely included a declaration of discharge in his Chapter 13 plan and mailed it to creditor\'s post office box. Does such procedure meet the rigorous demands of due process and entitle the resulting orders to respect under principles of res judicata?',
    'Must a district court admit "me, too" evidence -- i.e., testimony, by nonparties, alleging discrimination at the hands of persons who played no role in the adverse employment decision challenged by the plaintiff?',
    'In Phillips v. Washington Legal Found., 524 U.S. 156 (1998), the Supreme Court held that the interest on clients\' funds held in IOLTA accounts was the property of the clients. Does a regulatory scheme for funding state legal services by systematically seizing this property violate the Takings Clause of the Fifth Amendment?',
    'Do 28 U.S.C. Sections 1346(b) and 2680(h) waive the sovereign immunity of the United States for the intentional torts of prison guards when they are acting within the scope of their employment but are not exercising their authority to execute searches, to seize evidence, or to make arrests for violations of federal law?',
    'A department of corrections revised its prison visitation policy to prohibit visits by minor children unless immediate family, require accompanying parent/guardian, prohibit former inmates from visiting, and impose a two-year visitation ban for substance abuse violations. Do these policies violate prisoners\' First and Fourteenth Amendment rights?',
    'Does a defendant who has fraudulently obtained a loan and thus owes restitution for the loan under 18 U.S.C. Section 3663A(b)(1)(B) return "any part" of the loan money by giving the lenders the collateral that secures the money?',
    'Does a case become moot, and thus beyond the judicial power of Article III, when the lone plaintiff receives an offer from the defendants to satisfy all of the plaintiff\'s claims?',
]

STEERING_VECTOR_PATH = "steering_materials/Vectors/llama3.1-8b_hallu_steering_layer_15.pt"


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("[OK] GPU memory cleared")


def evaluate_variant(variant_name, generator, detector, questions):
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {variant_name}")
    print("=" * 60)

    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question[:80]}...")

        print("  Generating answer...")
        start_time = time.time()
        gen_result = generator.generate(question)
        gen_latency = (time.time() - start_time) * 1000

        answer = gen_result.generated_answer
        retrieved_context = gen_result.retrieved_context

        display_answer = answer[:150] + "..." if len(answer) > 150 else answer
        print(f"  Answer: {display_answer}")
        print(f"  Generation time: {gen_latency:.1f}ms")

        print("  Running hallucination detection...")
        detection_result = detector.detect(
            query=question,
            answer=answer,
            evidence=retrieved_context,
        )

        print(f"  Hallucination: {detection_result.is_hallucinated}")
        print(f"  Contradiction rate: {detection_result.contradiction_rate:.3f}")
        print(f"  Support rate: {detection_result.support_rate:.3f}")
        print(f"  Detection time: {detection_result.latency_ms:.1f}ms")

        result = {
            "variant": variant_name,
            "question": question,
            "answer": answer,
            "has_sources": retrieved_context is not None,
            "num_sources": len(retrieved_context) if retrieved_context else 0,
            "generation_latency_ms": gen_latency,
            "is_hallucinated": detection_result.is_hallucinated,
            "contradiction_rate": detection_result.contradiction_rate,
            "support_rate": detection_result.support_rate,
            "unsupported_rate": detection_result.unsupported_rate,
            "num_claims": detection_result.num_claims,
            "detection_latency_ms": detection_result.latency_ms,
            "total_latency_ms": gen_latency + detection_result.latency_ms,
        }
        results.append(result)

    return results


def main():
    print("=" * 60)
    print("LegalLies: Running REMAINING Variants (Steering + RAG+Steering)")
    print("=" * 60)

    index_path = Path("data/indexes/oracle_corpus")
    steering_path = Path(STEERING_VECTOR_PATH)
    output_path = Path("data/evaluation_results.csv")

    # Load existing results
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        all_results = existing_df.to_dict('records')
        print(f"[OK] Loaded {len(all_results)} existing results")
    else:
        all_results = []

    # Initialize detector
    print("\nInitializing hallucination detector...")
    config = DetectorConfig(device="cuda")
    detector = UnifiedHallucinationDetector(config)

    def save_results(results_list):
        if results_list:
            df = pd.DataFrame(results_list)
            df.to_csv(output_path, index=False)
            print(f"[OK] Results saved to: {output_path}")

    # ========================================
    # Variant 3: Steering
    # ========================================
    try:
        print("\n\n" + "=" * 60)
        print("VARIANT 3: Steering")
        print("=" * 60)
        print("Loading Llama3.1-8B + Steering...")
        generator = create_steered_llama_generator(
            steering_vector_path=STEERING_VECTOR_PATH,
            multiplier=1.0,
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
    # Variant 4: RAG + Steering
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
    # Summary
    # ========================================
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_path, index=False)
        print(f"\n\n{'=' * 70}")
        print("FINAL RESULTS (Claim-Level Analysis)")
        print("=" * 70)
        print("\nHallucination Rate = Contradicted / (Contradicted + Supported)")

        print("\n" + "-" * 70)
        print(f"{'Variant':<18} {'Claims':>7} {'Testable':>9} {'Halluci':>8} {'Support':>8} {'Halluc %':>9}")
        print("-" * 70)

        for variant in ["Baseline", "RAG", "Steering", "RAG + Steering"]:
            if variant in df["variant"].values:
                v = df[df["variant"] == variant]
                total_claims = int(v["num_claims"].sum())
                total_contradicted = sum(v["contradiction_rate"] * v["num_claims"])
                total_supported = sum(v["support_rate"] * v["num_claims"])
                testable = total_contradicted + total_supported
                hallucin_rate = (total_contradicted / testable * 100) if testable > 0 else 0
                print(f"{variant:<18} {total_claims:>7} {int(testable):>9} {int(total_contradicted):>8} {int(total_supported):>8} {hallucin_rate:>8.1f}%")

        print("-" * 70)
        print(f"\nResults saved to: {output_path}")

    print("\n[OK] Evaluation complete")


if __name__ == "__main__":
    main()
