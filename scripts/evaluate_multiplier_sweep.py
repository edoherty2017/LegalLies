#!/usr/bin/env python3
"""
Sweep steering multiplier across [-3, -2, -1, 1, 2, 3] and evaluate hallucination rates.

Tests how steering strength affects hallucination performance.
Negative multipliers steer TOWARD hallucination (control).
Positive multipliers steer AWAY from hallucination.
"""

import sys
import time
import gc
from pathlib import Path

import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import DetectorConfig
from src.detection.unified_detector import UnifiedHallucinationDetector
from src.generation.llama_generator import LlamaGenerator, GenerationConfig
from src.generation.steered_generator import SteeredLlamaGenerator, SteeredRAGGenerator

# 15 lawyer-annotated test questions
TEST_QUESTIONS = [
    'Does the inevitable discovery doctrine create a per se exception to the exclusionary rule for evidence seized after a Fourth Amendment "knock and announce" violation?',
    'Is it the case that a claimed invention cannot be held "obvious", and thus unpatentable under 35 U.S.C. Section 103(a), in the absence of some specific, proven "teaching, suggestion, or motivation" that would have led a person of ordinary skill in the art to combine the relevant prior art teachings in the manner claimed?',
    'Did Congress divest the federal district courts of their federal-question jurisdiction under 28 U.S.C. Section 1331 over private actions brought under the Telephone Consumer Protection Act?',
    'Does imposing failure-to-train liability on a district attorney\'s office for a single Brady violation contravene the rigorous culpability and causation standards of Canton and Bryan County?',
    'Should the Federal Communications Commission\'s determination that the broadcast of vulgar expletives may violate federal restrictions on the broadcast of "any obscene, indecent, or profane language," 18 U.S.C. 1464, when the expletives are not repeated, be stricken down?',
    'Does the Equal Protection Clause preclude a local taxing authority from refusing to refund payments made by those who have paid their assessments in full, even though they forgave the obligations of identically situated taxpayers who chose to pay over a multi-year installment plan?',
    'Is a habeas petitioner\'s claim considered "adjudicated on the merits" for purposes of 28 U.S.C. Section 2254(d) where the state court denied relief in an explained decision but did not expressly acknowledge a federal-law basis for the claim?',
    'Are the deadlines established by the bankruptcy rules for objecting to discharge jurisdictional?',
    'Bankruptcy Rules permit discharge of a student loan only through an adversary proceeding, commenced by filing a complaint and serving it and a summons on an appropriate agent of the creditor. Instead, debtor merely included a declaration of discharge in his Chapter 13 plan and mailed it to creditor\'s post office box. Does such procedure meet the rigorous demands of due process and entitle the resulting orders to respect under principles of res judicata?',
    'Must a district court admit "me, too" evidence -- i.e., testimony, by nonparties, alleging discrimination at the hands of persons who played no role in the adverse employment decision challenged by the plaintiff?',
    'In Phillips v. Washington Legal Found., 524 U.S. 156 (1998), the Supreme Court held that the interest on clients\' funds held in IOLTA accounts was the property of the clients. Does a regulatory scheme for funding state legal services by systematically seizing this property violate the Takings Clause of the Fifth Amendment?',
    'Do 28 U.S.C. Sections 1346(b) and 2680(h) waive the sovereign immunity of the United States for the intentional torts of prison guards when they are acting within the scope of their employment but are not exercising their authority to execute searches, to seize evidence, or to make arrests for violations of federal law?',
    'A department of corrections revised its prison visitation policy to prohibit visits by minor children unless immediate family, require accompanying parent/guardian, prohibit former inmates from visiting, and impose a two-year visitation ban for substance abuse violations. Do these policies violate prisoners\' First and Fourteenth Amendment rights?',
    'Does a defendant who has fraudulently obtained a loan and thus owes restitution for the loan under 18 U.S.C. Section 3663A(b)(1)(B) return "any part" of the loan money by giving the lenders the collateral that secures the money?',
    'Does a case become moot when the lone plaintiff receives an offer from the defendants to satisfy all of the plaintiff\'s claims?',
]

STEERING_VECTOR_PATH = "steering_materials/Vectors/llama3.1-8b_hallu_steering_layer_15.pt"

# Multipliers to test: -3, -1, 1, 3
MULTIPLIERS = [-3.0, -1.0, 1.0, 3.0]


def clear_gpu():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def evaluate_multiplier(multiplier: float, detector: UnifiedHallucinationDetector) -> dict:
    """Evaluate a single multiplier value."""
    print(f"\n{'='*60}")
    print(f"MULTIPLIER: {multiplier}")
    print(f"{'='*60}")

    # Load RAG + Steering generator
    print(f"Loading RAG + Steering generator (multiplier={multiplier})...")
    generator = SteeredRAGGenerator(
        steering_vector_path=STEERING_VECTOR_PATH,
        index_path="data/indexes/oracle_corpus",
        multiplier=multiplier,
        top_k=3,
    )

    results = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/15] {question[:70]}...")

        # Generate
        start = time.time()
        result = generator.generate(question)
        gen_time = (time.time() - start) * 1000
        answer = result.generated_answer

        print(f"  Generated ({gen_time:.0f}ms): {answer[:80]}...")

        # Detect
        detection = detector.detect(query=question, answer=answer)

        print(f"  Testability: {detection.testability_rate:.1%} ({detection.num_testable}/{detection.num_claims})")
        hr = detection.hallucination_rate
        hr_s = "n/a (nothing testable)" if hr is None else f"{hr:.1%}"
        print(f"  Hallucination: {hr_s} ({detection.num_contradicted}/{detection.num_testable})")

        results.append({
            "multiplier": multiplier,
            "question_idx": i,
            "num_claims": detection.num_claims,
            "num_testable": detection.num_testable,
            "num_contradicted": detection.num_contradicted,
            "num_supported": detection.num_supported,
            "num_unsupported": detection.num_unsupported,
            "hallucination_rate": detection.hallucination_rate,
            "testability_rate": detection.testability_rate,
        })

    # Cleanup
    del generator
    clear_gpu()

    return results


def main():
    print("="*70)
    print("RAG + STEERING MULTIPLIER SWEEP: [-3, -1, 1, 3]")
    print("="*70)
    print(f"\nMultipliers: {MULTIPLIERS}")
    print(f"Questions: {len(TEST_QUESTIONS)}")
    print(f"Steering vector: {STEERING_VECTOR_PATH}")

    # Initialize detector (reuse across all multipliers)
    print("\nInitializing detector...")
    config = DetectorConfig(device="cuda")
    detector = UnifiedHallucinationDetector(config)

    all_results = []

    for multiplier in MULTIPLIERS:
        results = evaluate_multiplier(multiplier, detector)
        all_results.extend(results)

        # Save intermediate results
        df = pd.DataFrame(all_results)
        df.to_csv("data/multiplier_sweep_results.csv", index=False)
        print(f"\n[Saved intermediate results to data/multiplier_sweep_results.csv]")

    # Final summary
    df = pd.DataFrame(all_results)

    print("\n" + "="*80)
    print("MULTIPLIER SWEEP RESULTS")
    print("="*80)
    print("\nDual Metrics by Multiplier:")
    print("-"*80)
    print(f"{'Multiplier':>10} {'Claims':>8} {'Testable':>9} {'Contra':>8} {'Supp':>6} {'Testabil%':>10} {'Halluc%':>10}")
    print("-"*80)

    for mult in MULTIPLIERS:
        m = df[df["multiplier"] == mult]
        total_claims = int(m["num_claims"].sum())
        total_testable = int(m["num_testable"].sum())
        total_contra = int(m["num_contradicted"].sum())
        total_supp = int(m["num_supported"].sum())
        testability = (total_testable / total_claims * 100) if total_claims > 0 else 0
        halluc = (total_contra / total_testable * 100) if total_testable > 0 else 0

        print(f"{mult:>10.1f} {total_claims:>8} {total_testable:>9} {total_contra:>8} {total_supp:>6} {testability:>9.1f}% {halluc:>9.1f}%")

    print("-"*80)

    # Find best multiplier
    summary = []
    for mult in MULTIPLIERS:
        m = df[df["multiplier"] == mult]
        total_testable = int(m["num_testable"].sum())
        total_contra = int(m["num_contradicted"].sum())
        halluc = (total_contra / total_testable * 100) if total_testable > 0 else 0
        summary.append({"multiplier": mult, "hallucination_rate": halluc})

    summary_df = pd.DataFrame(summary)
    best = summary_df.loc[summary_df["hallucination_rate"].idxmin()]

    print(f"\nBEST MULTIPLIER: {best['multiplier']:.1f} (Hallucination Rate: {best['hallucination_rate']:.1f}%)")

    # Save final results
    df.to_csv("data/multiplier_sweep_results.csv", index=False)
    print(f"\nResults saved to: data/multiplier_sweep_results.csv")

    print("\n[OK] Multiplier sweep complete!")


if __name__ == "__main__":
    main()
