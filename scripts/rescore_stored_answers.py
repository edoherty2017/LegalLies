"""
Re-score the stored generated answers with the current detector.

Generation is not repeated. The answers produced for the reported experiment are
stored verbatim in data/evaluation_results.csv (columns: variant, question,
answer), so changing the detector and re-running this script isolates the effect
of the scorer: the generations are held fixed by construction.

Every variant is scored through the same path - auto-retrieval from the oracle
corpus - which is what makes the comparison across variants meaningful. Note that
the CSV does not store the passages the RAG variants were shown at generation
time, so the pre-fix RAG numbers (which graded against those passages) cannot be
reproduced from this file; the Baseline and Steering columns, which already used
auto-retrieval, can be and serve as the fidelity check.

Usage:
    python scripts/rescore_stored_answers.py --out data/rescored.csv
    python scripts/rescore_stored_answers.py --limit 3 --out /tmp/smoke.csv
    python scripts/rescore_stored_answers.py --variants Baseline Steering --out ...
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.core.config import DetectorConfig  # noqa: E402
from src.detection.unified_detector import FactualityDetector  # noqa: E402

FIELDS = [
    "variant",
    "question",
    "answer",
    "num_claims",
    "num_testable",
    "num_contradicted",
    "num_supported",
    "num_unsupported",
    "hallucination_rate",
    "testability_rate",
    "detection_latency_ms",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO / "data" / "evaluation_results.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows")
    ap.add_argument("--variants", nargs="*", default=None)
    args = ap.parse_args()

    rows = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            if args.variants and r["variant"] not in args.variants:
                continue
            if not (r.get("answer") or "").strip():
                continue
            rows.append(r)
    if args.limit:
        rows = rows[: args.limit]

    print(f"Scoring {len(rows)} stored answers from {args.csv}")
    config = DetectorConfig()
    print(f"index: {config.retrieval_index_path}")
    print(f"NLI:   {config.cross_encoder_model}")
    detector = FactualityDetector(config=config)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg = defaultdict(lambda: defaultdict(int))
    t0 = time.time()

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            res = detector.detect(query=r["question"], answer=r["answer"])
            hr = res.hallucination_rate
            w.writerow(
                {
                    "variant": r["variant"],
                    "question": r["question"],
                    "answer": r["answer"],
                    "num_claims": res.num_claims,
                    "num_testable": res.num_testable,
                    "num_contradicted": res.num_contradicted,
                    "num_supported": res.num_supported,
                    "num_unsupported": res.num_unsupported,
                    # may be None once the untestable case is fixed; write blank
                    "hallucination_rate": "" if hr is None else f"{hr:.6f}",
                    "testability_rate": f"{res.testability_rate:.6f}",
                    "detection_latency_ms": f"{res.latency_ms:.1f}",
                }
            )
            fh.flush()
            a = agg[r["variant"]]
            a["claims"] += res.num_claims
            a["con"] += res.num_contradicted
            a["sup"] += res.num_supported
            a["uns"] += res.num_unsupported
            a["n"] += 1
            el = time.time() - t0
            print(
                f"  [{i}/{len(rows)}] {r['variant']:<15} "
                f"claims={res.num_claims:>3} testable={res.num_testable:>3} "
                f"con={res.num_contradicted:>3} ({el/i:.1f}s/row, {el/60:.1f}m total)",
                flush=True,
            )

    print(f"\nWrote {out_path}")
    print(f"\n{'variant':<16}{'n':>4}{'claims':>8}{'testable':>9}{'testability':>12}{'hall.rate':>11}")
    for v in sorted(agg):
        a = agg[v]
        t = a["sup"] + a["con"]
        tr = t / a["claims"] if a["claims"] else 0.0
        hr = a["con"] / t if t else float("nan")
        print(f"{v:<16}{a['n']:>4}{a['claims']:>8}{t:>9}{tr:>11.1%}{hr:>11.1%}")
    print("\n(pooled: sum contradicted / sum testable - the same method the README uses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
