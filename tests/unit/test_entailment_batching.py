"""
Prove the batched NLI path is verdict-identical to the sequential one.

verify_claim_against_multiple_sources used to call verify_claim once per
passage, issuing a separate model.predict per source. On CPU those unbatched
forward passes dominate the entire run. It now makes one batched call through
verify_batch, which already existed and is documented as a 10-20x speedup.

That optimisation is only safe if it cannot change a single label or
confidence, which is what these tests check: both paths must apply the same
truncation, the same [CONTRADICTION, ENTAILMENT, NEUTRAL] mapping and the same
thresholds. A stub model stands in for the cross-encoder, so this runs with no
model download and no network.
"""

import numpy as np
import pytest

from src.detection.entailment_verifier import (
    EntailmentLabel,
    EntailmentVerifier,
)

ENTAIL_T = 0.70
CONTRA_T = 0.70


class StubCrossEncoder:
    """Deterministic stand-in for a 3-class NLI cross-encoder."""

    def __init__(self, n_labels: int = 3) -> None:
        self.n_labels = n_labels
        self.calls = 0
        self.max_length = 512

    def predict(self, pairs, apply_softmax: bool = False):
        # One row per pair, derived deterministically from the pair's text so
        # that batching cannot change a row's scores.
        self.calls += 1
        rows = []
        for source, claim in pairs:
            seed = (len(source) * 31 + len(claim) * 17) % 100 / 100.0
            if self.n_labels == 3:
                contradiction = seed
                entailment = 1.0 - seed
                neutral = abs(0.5 - seed)
                total = contradiction + entailment + neutral
                rows.append([contradiction / total, entailment / total, neutral / total])
            else:
                rows.append([1.0 - seed, seed])
        return np.array(rows)


def make_verifier(n_labels: int = 3) -> EntailmentVerifier:
    """
    Build a verifier around the stub, bypassing __init__.

    __init__ now raises when a model cannot be loaded (so a broken environment
    can never silently report a 0% hallucination rate), which is exactly what we
    do not want to trigger in a unit test.
    """
    v = EntailmentVerifier.__new__(EntailmentVerifier)
    v.model = StubCrossEncoder(n_labels=n_labels)
    v.entailment_threshold = ENTAIL_T
    v.contradiction_threshold = CONTRA_T
    return v


PAIRS = [
    ("Qualified immunity protects state officials.", "Officials enjoy qualified immunity."),
    ("Under 42 U.S.C. Section 1983 a plaintiff may sue.", "Section 1983 creates a cause of action."),
    ("Murder is a federal felony in all cases.", "Murder is generally prosecuted by states."),
    ("The statute was enacted in 1871.", "Congress passed the Act in 1871."),
    ("x" * 4000, "y" * 2000),  # exercises truncation on both paths
]


class TestBatchedMatchesSequential:
    def test_same_label_and_confidence_per_pair(self) -> None:
        seq_v = make_verifier()
        batch_v = make_verifier()

        sequential = [seq_v.verify_claim(claim, source) for claim, source in
                      [(c, s) for s, c in PAIRS]]
        batched = batch_v.verify_batch([(c, s) for s, c in PAIRS])

        assert len(sequential) == len(batched) == len(PAIRS)
        for seq, bat in zip(sequential, batched):
            assert seq.label == bat.label, (seq.claim[:40], seq.label, bat.label)
            assert seq.confidence == pytest.approx(bat.confidence, abs=1e-9)
            assert seq.is_supported == bat.is_supported
            assert seq.is_contradicted == bat.is_contradicted

    def test_two_class_model_also_matches(self) -> None:
        """The 2-class fallback in verify_batch was aligned with verify_claim."""
        seq_v = make_verifier(n_labels=2)
        batch_v = make_verifier(n_labels=2)

        sequential = [seq_v.verify_claim(c, s) for s, c in PAIRS]
        batched = batch_v.verify_batch([(c, s) for s, c in PAIRS])

        for seq, bat in zip(sequential, batched):
            assert seq.label == bat.label
            assert seq.confidence == pytest.approx(bat.confidence, abs=1e-9)


class TestMultiSourceUsesOneCall:
    def test_single_predict_call_for_all_sources(self) -> None:
        """The whole point: one forward pass, not one per source."""
        v = make_verifier()
        sources = [s for s, _ in PAIRS]
        v.verify_claim_against_multiple_sources("Some testable legal claim.", sources)
        assert v.model.calls == 1, f"expected 1 batched call, got {v.model.calls}"

    def test_aggregate_matches_sequential_aggregate(self) -> None:
        v = make_verifier()
        sources = [s for s, _ in PAIRS]
        claim = "Qualified immunity shields officials from damages."

        out = v.verify_claim_against_multiple_sources(claim, sources)

        # Recompute the same aggregate the slow way.
        ref = make_verifier()
        manual = [ref.verify_claim(claim, s) for s in sources]
        assert out["supporting_sources"] == sum(1 for r in manual if r.is_supported)
        assert out["contradicting_sources"] == sum(1 for r in manual if r.is_contradicted)
        assert out["num_sources"] == len(sources)
        assert len(out["per_source_results"]) == len(sources)

    def test_source_order_is_preserved(self) -> None:
        v = make_verifier()
        sources = [s for s, _ in PAIRS]
        out = v.verify_claim_against_multiple_sources("A claim about the statute.", sources)
        returned = [r["source"] for r in out["per_source_results"]]
        # EntailmentResult.to_dict truncates the source to 200 chars for
        # readability, so compare against that rather than the full text.
        assert returned == [s[:200] for s in sources]
