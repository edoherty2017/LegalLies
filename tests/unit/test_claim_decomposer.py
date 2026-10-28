"""
Unit tests for ClaimDecomposer segmentation.

These cover the two defects found in external review of the detection code:

1. Citations were split into fragments. "U.S." was masked before "U.S.C.", so
   "U.S.C." became "<US>C." and the period after "C." stayed exposed, splitting
   every U.S. Code citation mid-claim. "S. Ct.", "F. Supp.", "No." and "e.g."
   were not protected at all.
2. Compound-sentence splitting fired on any "<word> and the <word>", including
   ordinary noun-phrase coordination ("the landlord and the tenant"), producing
   fragments that are not propositions.

Fragments are not verifiable claims: they go to NLI, come back mostly
UNSUPPORTED, and so inflate claim counts while depressing the testability rate.

The regex fallback is exercised directly (no spaCy model required) so these tests
run in any environment.
"""

import pytest

from src.preprocessing.claim_decomposer import ClaimDecomposer


def segment(text: str):
    """Exercise the dependency-free fallback segmenter."""
    return ClaimDecomposer._segment_sentences(text)


class TestCitationsStayIntact:
    """A citation must never create a sentence boundary."""

    @pytest.mark.parametrize(
        "text",
        [
            "Under 42 U.S.C. Section 1983, a plaintiff may sue state officials.",
            "The court applied 42 U.S.C. 1983 to the facts.",
            "The holding appears at 123 S. Ct. 456 in the majority opinion.",
            "The district court opinion is reported at 500 F. Supp. 2d 100.",
            "The panel relied on 410 U.S. 113 for that proposition.",
            "See 29 C.F.R. 1630.2 for the regulatory definition.",
            "The rule is codified at 5 U.S.C. 552, e.g. the FOIA exemptions.",
            "Docket No. 23-451 was consolidated with the companion case.",
            "The Ninth Circuit so held in Jones v. Smith, 800 F.3d 1 (9th Cir. 2015).",
        ],
    )
    def test_single_sentence_is_not_split(self, text: str) -> None:
        assert segment(text) == [text], f"citation split into fragments: {segment(text)}"

    def test_usc_citation_keeps_section_number_attached(self) -> None:
        text = "Under 42 U.S.C. Section 1983, a plaintiff may sue state officials."
        (only,) = segment(text)
        assert "U.S.C." in only
        assert "1983" in only

    def test_reporter_abbreviation_is_not_mangled(self) -> None:
        """Regression: masking "U.S." first used to leave "<US>C." behind."""
        (only,) = segment("The statute is 42 U.S.C. 1983 today.")
        assert "<US>" not in only and "<USC>" not in only
        assert only == "The statute is 42 U.S.C. 1983 today."


class TestRealBoundariesStillSplit:
    """Fixing over-splitting must not cause under-splitting."""

    def test_two_plain_sentences(self) -> None:
        text = "Qualified immunity protects officials. It does not protect municipalities."
        assert segment(text) == [
            "Qualified immunity protects officials.",
            "It does not protect municipalities.",
        ]

    def test_sentence_after_a_citation_still_splits(self) -> None:
        text = "The rule appears at 500 F. Supp. 2d 100. The panel disagreed."
        assert segment(text) == [
            "The rule appears at 500 F. Supp. 2d 100.",
            "The panel disagreed.",
        ]

    def test_question_and_exclamation_boundaries(self) -> None:
        assert len(segment("Is the claim testable? The court said yes.")) == 2


class TestCompoundSplitIsConservative:
    """Noun-phrase coordination must not be split into non-propositions."""

    @pytest.mark.parametrize(
        "text",
        [
            "The dispute arose between the landlord and the tenant over the lease.",
            "The court and the jury reached different conclusions about intent.",
            "The plaintiff sued the officer and the municipality in federal court.",
        ],
    )
    def test_noun_coordination_is_not_split(self, text: str) -> None:
        parts = ClaimDecomposer._split_compound_sentence(text)
        assert parts == [text], f"noun phrase split into fragments: {parts}"

    def test_no_fragment_loses_its_determiner(self) -> None:
        """Regression: the old split consumed "the", yielding "tenant over the lease."."""
        text = "The dispute arose between the landlord and the tenant over the lease."
        for part in ClaimDecomposer._split_compound_sentence(text):
            assert not part.startswith("tenant")

    def test_every_returned_part_is_nonempty(self) -> None:
        text = "The statute applies and the regulation does not apply here."
        for part in ClaimDecomposer._split_compound_sentence(text):
            assert part.strip()


class TestEndToEndClaimCounts:
    """The decomposer's claim count drives testability, so pin it down."""

    def test_citation_heavy_answer_yields_one_claim_per_sentence(self) -> None:
        answer = (
            "Under 42 U.S.C. Section 1983, a plaintiff may sue state officials. "
            "Qualified immunity shields them unless the right was clearly established."
        )
        claims = ClaimDecomposer._create_claims(segment(answer))
        assert len(claims) == 2, [c.claim_text for c in claims]
        assert all(len(c.claim_text) > 10 for c in claims)
