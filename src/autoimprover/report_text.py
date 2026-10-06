"""The fixed lines of the report that say what a result means (SPEC R2, R3, R11, R13, R25), kept out
of `report.py`, which re-exports REASON_LINES and FAST_REASON_LINES; the reference-scored ones are
in `reference_text.py`. Constants only, with no imports."""

# What each reason code means for the user, and what to do about it (SPEC R2, R3, R11, R13).
REASON_LINES = {
    "improved": "a rewrite scored higher than the original on scenarios the search never saw, "
    "by more than the measured noise",
    "no_reliable_improvement": "no rewrite beat the original by more than the measured noise, so "
    "the original is kept; this is the normal result for a prompt that already works",
    "already_strong": "the original already passes nearly every check on the held-out scenarios, "
    "so no search ran; high scores can also mean weak checks, so read the checks below",
    "no_holdout": "with fewer than 8 scenarios there is nothing held out to verify a result on, "
    "so the original is kept; give 8 or more examples with --examples, or pass --trust-search "
    "to accept an unverified result",
    "no_candidate_beat_seed": "no rewrite beat the original on the scenarios the search used "
    "(--trust-search, no holdout), so the original is kept",
    "unconfirmed_out_of_budget": "the calls or the clock ran out before a rewrite was confirmed "
    "on the holdout, so the original is kept; a larger --budget leaves more for the final steps",
    "ungated_best_candidate": "with --ungated the best-ranked rewrite that kept the intent "
    "contract and passed the free gates is returned whether or not it beat the original; no win "
    "over the original was shown, so it is not verified and may be worse (a measuring aid)",
}
# The same for the fast pipeline, which has stages, not a search (SPEC R25).
FAST_REASON_LINES = {
    "no_reliable_improvement": "no rewrite kept the contract and was preferred over the original "
    "clearly enough on the scenarios, so the original is kept; this is the normal result for a "
    "prompt that already works",
    "no_holdout": "the checked tier holds out the examples after the ones it picks on, and none "
    "were left, so the original is kept; give more examples, or a shorter --time",
    "unconfirmed_out_of_budget": "the clock or the calls ran out before a rewrite passed every "
    "gate, so the original is kept; a longer --time leaves more room",
}
FAST_VERIFIED = (
    "a rewrite beat the original on held-out scenarios, on the target model; no noise was "
    "measured, so a small margin is a weak signal"
)
FAST_UNVERIFIED = (
    "a rewrite kept the intent contract and passed the free gates in a short run; it is not "
    "verified on held-out scenarios and no noise was measured, so read it before you use it"
)
FAST_UNVERIFIED_NOISE = (
    "a rewrite kept the intent contract, passed the free gates and was preferred over the "
    "original by a judge on more of the few scenarios it was picked on than it lost, by more than "
    "the noise of two runs of the original; it is not verified on held-out scenarios, so read it "
    "before you use it"
)
