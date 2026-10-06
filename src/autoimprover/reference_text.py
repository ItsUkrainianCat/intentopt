"""The fixed texts of a reference-scored result (SPEC R2, R25; WP21): the reasons a fast or checked
run gives when every example carries a reference and the pick decided by agreement with it, what
the report says they mean, and the evidence line of the plan. Constants only, with no imports; every
reason holds MARK, by which the report knows a reference-scored outcome (the label is built as the
fast label is, in the reason)."""

MARK = "reference-scored"

# The reasons (`fast.py`), after "tier <tier>: ".
FAST_LABEL = (
    "fast check: reference-scored: agreed with the user's reference answers better than the "
    "original on the same few examples it was picked on, noise measured by scoring the original "
    "twice, not verified on held-out examples"
)
NO_WIN = (
    "reference-scored: no rewrite kept the contract and agreed with the reference answers better "
    "than the original, on more examples than it did worse, by more than the noise of the "
    "original's two runs"
)
UNGATED_LABEL = (
    "ungated: the best-ranked candidate; no win over the original was shown (reference-scored)"
)
HELD_WIN = (
    "reference-scored: agrees with the reference answers better than the original on {held}, by "
    "more than the noise of the original's two runs there"
)
HELD_LOSS = (
    "reference-scored: the winner did not agree with the reference answers better than the "
    "original on {held}, by more than the noise of the original's two runs there"
)

# What the report says a result means (`report.py`).
MEANING_UNVERIFIED = (
    "a rewrite kept the intent contract, passed the free gates and agreed with your reference "
    "answers better than the original on the few examples it was picked on, by more than the "
    "noise of two runs of the original; it is not verified on held-out examples, so read it "
    "before you use it"
)
MEANING_VERIFIED = (
    "a rewrite agreed with your reference answers better than the original on held-out examples, "
    "on the target model, by more than the noise of two runs of the original there"
)
MEANING_NO_WIN = (
    "no rewrite kept the contract and agreed with your reference answers clearly better than the "
    "original, so the original is kept; this is the normal result for a prompt that already "
    "works"
)

# The evidence line of a plan with references (`cli_plan.py`), by tier.
EVIDENCE = {
    "fast": "a fast check, reference-scored: agreement with your reference answers on the "
    "examples it is picked on, noise measured by scoring the original twice, not verified on "
    "held-out examples",
    "checked": "reference-scored: agreement with your reference answers, verified on held-out "
    "examples on the target model, noise measured by scoring the original twice",
}
