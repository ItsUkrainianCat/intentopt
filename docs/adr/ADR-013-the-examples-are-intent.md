# ADR-013: The user's examples are part of the intent the rewrites may learn from

- **Status:** Accepted 2026-10-06 (from the benches: without a reference the best candidate got 4 wins, 9 ties and 7 losses against the original over 20 prompts; the references the user gives were ignored when writing rewrites)
- **Requirement(s):** R5, R6, R7, R11, R16, R25

## Context

GEPA's gains come from reflecting on failures against a reference and writing what was learned into the prompt. A prompt such as "Assign a priority to this support ticket." hides a policy (P1 to P4, and when) that only the user's examples show. Until now the rewrites were written from the prompt alone, the meaning check (R6) vetoes added requirements, and the length cap (R7) leaves a 9-token prompt 40 tokens of room, so a learned policy could neither be written nor pass.

## Decision

- **Training and held-out are separate.** The examples given with `--examples` split as today: the pick examples (the first ones) are the training set, the later ones are held out (stage E and the bench's `eval_from`). Only pick examples are ever shown to a generating call.
- **Intake and rewrites see the pick examples** when every example has `expected` or `criteria` (R25 reference mode): up to 6 pairs of input and reference, sent as data. The intake records `from_examples`: the labels, rules, decisions and output format the examples demonstrate that the prompt leaves unsaid. A rewrite is asked to state those as instructions in the author's own voice, with a new strategy `induce`; it must not copy an example's input or reference sentences, and must not add a fact the examples do not show.
- **The meaning check accepts them.** `from_examples` items are part of the user's intent: `no-new-goal` passes a requirement that is one of them or follows from them; any other new topic, fact or requirement still fails.
- **Amendment 2026-10-07 (WP24, from the first reflective run):** a rule the reflection learns from failures is not in `from_examples`, and the meaning check vetoed the only second-generation lesson of the refund run (a return-window clause). In reference mode the contract check therefore also receives the pick examples (input and reference) and `no-new-goal` passes a requirement that at least one of them supports; the check must quote the supporting example's input. A requirement that no shown example supports still fails. Held-out examples are never shown to the check.
- **The length cap grows with the references:** in reference mode the cap is the larger of the strictness cap and the original plus 150 tokens, unless the strictness is `conservative`.
- **A free gate against memorising:** a rewrite that contains an example's input verbatim (20 or more characters) is dropped.
- The decision stays the reference-scored one of ADR-012's exception: the held-out references, not a judge's taste, confirm that the learned rules help.

## Consequences

- A prompt whose quality is its hidden policy can now show a measurable, verified gain. Prompts without references are unchanged.
- The user's examples reach the generating model; they were already sent to task and judge calls, so the data flow does not widen (R19 stands).
- The meaning check gets one more input; an example set that contradicts the prompt (a label the prompt forbids) is not resolved by the tool: the rewrite keeps the prompt's own literals (R9).
- Overfitting to few examples is bounded by the held-out check, not removed: a checked result says how many held-out examples it was confirmed on.

## Alternatives rejected

- **Few-shot examples pasted into the returned prompt:** exceeds the cap, leaks the user's data into a prompt that is then shared, and tests memorisation, not the rule.
- **Learning only in the deep tier:** the 20-minute search is the one place that already reflects on references; the point of the fast tiers is the same lift in a minute.
- **Letting the rewrite call infer the policy from the prompt alone:** it cannot know a hidden policy.
