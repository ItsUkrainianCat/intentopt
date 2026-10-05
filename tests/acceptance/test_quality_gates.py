"""Acceptance tests for the gates every answer passes: SPEC R3 (beat the noise on the holdout), R6
(contract check), R7 (length cap), R9 (literals kept), R11 (no holdout below 8 scenarios), R13
(already strong) and R14a (confirmed on the target model).

Every test that shows a candidate is NOT returned has a twin built the same way in which one IS
returned, so it cannot pass because the scripted replies were unusable (ARCHITECTURE section 3).
"""

import json
from collections import Counter

import pytest
from fakes import MARKER, happy_backend, judge_reply

from autoimprover.types import Call

ORIGINAL = "Summarise the meeting notes for the team in five bullet points."
IMPROVED = f"{ORIGINAL} {MARKER} Keep each bullet short."
PLAIN = f"{ORIGINAL} Keep each bullet short."
TASK_MODEL, TARGET = "claude-haiku-4-5-20251001", "claude-sonnet-5-5"


def _good(call: Call) -> bool:
    return MARKER in call.system + call.user


def _scoring_judge(call: Call) -> str:
    return judge_reply(call, lambda _s, _c, output: output.startswith("GOOD"))


def _contract_calls(calls: list[Call]) -> list[dict]:
    items = []
    for call in calls:
        if call.role == "judge":
            items += [i for i in json.loads(call.user)["scenarios"] if i["scenario"] == "contract"]
    return items


@pytest.mark.parametrize(
    "proposal, status", [(PLAIN, "unchanged"), (IMPROVED, "improved")], ids=["no-gain", "twin"]
)
def test_original_returned_unless_a_candidate_beats_it(run_cli, proposal, status):
    """R3: if no candidate beats the original on the holdout, the original is returned unchanged,
    the report says "no reliable improvement" and the exit code is 0. Twin: a better one is
    returned."""
    r = run_cli([ORIGINAL], happy_backend(proposal))
    assert r.code == 0, r.err
    if status == "unchanged":
        assert r.out == ORIGINAL + "\n"
        assert "no reliable improvement" in r.err
    else:
        assert r.out == IMPROVED + "\n"
        assert "no reliable improvement" not in r.err
    obj = run_cli(["--json", ORIGINAL], happy_backend(proposal)).json()
    assert obj["status"] == status
    assert ("no reliable improvement" in obj["reason"]) is (status == "unchanged")


@pytest.mark.parametrize(
    "seed2_good, candidate_good, status",
    [(1, 3, "unchanged"), (1, 4, "improved"), (2, 3, "improved")],
    ids=["gain-inside-noise", "twin-gain-above-noise", "twin-same-gain-no-noise"],
)
def test_win_threshold_is_twice_the_seed_noise(
    run_cli, override, seed2_good, candidate_good, status
):
    """R3, R12, R14a: on the 4 holdout scenarios, on the target model, seed run 1 passes 2 and seed
    run 2 passes `seed2_good`; the bar is max(0.05, 2 x |difference|). A candidate passing 3 of 4
    does not clear 0.375 + 0.5 but does clear 0.5 + 0.05; one passing 4 clears both."""
    served: Counter = Counter()

    def task(call: Call) -> str:
        if call.model != TARGET:
            return "GOOD answer" if _good(call) else "BAD answer"
        key = ("candidate",) if _good(call) else ("seed", call.sample)
        served[key] += 1
        limit = {("seed", 0): 2, ("seed", 1): seed2_good, ("candidate",): candidate_good}[key]
        return "GOOD answer" if served[key] <= limit else "BAD answer"

    r = run_cli(["--json", ORIGINAL], override(happy_backend(IMPROVED), task=task))
    assert r.code == 0, r.err
    obj = r.json()
    assert served[("candidate",)] == 4  # the finalist was run on the target model's holdout
    noise = abs(0.5 - seed2_good / 4)
    assert obj["noise"] == pytest.approx(noise)
    assert obj["status"] == status
    if status == "unchanged":
        assert obj["prompt"] == ORIGINAL
        assert obj["reason_code"] == "no_reliable_improvement"
    else:
        before = (0.5 + seed2_good / 4) / 2
        assert obj["score_before"] == pytest.approx(before)
        assert obj["score_after"] == pytest.approx(candidate_good / 4)
        threshold = max(0.05, 2 * noise)
        assert obj["margin"] == pytest.approx(candidate_good / 4 - before - threshold)


def _contract_judge(verdict: str):
    """Grade scoring batches like happy_backend; answer the R6 contract check with `verdict` on
    its last question: pass, fail, omit, or a pass whose quote is not in the candidate."""

    def answer(call: Call) -> str:
        request = json.loads(call.user)
        item = request["scenarios"][0]
        if item["scenario"] != "contract":
            return _scoring_judge(call)
        checks = [
            {"id": c["id"], "pass": True, "quote": item["output"][:20]} for c in item["checks"]
        ]
        if verdict == "fail-all":
            for check in checks:
                check["pass"] = False
        elif verdict == "fail":
            checks[-1]["pass"] = False
        elif verdict == "omit":
            checks.pop()
        elif verdict == "invented-quote":
            checks[-1]["quote"] = "a sentence the candidate never says"
        return json.dumps({"results": [{"scenario": "contract", "checks": checks}]})

    return answer


@pytest.mark.parametrize("verdict", ["fail-all", "fail", "omit", "invented-quote", "pass"])
def test_candidate_failing_the_contract_check_is_never_returned(run_cli, override, verdict):
    """R6, R10b: the answer must pass the contract check (judged with the original as input and the
    candidate as output, ADR-008); a failed question, an unanswered one or a pass whose quote is not
    in the candidate is a violation and the candidate is never returned. Twin: "pass"."""
    backend = override(happy_backend(IMPROVED), judge=_contract_judge(verdict))
    r = run_cli(["--json", ORIGINAL], backend)
    assert r.code == 0, r.err
    obj = r.json()
    contract = _contract_calls(backend.calls)
    assert contract, "the contract check was never asked"
    assert all(i["input"] == ORIGINAL and i["output"] == IMPROVED for i in contract)
    if verdict == "pass":
        assert obj["prompt"] == IMPROVED
        assert obj["status"] == "improved"
    else:
        assert obj["prompt"] == ORIGINAL
        assert obj["status"] == "unchanged"
        assert MARKER not in r.out


LITERALS = (
    "Reply to {customer_name} with the template at /srv/app/templates/reply.md and link "
    'https://example.com/help/returns. Open with "Thank you for waiting". Example:\n'
    '```python\nprint("ok")\n```'
)
MUTATIONS = {
    "placeholder": ("{customer_name}", "{client_name}"),
    "file-path": ("/srv/app/templates/reply.md", "/srv/app/templates/answer.md"),
    "url": ("https://example.com/help/returns", "https://example.com/help/refunds"),
    "quoted-string": ('"Thank you for waiting"', '"Thanks for waiting"'),
    "code-block": ('print("ok")', 'print("done")'),
    "kept": ("", ""),
}


@pytest.mark.parametrize("literal", list(MUTATIONS))
def test_candidate_that_changes_a_literal_is_never_returned(run_cli, literal):
    """R9, R6: placeholders, code blocks, URLs, file paths and quoted strings of the original appear
    unchanged in the result; a better-scoring candidate that changes one is not returned. Twin:
    "kept" changes none and is returned."""
    old, new = MUTATIONS[literal]
    candidate = f"{LITERALS.replace(old, new) if old else LITERALS}\n{MARKER} Be brief."
    r = run_cli(["--json", LITERALS], happy_backend(candidate))
    assert r.code == 0, r.err
    obj = r.json()
    if literal == "kept":
        assert obj["prompt"] == candidate
    else:
        assert candidate != f"{LITERALS}\n{MARKER} Be brief."
        assert obj["prompt"] == LITERALS
        assert obj["status"] == "unchanged"


LONG = "Review the pull request and list every bug you find in it. " * 20
GROWTH = {  # a candidate about 1.75x the original: only `bold` (2.5x) or --allow-growth admit it
    "conservative": (["--strictness", "conservative"], "unchanged"),
    "balanced": (["--strictness", "balanced"], "unchanged"),
    "bold": (["--strictness", "bold"], "improved"),
    "allow-growth": (["--allow-growth"], "improved"),
}


@pytest.mark.parametrize("level", list(GROWTH))
def test_length_cap_follows_the_strictness_level(run_cli, level):
    """R7, R8: the result is at most 1.25x / 1.5x / 2.5x the original's tokens for conservative /
    balanced / bold; --allow-growth removes the cap; the report states the ratio."""
    extra, status = GROWTH[level]
    candidate = LONG + MARKER + " Explain each bug in one line." * 25
    r = run_cli(["--json", *extra, LONG], happy_backend(candidate))
    assert r.code == 0, r.err
    obj = r.json()
    assert obj["status"] == status
    if status == "improved":
        assert obj["prompt"] == candidate
        assert 1.5 < obj["length_ratio"] < 2.5
    else:
        assert obj["prompt"] == LONG


@pytest.mark.parametrize(
    "extra_words, status", [(8, "improved"), (120, "unchanged")], ids=["a-sentence", "a-page"]
)
def test_short_prompt_may_gain_a_sentence_but_not_a_page(run_cli, extra_words, status):
    """R7: the cap's floor is the original plus 40 tokens, so a short prompt can gain a sentence
    (ratio above 1.25x) at conservative, while a much longer rewrite is still refused."""
    candidate = f"{ORIGINAL} {MARKER} " + " ".join(["detail"] * extra_words)
    r = run_cli(["--json", ORIGINAL], happy_backend(candidate))
    assert r.code == 0, r.err
    obj = r.json()
    assert obj["status"] == status
    if status == "improved":
        assert obj["length_ratio"] > 1.25
    else:
        assert obj["prompt"] == ORIGINAL


def test_fewer_than_8_scenarios_return_the_original_without_a_paid_call(run_cli, examples):
    """R11: with fewer than 8 scenarios there is no holdout; the original is returned (exit 0), the
    report says so, and no model call is made."""
    backend = happy_backend(IMPROVED)
    r = run_cli(["--examples", examples(7), ORIGINAL], backend)
    assert r.code == 0, r.err
    assert r.out == ORIGINAL + "\n"
    assert "holdout" in r.err
    assert backend.calls == []
    obj = run_cli(["--json", "--examples", examples(7), ORIGINAL], backend).json()
    assert obj["prompt"] == ORIGINAL
    assert obj["reason_code"] == "no_holdout"
    assert obj["verified"] is False
    assert backend.calls == []


@pytest.mark.parametrize("n, verified", [(7, False), (8, True)])
def test_trust_search_below_8_returns_an_unverified_win(run_cli, examples, n, verified):
    """R11, R2: with --trust-search and 7 scenarios the best candidate that beats the original on
    the valset is returned, marked unverified, with a loud notice on stderr and no seed or finalist
    run on the target model. Twin: 8 scenarios have a holdout and give a verified result."""
    backend = happy_backend(IMPROVED)
    r = run_cli(["--json", "--trust-search", "--examples", examples(n), ORIGINAL], backend)
    assert r.code == 0, r.err
    obj = r.json()
    assert obj["prompt"] == IMPROVED
    assert obj["verified"] is verified
    target_runs = [c for c in backend.calls if c.role == "task" and c.model == TARGET]
    if verified:
        assert target_runs
        assert "not verified on a holdout" not in r.err.lower()
    else:
        assert target_runs == []
        assert "not verified on a holdout" in r.err.lower()


def test_unverified_result_report_never_claims_a_holdout_check(run_cli, examples):
    """R2, R11: the report says whether the result is holdout-verified; a --trust-search result
    says it is not, and no line of the report claims it was checked on scenarios the search never
    saw (ARCHITECTURE section 8: no row reads as success without being one)."""
    r = run_cli(["--trust-search", "--examples", examples(7), ORIGINAL], happy_backend(IMPROVED))
    assert r.code == 0, r.err
    assert r.out == IMPROVED + "\n"
    assert "not verified on a holdout" in r.err.lower()
    claims = [line for line in r.err.splitlines() if "never saw" in line]
    assert claims == [], claims


def test_trust_search_keeps_the_original_when_nothing_beats_it(run_cli, examples):
    """R11: with --trust-search a candidate is returned only if it beats the original on the
    valset; otherwise the original, with its own reason code."""
    r = run_cli(
        ["--json", "--trust-search", "--examples", examples(7), ORIGINAL], happy_backend(PLAIN)
    )
    assert r.code == 0, r.err
    obj = r.json()
    assert obj["prompt"] == ORIGINAL
    assert obj["reason_code"] == "no_candidate_beat_seed"


@pytest.mark.parametrize("strong", [True, False], ids=["already-strong", "twin-weak-seed"])
def test_strong_original_stops_before_the_search(run_cli, strong):
    """R13: an original scoring at least 0.95 on the holdout stops early, says "already strong,
    relative to the generated checks", lists the checks, and runs no search. Twin: a weak original
    is searched and improved."""
    prompt = f"{ORIGINAL} {MARKER}" if strong else ORIGINAL
    proposal = f"{prompt} {MARKER} Keep each bullet short."
    backend = happy_backend(proposal)
    r = run_cli([prompt], backend)
    assert r.code == 0, r.err
    reflections = backend.count("reflect")
    if strong:
        assert r.out == prompt + "\n"
        assert "already strong, relative to the generated checks" in r.err
        assert "answers the request" in r.err  # the check, listed so the user can judge it
        assert reflections == 0
        assert not [c for c in backend.calls if c.role == "task" and c.model == TASK_MODEL]
        assert run_cli(["--json", prompt], happy_backend(proposal)).json()["stop"] is None
    else:
        assert r.out == proposal + "\n"
        assert reflections > 0


@pytest.mark.parametrize("failed_per_scenario, strong", [(1, True), (2, False)])
def test_already_strong_starts_at_exactly_0_95(
    run_cli, override, examples, failed_per_scenario, strong
):
    """R13 boundary: 20 judged checks per scenario (1 from the contract, 19 criteria); failing 1
    scores exactly 0.95 (already strong), failing 2 scores 0.90 (the search runs)."""
    path = examples(12, criteria=[f"mentions point {i}" for i in range(1, 20)])

    def judge(call: Call) -> str:
        request = json.loads(call.user)
        results = []
        for item in request["scenarios"]:
            n = len(item["checks"])
            checks = [
                {"id": c["id"], "pass": k < n - failed_per_scenario, "quote": item["output"][:20]}
                for k, c in enumerate(item["checks"])
            ]
            results.append({"scenario": item["scenario"], "checks": checks})
        return json.dumps({"results": results})

    backend = override(happy_backend(IMPROVED), judge=judge)
    r = run_cli(["--json", "--examples", path, ORIGINAL], backend)
    assert r.code == 0, r.err
    obj = r.json()
    assert obj["score_before"] == pytest.approx(1 - failed_per_scenario / 20)
    assert (obj["reason_code"] == "already_strong") is strong
    assert (backend.count("reflect") == 0) is strong


def _on_target(outcome: str):
    """Task replies: the search model answers GOOD exactly for MARKER prompts; on the target model
    both seed runs pass 2 of 4 and the candidate ties (2), loses (0) or wins (4)."""
    served: Counter = Counter()

    def task(call: Call) -> str:
        if call.model != TARGET:
            return "GOOD answer" if _good(call) else "BAD answer"
        key = "candidate" if _good(call) else f"seed-{call.sample}"
        served[key] += 1
        limit = {"ties": 2, "loses": 0, "wins": 4}[outcome] if key == "candidate" else 2
        return "GOOD answer" if served[key] <= limit else "BAD answer"

    return served, task


@pytest.mark.parametrize("on_target", ["ties", "loses", "wins"], ids=["tie", "loss", "twin-win"])
def test_search_winner_must_also_win_on_the_target_model(run_cli, override, on_target):
    """R14a: the final comparison runs on the target model for both seed runs and the finalist; a
    candidate that wins on the search model but ties or loses on the target model is not returned.
    Twin: it also wins there, and the report shows the scores on both models."""
    served, task = _on_target(on_target)
    r = run_cli(["--json", ORIGINAL], override(happy_backend(IMPROVED), task=task))
    assert r.code == 0, r.err
    obj = r.json()
    assert served == Counter({"seed-0": 4, "seed-1": 4, "candidate": 4})
    assert obj["search_score_before"] == pytest.approx(0.0)
    if on_target == "wins":
        assert obj["search_score_after"] == pytest.approx(1.0)
        assert obj["prompt"] == IMPROVED
        assert obj["score_before"] == pytest.approx(0.5)
        assert obj["score_after"] == pytest.approx(1.0)
        text = run_cli([ORIGINAL], override(happy_backend(IMPROVED), task=_on_target("wins")[1]))
        scores = [line for line in text.err.splitlines() if "score" in line]
        assert any(TARGET in line for line in scores), scores
        assert any(TASK_MODEL in line for line in scores), scores
    else:
        assert obj["prompt"] == ORIGINAL
        assert obj["status"] == "unchanged"
