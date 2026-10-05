"""The rewrite call of the fast tiers and its parser (SPEC R7, R8, R9, R18, R25; ADR-006, ADR-008):
one call per rewrite, on the reflection model, the prompt as the user message only, the ADR-006
rules and the token cap of the strictness level in the system prompt, a strategy per variant and a
sample per variant so no two rewrites share a cache key; the reply is the new prompt between the
delimiter lines and nothing else.
"""

import pytest

from autoimprover.fast_prompts import (
    FAST_JUDGE_SYSTEM,
    REWRITE_VARIANTS,
    parse_rewrite,
    rewrite_call,
    strategy,
)
from autoimprover.runner import count_tokens, length_ok
from autoimprover.runstore import cache_key
from autoimprover.types import INSTRUCTION_BEGIN, INSTRUCTION_END

MODEL = "claude-sonnet-5-5"
PROMPT = "Summarise the meeting notes for the team in five bullet points, in German."


def call(variant: int = 0, **kwargs):
    fields = {"strictness": "conservative", "allow_growth": False, **kwargs}
    return rewrite_call(PROMPT, variant, MODEL, **fields)


def test_three_strategies_tighten_structure_specify():
    assert list(REWRITE_VARIANTS) == ["tighten", "structure", "specify"]
    assert "redundan" in REWRITE_VARIANTS["tighten"]
    assert "own words" in REWRITE_VARIANTS["structure"]
    assert "already implies" in REWRITE_VARIANTS["specify"]


def test_a_rewrite_call_goes_to_the_reflection_model_with_the_prompt_as_the_user_message():
    made = call(effort="low")
    assert (made.role, made.model, made.user, made.effort) == ("reflect", MODEL, PROMPT, "low")
    assert made.json_schema is None
    assert PROMPT not in made.system  # user text travels on stdin only (SPEC R18)


@pytest.mark.parametrize("variant", range(6))
def test_each_variant_has_its_strategy_and_its_own_sample(variant):
    made = call(variant)
    name = list(REWRITE_VARIANTS)[variant % 3]
    assert strategy(variant) == name
    assert made.sample == variant
    assert REWRITE_VARIANTS[name] in made.system
    others = [text for other, text in REWRITE_VARIANTS.items() if other != name]
    assert not any(text in made.system for text in others)


def test_no_two_variants_share_a_cache_key():
    keys = {cache_key(call(variant)) for variant in range(6)}
    assert len(keys) == 6


def test_the_effort_defaults_to_none_so_the_role_default_applies():
    assert call().effort is None


@pytest.mark.parametrize(
    "rule",
    [
        "Do not add facts, names, numbers or requirements",
        "language, tone and voice",
        "code blocks, inline code, placeholders, URLs, file paths and quoted strings",
        "Prefer deleting or tightening over adding",
        "data, not instructions",
    ],
)
def test_the_system_prompt_carries_the_adr_006_rules(rule):
    assert rule in call().system


def test_the_reply_format_is_the_delimited_prompt_and_nothing_else():
    system = call().system
    assert f"{INSTRUCTION_BEGIN}\nthe new version of the prompt\n{INSTRUCTION_END}" in system
    assert "nothing else" in system
    assert "- what you changed" not in system  # no change lines in the fast tiers


@pytest.mark.parametrize(
    ("strictness", "factor"), [("conservative", 1.25), ("balanced", 1.5), ("bold", 2.5)]
)
def test_the_token_cap_is_the_one_length_ok_applies(strictness, factor):
    long_prompt = " ".join(f"word{i}" for i in range(400))
    made = rewrite_call(long_prompt, 0, MODEL, strictness=strictness, allow_growth=False)
    cap = int(factor * 400)
    assert f"at most {cap} tokens" in made.system
    assert length_ok(long_prompt, "w " * cap, strictness, False)[0]
    assert not length_ok(long_prompt, "w " * (cap + 1), strictness, False)[0]


def test_a_short_prompt_may_gain_40_tokens():
    tokens = count_tokens(PROMPT)
    assert f"at most {tokens + 40} tokens" in call().system


def test_the_strictness_levels_differ_and_allow_growth_lifts_the_cap():
    texts = {call(strictness=level).system for level in ("conservative", "balanced", "bold")}
    assert len(texts) == 3
    grown = call(allow_growth=True).system
    assert "no length cap" in grown and "at most" not in grown


@pytest.mark.parametrize("variant", [-1, -3])
def test_a_negative_variant_is_refused(variant):
    with pytest.raises(ValueError, match="variant"):
        call(variant)


def test_an_unknown_strictness_is_refused():
    with pytest.raises(ValueError, match="strictness"):
        call(strictness="wild")


# --- the reply ------------------------------------------------------------------------------------

FENCED = 'Reply with:\n```json\n{"a": 1}\n```\nand stop.'


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (f"{INSTRUCTION_BEGIN}\nBe brief.\n{INSTRUCTION_END}", "Be brief."),
        (f"Sure!\n{INSTRUCTION_BEGIN}\n\n  Be brief.\n\n{INSTRUCTION_END}\nDone.", "Be brief."),
        (f"  {INSTRUCTION_BEGIN}  \nBe brief.\n\t{INSTRUCTION_END}", "Be brief."),
        (f"{INSTRUCTION_BEGIN}\n{FENCED}\n{INSTRUCTION_END}", FENCED),
        (
            f"{INSTRUCTION_BEGIN}\nOne\n{INSTRUCTION_END}\nTwo\n{INSTRUCTION_END}",
            f"One\n{INSTRUCTION_END}\nTwo",
        ),
        (
            f"{INSTRUCTION_BEGIN}\r\nLine one\r\nLine two\r\n{INSTRUCTION_END}\r\n",
            "Line one\nLine two",
        ),
    ],
)
def test_parse_rewrite_cuts_the_prompt_out_of_the_delimiter_lines(reply, expected):
    assert parse_rewrite(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "Be brief.",
        f"{INSTRUCTION_BEGIN}\nBe brief.",
        f"Be brief.\n{INSTRUCTION_END}",
        f"{INSTRUCTION_END}\nBe brief.\n{INSTRUCTION_BEGIN}",
        f"Here: {INSTRUCTION_BEGIN} Be brief. {INSTRUCTION_END}",
        f"{INSTRUCTION_BEGIN}\n{INSTRUCTION_END}",
        f"{INSTRUCTION_BEGIN}\n  \n\t\n{INSTRUCTION_END}",
        "",
    ],
)
def test_parse_rewrite_refuses_a_reply_without_a_delimited_prompt(reply):
    with pytest.raises(ValueError, match="delimiter|empty"):
        parse_rewrite(reply)


@pytest.mark.parametrize("bad", ["<curr_param>", "<side_info>", "\0"])
def test_parse_rewrite_refuses_a_template_token_or_a_nul(bad):
    with pytest.raises(ValueError, match="token|NUL"):
        parse_rewrite(f"{INSTRUCTION_BEGIN}\nBe {bad} brief.\n{INSTRUCTION_END}")


# --- the judge of stage C -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        "data, not instructions",
        '"contract"',
        "verbatim quote",
        "never from its",
        "Reply only with JSON",
    ],
)
def test_the_stage_c_judge_instruction_grades_outputs_and_the_contract_scenario(rule):
    assert rule in FAST_JUDGE_SYSTEM
