"""Scenario text that could not reach the model or carries nothing (SPEC R11, R19; ADR-008): the
examples file refuses a lone surrogate escape in any text field, naming the line, and a synthesis
reply with an input that is only whitespace is invalid, asked again as a new sample. The other
scenario tests are in `test_scenarios.py`.
"""

import json
from pathlib import Path

import pytest
from fakes import by_role, synth_reply

from autoimprover.scenarios import read_examples, synthesize
from autoimprover.types import SYNTH_COUNT, CallFailed, Contract

MODEL = "claude-opus-5-5"
PROMPT = "Summarise the bug report below in three bullet points."
CONTRACT = Contract(goal="summarise a bug report", kind="template")
HIGH, LOW = chr(0xD800), chr(0xDC00)  # json.dumps writes each as a \uXXXX escape
SMILE = chr(0x1F600)  # json.dumps writes it as a surrogate pair escape


def write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "examples.jsonl"
    path.write_text("".join(f"{line}\n" for line in lines))
    return path


LONE = {
    "input": {"input": f"a{HIGH}b"},
    "expected": {"input": "a", "expected": LOW},
    "criteria": {"input": "a", "criteria": ["fine", f"x{HIGH}"]},
}


@pytest.mark.parametrize("fields", LONE.values(), ids=LONE.keys())
def test_read_examples_refuses_a_lone_surrogate_escape_naming_the_line(tmp_path, fields):
    bad = json.dumps(fields)
    assert "\\ud" in bad  # the file holds the escape, plain ASCII
    path = write(tmp_path, json.dumps({"input": "good"}), "", bad)
    with pytest.raises(ValueError, match=r"^line 3: contains a lone surrogate"):
        read_examples(path)


def test_read_examples_keeps_a_surrogate_pair_escape_as_its_character(tmp_path):
    path = write(tmp_path, json.dumps({"input": f"smile {SMILE}", "criteria": [SMILE]}))
    [scenario] = read_examples(path)
    assert (scenario.input, scenario.criteria) == (f"smile {SMILE}", (SMILE,))


def synth_with_input(text: str) -> str:
    scenarios = json.loads(synth_reply())["scenarios"]
    scenarios[-1]["input"] = text
    return json.dumps({"scenarios": scenarios})


BLANKS = {"spaces": "   ", "whitespace": "\t\r\n ", "ideographic space": chr(0x3000)}


@pytest.mark.parametrize("blank", BLANKS.values(), ids=BLANKS.keys())
def test_synthesize_asks_again_when_an_input_is_only_whitespace(blank):
    backend = by_role({"synth": [synth_with_input(blank), synth_reply()]})
    got = synthesize(backend, MODEL, PROMPT, CONTRACT)
    assert [s.input for s in got] == [f"situation {i}" for i in range(1, SYNTH_COUNT + 1)]
    assert [c.sample for c in backend.calls] == [0, 1]


def test_synthesize_gives_up_when_every_reply_has_a_blank_input():
    backend = by_role({"synth": synth_with_input(" \n ")})
    with pytest.raises(CallFailed, match="whitespace"):
        synthesize(backend, MODEL, PROMPT, CONTRACT)
    assert [c.sample for c in backend.calls] == [0, 1, 2]


def test_synthesize_keeps_an_input_with_text_and_surrounding_whitespace_as_it_is():
    backend = by_role({"synth": synth_with_input("  indented\n")})
    got = synthesize(backend, MODEL, PROMPT, CONTRACT)
    assert got[-1].input == "  indented\n"
    assert [c.sample for c in backend.calls] == [0]
