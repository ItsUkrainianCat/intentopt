"""The prompt set of `autoimprover bench` (SPEC R26): `bench.load_prompts` reads JSON Lines with
`id`, `prompt`, optional `kind` and optional `examples`, and refuses a bad line naming it; the
repository's own set, `bench/prompts.jsonl`, loads and is varied on purpose."""

import json
from pathlib import Path

import pytest

from autoimprover.bench import MAX_PROMPTS, SHIPPED_PROMPTS, BenchPrompt, load_prompts
from autoimprover.runner import count_tokens
from autoimprover.types import PROMPT_MAX_CHARS, Scenario


def write(tmp_path: Path, *lines: object, name: str = "set.jsonl") -> Path:
    path = tmp_path / name
    text = "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def refused(path: Path, limit: int | None = None) -> str:
    with pytest.raises(ValueError) as caught:
        load_prompts(path, limit)
    return str(caught.value)


# --- what loads ----------------------------------------------------------------------------------


def test_a_set_loads_in_order_with_its_kind_and_examples(tmp_path):
    path = write(
        tmp_path,
        {"id": "a", "prompt": "Write a haiku about rain."},
        "",
        {
            "id": "t-1",
            "prompt": "Translate {text} into French.",
            "kind": "template",
            "examples": [{"input": "hello"}, {"input": "bye", "expected": "au revoir"}],
            "note": "other keys are ignored",
        },
    )
    assert load_prompts(path) == [
        BenchPrompt(id="a", prompt="Write a haiku about rain.", line=1),
        BenchPrompt(
            id="t-1",
            prompt="Translate {text} into French.",
            kind="template",
            examples=(
                Scenario(id="e1", input="hello"),
                Scenario(id="e2", input="bye", expected="au revoir"),
            ),
            line=3,
        ),
    ]


def test_crlf_in_a_prompt_becomes_lf_as_the_tool_reads_prompts(tmp_path):
    path = write(tmp_path, {"id": "a", "prompt": "line one\r\nline two\rthree"})
    assert load_prompts(path)[0].prompt == "line one\nline two\nthree"


def test_limit_takes_the_first_prompts(tmp_path):
    path = write(tmp_path, *({"id": f"p{i}", "prompt": f"prompt {i}"} for i in range(5)))
    assert [p.id for p in load_prompts(path, 2)] == ["p0", "p1"]


def test_more_than_the_most_prompts_needs_a_limit_and_names_the_line(tmp_path):
    lines = [{"id": f"p{i}", "prompt": f"prompt {i}"} for i in range(MAX_PROMPTS + 1)]
    path = write(tmp_path, *lines)
    message = refused(path)
    assert message.startswith(f"line {MAX_PROMPTS + 1}: ")
    assert "--limit" in message and str(MAX_PROMPTS) in message
    assert len(load_prompts(path, MAX_PROMPTS)) == MAX_PROMPTS
    assert len(load_prompts(write(tmp_path, *lines[:MAX_PROMPTS], name="ok.jsonl"))) == 25


def test_a_bom_is_accepted(tmp_path):
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"id": "a", "prompt": "p"}).encode() + b"\n")
    assert [p.id for p in load_prompts(path)] == ["a"]


# --- what is refused, naming the line -------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "says"),
    [
        ("not json", "not valid JSON"),
        ('["a list"]', "not a JSON object"),
        ({"prompt": "no id"}, "`id`"),
        ({"id": 7, "prompt": "an id that is a number"}, "`id`"),
        ({"id": "../up", "prompt": "an id that is a path"}, "`id`"),
        ({"id": "", "prompt": "an empty id"}, "`id`"),
        ({"id": "x" * 65, "prompt": "a long id"}, "`id`"),
        ({"id": "a b", "prompt": "an id with a space"}, "`id`"),
        ({"id": "b"}, "`prompt`"),
        ({"id": "b", "prompt": 3}, "`prompt`"),
        ({"id": "b", "prompt": "   \n "}, "empty"),
        ({"id": "b", "prompt": "a\u0000b"}, "NUL"),
        ({"id": "b", "prompt": "x" * (PROMPT_MAX_CHARS + 1)}, "20,000"),
        ({"id": "b", "prompt": "use <curr_param> here"}, "<curr_param>"),
        ({"id": "b", "prompt": "a\ud800b"}, "surrogate"),
        ({"id": "b", "prompt": "p", "kind": "essay"}, "`kind`"),
        ({"id": "b", "prompt": "p", "kind": None}, "`kind`"),
        ({"id": "b", "prompt": "p", "examples": "one"}, "`examples`"),
        ({"id": "b", "prompt": "p", "examples": []}, "`examples`"),
        ({"id": "b", "prompt": "p", "examples": [{"expected": "x"}]}, "`input`"),
    ],
)
def test_a_bad_line_is_refused_naming_it(tmp_path, line, says):
    path = write(tmp_path, {"id": "a", "prompt": "fine"}, "", line)
    message = refused(path)
    assert message.startswith("line 3: "), message
    assert says in message


def test_a_duplicate_id_is_refused_naming_both_lines(tmp_path):
    path = write(
        tmp_path,
        {"id": "a", "prompt": "one"},
        {"id": "b", "prompt": "two"},
        {"id": "a", "prompt": "three"},
    )
    message = refused(path)
    assert message.startswith("line 3: ") and "line 1" in message


def test_a_line_that_is_not_utf8_is_refused(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(json.dumps({"id": "a", "prompt": "p"}).encode() + b"\n\xff\xfe\n")
    assert refused(path).startswith("line 2: ")


def test_an_empty_set_or_a_missing_file_is_refused(tmp_path):
    assert "no prompts" in refused(write(tmp_path, "", "  "))
    assert "cannot read" in refused(tmp_path / "missing.jsonl")


def test_the_prompt_text_is_never_part_of_an_error(tmp_path):
    secret = "my private prompt text"
    path = write(tmp_path, {"id": "a", "prompt": secret, "kind": secret})
    assert secret not in refused(path)


# --- the repository's own set (SPEC R26) ---------------------------------------------------------

CATEGORIES = {"vague": 5, "task": 5, "template": 4, "code": 3, "long": 3}


def test_the_shipped_set_loads_with_twenty_prompts():
    prompts = load_prompts(SHIPPED_PROMPTS)
    assert len(prompts) == 20
    assert len({p.prompt for p in prompts}) == 20


def test_the_shipped_set_is_varied_as_the_spec_asks():
    prompts = load_prompts(SHIPPED_PROMPTS)
    found = {name: [p for p in prompts if p.id.startswith(f"{name}-")] for name in CATEGORIES}
    assert {name: len(items) for name, items in found.items()} == CATEGORIES
    assert all("{" in p.prompt and "}" in p.prompt for p in found["template"])
    assert all(p.kind == "template" for p in found["template"])
    assert all(300 <= len(p.prompt) <= 1200 and "\n\n" in p.prompt for p in found["long"])
    # the vague ones state a situation or an intention and ask for nothing
    assert all("?" not in p.prompt and count_tokens(p.prompt) < 60 for p in found["vague"])


def test_the_shipped_set_is_not_the_readme_examples():
    readme = (SHIPPED_PROMPTS.parents[1] / "README.md").read_text(encoding="utf-8")
    assert not any(p.prompt in readme for p in load_prompts(SHIPPED_PROMPTS))
