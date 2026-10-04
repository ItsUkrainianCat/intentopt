"""Literals a rewrite must keep verbatim, and the check that it kept them (SPEC R6, R9):
extraction cases, apostrophes that are not quotes, fences, adversarial text of 20,000 characters,
the literal part of `check`, and a hand-written property test over a seeded generator (no new
dependency)."""

import random

import pytest
from fakes import ScriptedBackend, judge_reply

from autoimprover.contract import Violation, check, literals, literals_preserved
from autoimprover.types import Contract

FENCE = "```python\ndef area(r):\n    return 3.14 * r * r\n```"

# --- literals (SPEC R9) ------------------------------------------------------------------------

CASES = {
    "fence with a language tag": (f"Use this:\n{FENCE}\nThen stop.", (FENCE,)),
    "unterminated fence runs to the end": ("Run:\n```bash\nls -la\n\n", ("```bash\nls -la",)),
    "tilde fence": ("Query:\n~~~sql\nSELECT 1;\n~~~\n", ("~~~sql\nSELECT 1;\n~~~",)),
    "a longer fence holds a shorter one": (
        "Example:\n````md\n```py\nx = 1\n```\n````\nEnd.",
        ("````md\n```py\nx = 1\n```\n````",),
    ),
    "same-length fences close at the first closing line": (
        "```\na\n```\nb\n```\nc",
        ("```\na\n```", "```\nc"),
    ),
    "CRLF text": ("Do:\r\n```sh\r\necho hi\r\n```\r\nDone.\r\n", ("```sh\r\necho hi\r\n```",)),
    "an indented fence keeps its inner indentation": (
        "Steps:\n  1. build:\n     ```\n     make all\n     ```\n",
        ("```\n     make all\n     ```",),
    ),
    "a fence line with text after it does not close": (
        "```\na\n``` not a close\nb\n",
        ("```\na\n``` not a close\nb",),
    ),
    "backticks in the info string: not a fence": ("```js```\nplain", ("`js`",)),
    "inline code": ("Call `run()` and then `stop()`.", ("`run()`", "`stop()`")),
    "placeholders, deduplicated": (
        "Write about {topic} for {{ audience }} in ${LANG} into <PATH> as %(user)s, then {topic}.",
        ("{topic}", "{{ audience }}", "${LANG}", "<PATH>", "%(user)s"),
    ),
    "angle brackets that are not placeholders": ("if a < b and c > d, or x<y z>w, or <a b>", ()),
    "URLs lose trailing punctuation": (
        "See https://example.com/a?b=1, (http://x.org/p). Or https://y.io/z.",
        ("https://example.com/a?b=1", "http://x.org/p", "https://y.io/z"),
    ),
    "file paths": (
        "Edit /etc/hosts, ./run.sh, ../lib/util.py, ~/notes.md and src/app/main.py.",
        ("/etc/hosts", "./run.sh", "../lib/util.py", "~/notes.md", "src/app/main.py"),
    ),
    "file paths without an extension": (
        "Run ./configure in ../build from ~/projects, logs in /var/log/ (C:/Users/me/a.txt).",
        ("./configure", "../build", "~/projects", "/var/log/", "/Users/me/a.txt"),
    ),
    "slashes that are not paths": (
        "Answer yes/no, and/or km/h at 24/7; ratio 1/2.5; TCP/IP; folder src/app/",
        (),
    ),
    "quoted strings": (
        "Say \"hello world\", then “bonjour”, then reply 'yes' or 'no way'.",
        ('"hello world"', "“bonjour”", "'yes'", "'no way'"),
    ),
    "a literal inside another is kept too": (
        "Quote \"see https://x.io/a and {name}\" exactly.\n```\nprint('hi')\n```",
        (
            '"see https://x.io/a and {name}"',
            "https://x.io/a",
            "{name}",
            "```\nprint('hi')\n```",
            "'hi'",
        ),
    ),
    "a stray double quote does not pair across lines": ('He is 6" tall.\nSay "hi".', ('"hi"',)),
    "plain prose": ("Please write a short, friendly summary.", ()),
    "empty text": ("", ()),
}


@pytest.mark.parametrize(("text", "expected"), CASES.values(), ids=CASES.keys())
def test_literals_finds_exactly_these_in_order_of_first_appearance(text, expected):
    assert literals(text) == expected


APOSTROPHES = {
    "contractions": "Don't panic, it's fine, we'll see.",
    "rock 'n' roll": "Play rock 'n' roll loud.",
    "a decade": "Music from the '90s only.",
    "a decade, then a plural possessive": "In the '90s, the users' data was small.",
    "a decade with a possessive right after": "Like the '90s kids' toys.",
    "an elision, then a possessive": "Tell 'em the admins' rules.",
    "possessives": "The users' and the admins' views.",
    "an apostrophe in a word, then a possessive": "The dog's toy and the cats' bed.",
    "one character between quotes": "Pick 'a' or 'b'.",
}


@pytest.mark.parametrize("text", APOSTROPHES.values(), ids=APOSTROPHES.keys())
def test_literals_does_not_take_apostrophes_for_quotes(text):
    assert literals(text) == ()


def test_an_apostrophe_inside_a_word_never_closes_a_quote():
    assert "'it'" not in literals("Write 'it's done' here.")


def test_literals_still_finds_a_single_quoted_string_next_to_apostrophes():
    # The twin of the test above: the same apostrophes around a real quoted string.
    text = "Don't change 'Acme Corp', it's the users' name, or 'maybe (later)'."
    assert literals(text) == ("'Acme Corp'", "'maybe (later)'")


def test_every_literal_is_a_span_of_the_prompt():
    text = "\n".join(case for case, _ in CASES.values())
    found = literals(text)
    assert found and all(lit and lit in text for lit in found)
    assert len(found) == len(set(found))


# --- linear time on adversarial text (sizes only, no clock) -------------------------------------

ADVERSARIAL_UNITS = [
    "'",
    '"',
    "`",
    "{",
    "/",
    "~",
    " 'a",
    "'a ",
    " 'ab",
    "'90s a' ",
    '"a\n',
    "`a\r\n",
    "``` ",
    "```\n",
    "````\n```\n",
    "{{a",
    "{{ ",
    "${",
    "{a.",
    "<a",
    "<a-",
    "%(a",
    "a/",
    "./",
    "~/",
    "a.b/",
    "http://",
    "https:",
    "“",
    "“a",
    # An opener, then a long run that never closes: a pattern that backtracks badly blows up here.
    "`" + "ab " * 20 + "\n",
    "\n'" + "ab " * 20,
    '"' + "a" * 60 + "\n",
    "“" + "a" * 60 + "\n",
    "{" + "a" * 60 + " ",
    "{{ " + "a" * 60 + " ",
    "${" + "a." * 30 + " ",
    "<" + "a-" * 30 + " ",
    "%(" + "a" * 60 + " ",
    "https://" + "a" * 60 + ")",
    "/" + "a/" * 30 + "/ ",
    "a/" * 30 + "a.1 ",
]


def _fill(unit: str, size: int = 20_000) -> str:
    return (unit * (size // len(unit) + 1))[:size]


@pytest.mark.parametrize("opener", ["`", "'", '"', "“", "{", "{{ ", "<", "%(", "```\n", "/"])
def test_literals_on_one_unclosed_run_of_20000_characters_finishes(opener):
    text = opener + "a" * (20_000 - len(opener))
    assert len(text) == 20_000
    found = literals(text)
    assert all(lit and lit in text for lit in found)


@pytest.mark.parametrize("unit", ADVERSARIAL_UNITS)
def test_literals_on_20000_characters_of_repeated_adversarial_text_finishes(unit):
    text = _fill(unit)
    assert len(text) == 20_000
    found = literals(text)
    assert all(lit and lit in text for lit in found)


def test_literals_on_20000_characters_of_seeded_random_adversarial_text_finishes():
    rng = random.Random(9)
    for _ in range(5):
        text = "".join(rng.choices("'\"`{}$<>%()/~.:- \n\rab1“”", k=20_000))
        assert len(text) == 20_000
        found = literals(text)
        assert all(lit and lit in text for lit in found)


# --- literals_preserved (SPEC R9) --------------------------------------------------------------

ORIGINAL = f"Summarise {{topic}} for `make test` users, see https://x.io/a.\n{FENCE}\nKeep 'Acme'."


def test_a_candidate_that_keeps_every_literal_elsewhere_in_other_words_is_preserved():
    candidate = f"Keep 'Acme'.\n{FENCE}\nRun `make test`, read https://x.io/a, summarise {{topic}}."
    assert literals_preserved(ORIGINAL, candidate) is True
    assert literals_preserved(ORIGINAL, ORIGINAL) is True


@pytest.mark.parametrize(
    "lost",
    ["{topic}", "`make test`", "https://x.io/a", FENCE, "'Acme'"],
)
def test_a_candidate_that_loses_one_literal_is_not_preserved(lost):
    candidate = ORIGINAL.replace(lost, "it")
    assert literals_preserved(ORIGINAL, candidate) is False


@pytest.mark.parametrize(
    "changed",
    [
        FENCE.replace("    return", "  return"),  # indentation
        FENCE.replace("    return", "\treturn"),  # a tab for spaces
        FENCE.replace("(r):\n", "(r):  \n"),  # trailing spaces on a line
        FENCE.replace("\n", "\r\n"),  # line endings
    ],
    ids=["indentation", "tab", "trailing spaces", "CRLF"],
)
def test_whitespace_inside_a_code_block_must_match_exactly(changed):
    assert literals_preserved(ORIGINAL, ORIGINAL.replace(FENCE, changed)) is False


def test_a_prompt_without_literals_is_preserved_by_any_candidate():
    assert literals_preserved("Write a short poem.", "") is True


# --- check: the literal part, before any model call (SPEC R6, R9) -------------------------------

JUDGE = "claude-sonnet-5-5"
CONTRACT = Contract(goal="summarise the report", kind="task", keep=("Acme",))
SHORT = "Summarise the report for {audience} in `three bullets`. Mention Acme by name."


def judge_passing_everything() -> ScriptedBackend:
    return ScriptedBackend(lambda call: judge_reply(call))


def test_a_missing_literal_is_a_violation_and_no_model_is_called():
    # The judge would pass everything: the literal check must not depend on it.
    backend = judge_passing_everything()
    candidate = SHORT.replace("{audience}", "the audience")
    assert check(backend, JUDGE, CONTRACT, SHORT, candidate) == [
        Violation(check_id="literal", text="{audience}")
    ]
    assert backend.calls == []
    # The twin: with the literal kept, the same judge is asked and passes the candidate.
    assert check(backend, JUDGE, CONTRACT, SHORT, SHORT + " Be brief.") == []
    assert len(backend.calls) == 1


def test_every_missing_literal_is_a_violation_in_order_of_first_appearance():
    backend = judge_passing_everything()
    assert check(backend, JUDGE, CONTRACT, SHORT, "Summarise it.") == [
        Violation(check_id="literal", text="{audience}"),
        Violation(check_id="literal", text="`three bullets`"),
    ]
    assert backend.calls == []


def test_a_long_missing_literal_is_shortened_to_80_characters():
    block = "```\n" + "x = 1\n" * 40 + "```"
    exact = "`" + "y" * 78 + "`"
    backend = judge_passing_everything()
    violations = check(backend, JUDGE, CONTRACT, f"Run:\n{block}\nand {exact}.", "Run it.")
    assert violations == [
        Violation(check_id="literal", text=block[:77] + "..."),
        Violation(check_id="literal", text=exact),  # exactly 80 characters: kept whole
    ]
    assert len(violations[0].text) == len(exact) == 80


# --- property test: a seeded generator, no new dependency (SPEC R9) -----------------------------

# Literals that `literals` must find exactly as written; none is a part of another.
LITERAL_POOL = (
    FENCE,
    "~~~\nSELECT name FROM users;\n~~~",
    "`git status`",
    "{topic}",
    "{{audience}}",
    "${HOME}",
    "<PATH>",
    "%(user)s",
    "https://example.com/docs?page=2",
    "/etc/hosts",
    "./build.sh",
    "../lib/util.py",
    "~/notes/today.md",
    "src/app/main.py",
    '"exactly this phrase"',
    "“curly phrase”",
    "'single quoted'",
)
# Filler the rewrite may change freely, apostrophes included: none of it is a literal.
FILLER = (
    "please write a short summary of the report for our team",
    "and keep it clear don't ramble it's fine",
    "rock 'n' roll from the '90s the users' data the '90s kids' toys tell 'em",
)
REWRITTEN = ("kindly", "draft", "brief", "notes", "we're", "can't", "clients'", "quickly", "now")
PUNCTUATION = ("", ".", ",", ":", ";")
SEED = 20261004


def _is_fence(literal: str) -> bool:
    return literal.startswith(("```", "~~~"))


def _render(segments: list[tuple[str, bool]], rng: random.Random) -> str:
    """Join filler and literals: a fence on lines of its own, other literals between spaces and
    sometimes followed by punctuation, as in prose."""
    out = []
    for text, is_literal in segments:
        if is_literal and _is_fence(text):
            out.append(f"\n{text}\n")
        elif is_literal:
            out.append(f" {text}{rng.choice(PUNCTUATION)} ")
        else:
            out.append(text)
    return "".join(out)


def _filler(rng: random.Random, words: tuple[str, ...]) -> str:
    return " ".join(rng.choices(words, k=rng.randint(1, 8)))


def _generate(rng: random.Random) -> tuple[list[str], str, list[tuple[str, bool]]]:
    """The literals used (a literal may repeat), the prompt, and a candidate's segments that keep
    every literal but rewrite all the filler."""
    used = rng.choices(LITERAL_POOL, k=rng.randint(1, 6))
    words = tuple(" ".join(FILLER).split(" "))
    original: list[tuple[str, bool]] = []
    candidate: list[tuple[str, bool]] = []
    for literal in used:
        original += [(_filler(rng, words), False), (literal, True)]
        candidate += [(_filler(rng, REWRITTEN), False), (literal, True)]
    original.append((_filler(rng, words), False))
    candidate.append((_filler(rng, REWRITTEN), False))
    return used, _render(original, rng), candidate


def _mutations(literal: str, rng: random.Random) -> dict[str, str]:
    """The literal deleted, one character altered, and two neighbouring characters swapped."""
    i = rng.randrange(len(literal))
    altered = literal[:i] + ("Y" if literal[i] == "X" else "X") + literal[i + 1 :]
    j = rng.choice([k for k in range(len(literal) - 1) if literal[k] != literal[k + 1]])
    swapped = literal[:j] + literal[j + 1] + literal[j] + literal[j + 2 :]
    return {"deleted": "", "altered": altered, "reordered": swapped}


def test_the_pool_holds_no_literal_inside_another():
    assert not [(a, b) for a in LITERAL_POOL for b in LITERAL_POOL if a != b and a in b]


def test_property_rewritten_filler_passes_and_any_damaged_literal_fails():
    rng = random.Random(SEED)
    checked = 0
    for _ in range(200):
        used, prompt, segments = _generate(rng)
        found = literals(prompt)
        # Every pool literal is found, and nothing reaches into the filler.
        assert set(used) <= set(found), prompt
        assert all(any(lit in u for u in used) for lit in found), prompt
        assert literals_preserved(prompt, _render(segments, rng)) is True, prompt
        for literal in dict.fromkeys(used):
            for kind, damaged in _mutations(literal, rng).items():
                # Every copy of the literal is damaged: an intact copy would still keep it.
                broken = [(damaged if (t, f) == (literal, True) else t, f) for t, f in segments]
                candidate = _render(broken, rng)
                assert literal not in candidate, (kind, literal)
                assert literals_preserved(prompt, candidate) is False, (kind, literal, prompt)
                checked += 1
    assert checked > 600  # every literal of every prompt, three ways
