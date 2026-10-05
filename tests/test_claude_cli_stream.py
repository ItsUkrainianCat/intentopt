"""What `claude -p` prints and how one call reads it (SPEC R18; ADR-009): the lockdown check of
every `init` line and the reply taken from the last `result` line. The fake `claude` and the
helpers are those of `test_claude_cli.py`.
"""

import json

import pytest
from test_claude_cli import (
    BUILTIN_AGENTS,
    INIT,
    LS,
    MIDDLE,
    MODEL,
    RESULT,
    RLO,
    SCHEMA,
    FakeClaude,
    ask,
    backend,
    edited,
    install_fake,
    real,
)

from autoimprover import claude_cli
from autoimprover.types import Call, CallError, SessionNotLockedDown

# The final line of the user's real call with `--json-schema` (probe 4: claude 2.1.287, Haiku 4.5),
# trimmed: `result` is the answer as JSON text and `structured_output` the parsed object. The
# schema is answered through an internal tool, so `num_turns` 2 and `stop_reason` "tool_use" are
# normal there, not a failure.
PROBE4 = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "num_turns": 2,
    "stop_reason": "tool_use",
    "terminal_reason": "completed",
    "result": '{"name":"Alice","age":30}',
    "structured_output": {"name": "Alice", "age": 30},
    "usage": {"input_tokens": 1149},
}


@pytest.fixture
def fake(tmp_path, monkeypatch) -> FakeClaude:
    return install_fake(tmp_path, monkeypatch)


# --- lockdown on every call (SPEC R18, ADR-009) --------------------------------------------------


@real
@pytest.mark.parametrize(
    ("field", "value", "shown"),
    [
        ("tools", ["Bash", "Read"], "'Bash'"),
        ("mcp_servers", [{"name": "github", "status": "connected"}], "github"),
        ("skills", ["deploy"], "deploy"),
        ("slash_commands", ["compact"], "compact"),
        ("agents", [*BUILTIN_AGENTS, "rogue"], "rogue"),
        ("output_style", "Proactive", "Proactive"),
        ("tools", "none", "'none'"),
        ("tools", ["StructuredOutput"], "StructuredOutput"),
        ("agents", "claude", "'claude'"),
        ("mcp_servers", {}, "{}"),
    ],
)
def test_a_session_that_is_not_locked_down_is_refused(fake, tmp_path, field, value, shown):
    fake.answer(edited(INIT, **{field: value}), *MIDDLE, RESULT)
    with pytest.raises(SessionNotLockedDown) as caught:
        ask(tmp_path)
    assert field in str(caught.value) and shown in str(caught.value)


@real
def test_tools_are_checked_against_one_named_allowed_set(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(claude_cli, "_ALLOWED_TOOLS", frozenset({"StructuredOutput"}))
    fake.answer(edited(INIT, tools=["StructuredOutput"]), *MIDDLE, RESULT)
    assert ask(tmp_path).text == "OK"
    fake.answer(edited(INIT, tools=["StructuredOutput", "Bash"]), *MIDDLE, RESULT)
    with pytest.raises(SessionNotLockedDown, match="Bash"):
        ask(tmp_path)


@real
@pytest.mark.parametrize(
    "field", ["tools", "mcp_servers", "skills", "slash_commands", "agents", "output_style"]
)
def test_an_init_line_missing_a_field_is_refused(fake, tmp_path, field):
    fake.answer(edited(INIT, **{field: None}), *MIDDLE, RESULT)
    with pytest.raises(SessionNotLockedDown, match=field):
        ask(tmp_path)


@real
def test_the_real_init_is_accepted_whatever_plugins_it_lists(fake, tmp_path):
    plugins = [{"name": f"plugin-{i}", "path": f"/p/{i}"} for i in range(70)]
    fake.answer(edited(INIT, plugins=plugins, agents=["Plan"]), *MIDDLE, RESULT)
    assert ask(tmp_path).text == "OK"


@real
def test_a_stream_without_an_init_line_is_refused(fake, tmp_path):
    fake.answer(*MIDDLE, RESULT)
    with pytest.raises(SessionNotLockedDown, match="init"):
        ask(tmp_path)


@real
def test_every_init_line_is_checked(fake, tmp_path):
    fake.answer(INIT, edited(INIT, tools=["Bash"]), *MIDDLE, RESULT)
    with pytest.raises(SessionNotLockedDown, match="Bash"):
        ask(tmp_path)


@real
def test_every_call_is_checked_not_only_the_first(fake, tmp_path):
    claude = backend(tmp_path)
    assert claude.complete(Call("task", MODEL, "one")).text == "OK"
    fake.answer(edited(INIT, mcp_servers=[{"name": "late"}]), *MIDDLE, RESULT)
    with pytest.raises(SessionNotLockedDown, match="late"):
        claude.complete(Call("task", MODEL, "two"))


@real
def test_a_long_offending_value_is_shown_short(fake, tmp_path):
    fake.answer(edited(INIT, tools=[f"tool-{i}" for i in range(2000)]), *MIDDLE, RESULT)
    with pytest.raises(SessionNotLockedDown) as caught:
        ask(tmp_path)
    assert "tool-0" in str(caught.value) and len(str(caught.value)) < 400


@real
def test_a_failed_lockdown_wins_over_a_failed_exit(fake, tmp_path):
    fake.answer(edited(INIT, tools=["Bash"]), exit=1, stderr="boom")
    with pytest.raises(SessionNotLockedDown, match="tools"):
        ask(tmp_path)


# --- the reply (ADR-009) -------------------------------------------------------------------------


@real
def test_the_reply_is_the_result_of_the_last_result_line(fake, tmp_path):
    first = edited(RESULT, result="an earlier result")
    fake.answer(INIT, first, *MIDDLE, edited(RESULT, result=f"last{LS}line"))
    assert ask(tmp_path).text == f"last{LS}line"


@real
def test_the_real_schema_reply_gives_a_schema_call_the_parsed_object(fake, tmp_path):
    fake.answer(INIT, *MIDDLE, PROBE4)
    reply = ask(tmp_path, json_schema=SCHEMA)
    assert (reply.text, reply.tokens_in) == (json.dumps({"name": "Alice", "age": 30}), 1149)
    assert ask(tmp_path).text == '{"name":"Alice","age":30}'


@real
@pytest.mark.parametrize("shaped", [["Alice", 30], "Alice", 30, True])
def test_a_structured_output_that_is_not_an_object_leaves_the_result(fake, tmp_path, shaped):
    fake.answer(INIT, *MIDDLE, edited(PROBE4, structured_output=shaped))
    assert ask(tmp_path, json_schema=SCHEMA).text == '{"name":"Alice","age":30}'


@real
def test_a_schema_call_without_structured_output_uses_the_result(fake, tmp_path):
    fake.answer(INIT, *MIDDLE, edited(RESULT, result='{"results": []}'))
    assert ask(tmp_path, json_schema=SCHEMA).text == '{"results": []}'


@real
@pytest.mark.parametrize(
    ("objects", "message"),
    [
        ((INIT, *MIDDLE, edited(RESULT, is_error=True, result="API Error: busy")), "busy"),
        ((INIT, *MIDDLE, edited(RESULT, is_error=None)), "error"),
        ((INIT, *MIDDLE, edited(RESULT, terminal_reason="max_turns")), "max_turns"),
        ((INIT, *MIDDLE, edited(RESULT, terminal_reason=None)), "terminal_reason"),
        ((INIT, *MIDDLE), "no result"),
        ((INIT, "not json {", *MIDDLE, RESULT), "not json {"),
        ((INIT, *MIDDLE, RESULT, "[1, 2]"), r"\[1, 2\]"),
        ((INIT, '{"type": "assistant", "x": "\udcff"}', *MIDDLE, RESULT), "JSON"),
        ((INIT, *MIDDLE, edited(RESULT, result=None)), "result text"),
        ((INIT, *MIDDLE, edited(RESULT, result=["OK"])), "result text"),
    ],
    ids=[
        "is-error",
        "no-is-error",
        "turns",
        "no-reason",
        "no-result",
        "garbage",
        "array",
        "not-utf8",
        "no-text",
        "list",
    ],
)
def test_a_failed_or_unreadable_reply_is_a_call_error(fake, tmp_path, objects, message):
    fake.answer(*objects)
    with pytest.raises(CallError, match=message):
        ask(tmp_path)


@real
def test_error_text_is_short_and_free_of_control_characters(fake, tmp_path):
    fake.answer(INIT, edited(RESULT, is_error=True, result=f"\x1b{RLO}\x07" + "x" * 1000))
    with pytest.raises(CallError) as caught:
        ask(tmp_path)
    message = str(caught.value)
    assert "x" * 200 in message and "x" * 201 not in message
    assert "\x1b" not in message and RLO not in message


@real
@pytest.mark.parametrize("with_result", [False, True])
def test_a_non_zero_exit_is_a_call_error_with_the_code_and_the_end_of_stderr(
    fake, tmp_path, with_result
):
    stderr = "\x1b[31mearly" + "e" * 500 + "\nboom at the end\n"
    fake.answer(INIT, *([RESULT] if with_result else []), exit=7, stderr=stderr)
    with pytest.raises(CallError) as caught:
        ask(tmp_path)
    message = str(caught.value)
    assert "code 7" in message and "boom at the end" in message
    assert "\x1b" not in message and "early" not in message and "e" * 201 not in message


@real
def test_a_reply_over_one_mebibyte_is_a_call_error(fake, tmp_path):
    fake.answer(INIT, *MIDDLE, edited(RESULT, result="y" * (1 << 20)))
    with pytest.raises(CallError, match="MiB"):
        ask(tmp_path)
