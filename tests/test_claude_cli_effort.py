"""`--effort` in the `claude -p` command (SPEC R25; ADR-011), and calls of one backend on several
threads, each with its own child (SPEC R18, R19). The fake `claude` and the helpers are those of
`test_claude_cli.py`.
"""

import threading

import pytest
from test_claude_cli import (
    COMMAND,
    MODEL,
    RESULT,
    SCHEMA,
    FakeClaude,
    ask,
    backend,
    install_fake,
    real,
)

from autoimprover.types import EFFORT_LEVELS, Call

WAIT = 30.0  # seconds; only a broken implementation ever waits this long
AFTER_MODEL = COMMAND.index("--model") + 2


@pytest.fixture
def fake(tmp_path, monkeypatch) -> FakeClaude:
    return install_fake(tmp_path, monkeypatch)


@real
@pytest.mark.parametrize("level", EFFORT_LEVELS)
def test_effort_follows_the_model_in_the_command(fake, tmp_path, level):
    ask(tmp_path, effort=level)
    assert fake.argv()[1:] == [*COMMAND[:AFTER_MODEL], "--effort", level, *COMMAND[AFTER_MODEL:]]


@real
def test_effort_comes_before_the_system_prompt_and_the_schema(fake, tmp_path):
    ask(tmp_path, effort="low", system="You judge.", json_schema=SCHEMA)
    assert fake.argv()[1:] == [
        *COMMAND[:AFTER_MODEL],
        "--effort",
        "low",
        *COMMAND[AFTER_MODEL:],
        "--system-prompt=You judge.",
        "--json-schema",
        SCHEMA,
    ]


@real
def test_no_effort_no_flag(fake, tmp_path):
    ask(tmp_path)
    assert fake.argv()[1:] == COMMAND and "--effort" not in fake.argv()


@real
def test_calls_on_several_threads_each_get_their_own_child_and_reply(fake, tmp_path):
    shared = backend(tmp_path)
    start = threading.Barrier(4, timeout=WAIT)
    replies: list[object] = [None] * 4

    def run(i: int) -> None:
        start.wait()
        call = Call(role="task", model=MODEL, user=f"question {i} " * 10_000, effort="low")
        replies[i] = shared.complete(call).text

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)
    assert not any(thread.is_alive() for thread in threads)
    assert replies == [RESULT["result"]] * 4
