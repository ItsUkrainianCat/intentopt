# `just check` is the definition of done: it must be green before a change counts as finished.
# Every recipe goes through `uv run --frozen`, so each git worktree gets its own .venv from the
# shared uv cache and tests always import the code of the worktree they run in.

# format check, lint, types, tests
check: fmt-check lint types test

# run the test suite; extra args go to pytest (just test tests/test_x.py -k name)
test *args:
    timeout 300 uv run --frozen pytest {{ args }}

lint:
    uv run --frozen ruff check .

# pyright is the host binary; it reads the worktree's own interpreter
types:
    pyright --pythonpath "$(uv run --frozen python -c 'import sys; print(sys.executable)')"

# rewrite files: format, then apply safe lint fixes
fmt:
    uv run --frozen ruff format .
    uv run --frozen ruff check --fix .

fmt-check:
    uv run --frozen ruff format --check .

# the user's live check (SPEC section 5): runs the real tool once on a small prompt with the real
# backend (`claude -p`), so it spends subscription calls; never part of `check`. Needs the real
# backend module (WP1b) and a logged-in `claude`.

# live check on one small prompt (spends real calls): the plan, then the run
smoke prompt="Summarise the meeting notes for the team in five bullet points.":
    uv run --frozen autoimprover --dry "{{ prompt }}"
    uv run --frozen autoimprover "{{ prompt }}"
