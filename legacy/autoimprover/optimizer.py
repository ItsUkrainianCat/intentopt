#!/usr/bin/env python3
"""GEPA-powered prompt optimizer for Claude Code.

Runs entirely on Claude Code Max plan — no API keys needed.
All LLM calls route through `claude -p` (Claude Code CLI non-interactive mode).
"""

import json
import os
import re
import subprocess
import sys
import shutil
import time
import traceback
from typing import Any

import gepa.optimize_anything as oa
from gepa.optimize_anything import (
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
    optimize_anything,
)

# ---------------------------------------------------------------------------
# Claude CLI backend — uses Max plan, zero config
# ---------------------------------------------------------------------------

CLAUDE_BIN = shutil.which("claude") or "claude"
MAX_RETRIES = 3
CALL_TIMEOUT = 120
REFLECTION_TIMEOUT = 240  # reflection prompts are longer


def claude_call(prompt: str, timeout: int = CALL_TIMEOUT) -> str:
    """Call Claude via CLI non-interactive mode. Uses Max plan credits."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = subprocess.run(
                [CLAUDE_BIN, "-p", "--output-format", "text", "--max-turns", "1"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0 and result.stdout.strip():
                if result.stderr.strip():
                    print(f"[claude stderr] {result.stderr.strip()[:200]}", file=sys.stderr)
                return result.stdout.strip()
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
                continue
            raise RuntimeError(
                f"claude -p failed after {MAX_RETRIES} attempts. "
                f"exit={result.returncode} stderr={result.stderr[:300]}"
            )
        except subprocess.TimeoutExpired:
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
                continue
            raise RuntimeError(f"claude -p timed out after {timeout}s x {MAX_RETRIES} attempts")
    raise RuntimeError("claude_call: unreachable")


def reflection_lm(prompt: str | list[dict[str, Any]]) -> str:
    """GEPA reflection/mutation LM — routed through Claude CLI.

    Accepts str or list[dict] (GEPA's LanguageModel protocol).
    """
    if isinstance(prompt, list):
        # GEPA may pass chat-format messages; flatten to a single string
        parts = []
        for msg in prompt:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multimodal content blocks
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            if role:
                parts.append(f"[{role}]\n{content}")
            else:
                parts.append(str(content))
        prompt = "\n\n".join(parts)
    return claude_call(prompt, timeout=REFLECTION_TIMEOUT)


# ---------------------------------------------------------------------------
# Evaluator — Claude-as-judge scoring prompts for Claude Code usage
# ---------------------------------------------------------------------------

# Uses string concatenation instead of .format() to avoid brace conflicts
# when candidate prompts contain { } characters (code, JSON, f-strings)
JUDGE_PREAMBLE = """\
You are an expert prompt engineer. Evaluate the following prompt that is \
designed to be used with Claude Code (Anthropic's autonomous coding CLI).

Claude Code has these tools: Bash, Read, Write, Edit, Glob, Grep, \
sub-agents, MCP servers, WebFetch, WebSearch, task tracking, and \
multi-step autonomous reasoning.

Score the prompt on these 6 dimensions (each 0-10):

1. CLARITY — unambiguous, well-structured, no room for misinterpretation
2. SPECIFICITY — precise about inputs, outputs, constraints, edge cases
3. EFFECTIVENESS — would reliably produce high-quality results
4. AUTONOMY — enables multi-step autonomous execution, minimal back-and-forth
5. TOOL_AWARENESS — leverages Claude Code's tools (bash, file ops, MCP, agents)
6. ROBUSTNESS — handles errors, edge cases, unexpected states gracefully

Respond with ONLY this JSON (no markdown, no explanation):
{"clarity":N,"specificity":N,"effectiveness":N,"autonomy":N,"tool_awareness":N,"robustness":N,"reasoning":"one sentence","weaknesses":"what to improve"}

PROMPT TO EVALUATE:
```
"""

JUDGE_SUFFIX = "\n```"


def _extract_json(text: str) -> dict:
    """Extract JSON object from text that may contain markdown fences or prose."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Brace-counting parser for nested JSON
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    # Last resort: try json_repair if available
    try:
        from json_repair import repair_json
        return json.loads(repair_json(text))
    except Exception:
        pass

    raise ValueError(f"No valid JSON found in: {text[:200]}")


DIMS = ["clarity", "specificity", "effectiveness", "autonomy", "tool_awareness", "robustness"]


def evaluate_prompt(candidate: str) -> float:
    """Score a candidate prompt via Claude-as-judge with ASI logging."""
    try:
        # Concatenate instead of .format() to avoid brace issues in candidate
        full_prompt = JUDGE_PREAMBLE + candidate + JUDGE_SUFFIX
        text = claude_call(full_prompt)
        scores = _extract_json(text)
        values = [float(scores.get(d, 0)) for d in DIMS]
        raw = sum(values) / (len(DIMS) * 10)

        oa.log(f"Scores: { {d: v for d, v in zip(DIMS, values)} }")
        oa.log(f"Reasoning: {scores.get('reasoning', 'N/A')}")
        oa.log(f"Weaknesses: {scores.get('weaknesses', 'N/A')}")
        oa.log(f"Aggregate: {raw:.4f}")

        return raw

    except (ValueError, json.JSONDecodeError) as e:
        # Scoring failure (bad judge output) — return 0 and let GEPA continue
        oa.log(f"Scoring parse error: {type(e).__name__}: {e}")
        return 0.0
    except (RuntimeError, subprocess.SubprocessError) as e:
        # Infrastructure failure — re-raise so GEPA's error handling kicks in
        raise


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def run(prompt: str, max_calls: int = 30, run_dir: str | None = None) -> dict:
    """Run GEPA optimize_anything on a prompt. All LLM calls via Claude CLI."""
    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=max_calls,
            track_best_outputs=True,
            display_progress_bar=True,
            raise_on_exception=False,
            run_dir=run_dir,
            candidate_selection_strategy="pareto",
            frontier_type="hybrid",
            parallel=False,  # Sequential — avoid rate-limiting claude -p
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_lm,
            module_selector="all",
            skip_perfect_score=False,
        ),
    )

    result = optimize_anything(
        seed_candidate=prompt,
        evaluator=evaluate_prompt,
        objective=(
            "Optimize this prompt for use with Claude Code (Anthropic's autonomous coding CLI). "
            "Maximize clarity, specificity, effectiveness, autonomous execution capability, "
            "tool awareness (bash, file I/O, MCP, sub-agents), and robustness. "
            "The prompt should enable multi-step autonomous work without unnecessary back-and-forth. "
            "Preserve the user's original intent while dramatically improving structure and precision."
        ),
        config=config,
    )

    if not result.val_aggregate_scores:
        return {
            "best_prompt": prompt,
            "best_score": 0.0,
            "seed_score": 0.0,
            "improvement": 0.0,
            "candidates_explored": 0,
            "total_evaluations": 0,
        }

    best = result.best_candidate
    if isinstance(best, dict):
        best = best.get("current_candidate", next(iter(best.values())))

    seed_score = result.val_aggregate_scores[0]
    best_score = result.val_aggregate_scores[result.best_idx]

    return {
        "best_prompt": best,
        "best_score": round(best_score, 4),
        "seed_score": round(seed_score, 4),
        "improvement": round(best_score - seed_score, 4),
        "candidates_explored": result.num_candidates,
        "total_evaluations": result.total_metric_calls,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GEPA prompt optimizer for Claude Code (Max plan — no API key needed)"
    )
    parser.add_argument("prompt", nargs="?", help="Prompt to optimize (or pipe via stdin)")
    parser.add_argument("--max-calls", type=int, default=30,
                        help="Max evaluation calls (default: 30, use 60+ for deeper optimization)")
    parser.add_argument("--run-dir", default=None,
                        help="Checkpoint directory for resuming")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    prompt = args.prompt
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        parser.error("No prompt provided. Pass as argument or pipe via stdin.")

    if not shutil.which("claude"):
        print("Error: 'claude' CLI not found in PATH. Install Claude Code first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"GEPA optimizer — {args.max_calls} max evals via Claude CLI (Max plan)\n",
          file=sys.stderr)

    result = run(prompt, max_calls=args.max_calls, run_dir=args.run_dir)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nScore: {result['seed_score']:.4f} -> {result['best_score']:.4f}  "
              f"(+{result['improvement']:.4f})")
        print(f"Candidates explored: {result['candidates_explored']}")
        print(f"Total evaluations: {result['total_evaluations']}")
        print(f"\n{'='*60}")
        print("OPTIMIZED PROMPT:")
        print(f"{'='*60}\n")
        print(result["best_prompt"])


if __name__ == "__main__":
    main()
