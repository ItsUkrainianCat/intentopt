#!/usr/bin/env python3
"""GEPA-powered prompt optimizer for Claude Code."""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root if present
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    load_dotenv(_env)

import litellm
import gepa.optimize_anything as oa
from gepa.optimize_anything import (
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
    optimize_anything,
)

EVAL_MODEL = os.getenv("AUTOIMPROVER_EVAL_MODEL", "anthropic/claude-sonnet-4-6")
REFLECTION_MODEL = os.getenv("AUTOIMPROVER_REFLECTION_MODEL", "anthropic/claude-sonnet-4-6")

JUDGE_SYSTEM = """\
You are an expert prompt engineer evaluating prompts designed for Claude Code \
(Anthropic's autonomous coding CLI). Claude Code has access to: Bash, file I/O, \
Grep, Glob, Edit, Write, Read, sub-agents, MCP servers, web fetch, and multi-step \
reasoning.

Score the candidate prompt on these dimensions (each 0-10):

1. CLARITY — unambiguous, well-structured, no room for misinterpretation
2. SPECIFICITY — precise about inputs, outputs, constraints, and edge cases
3. EFFECTIVENESS — would reliably produce high-quality results
4. AUTONOMY — enables autonomous multi-step execution, minimal back-and-forth
5. TOOL_AWARENESS — leverages Claude Code's tools (bash, file ops, MCP, agents)
6. ROBUSTNESS — handles errors, edge cases, and unexpected states gracefully

Return ONLY valid JSON:
{"clarity":N,"specificity":N,"effectiveness":N,"autonomy":N,"tool_awareness":N,\
"robustness":N,"reasoning":"one sentence","weaknesses":"what to improve"}\
"""


def evaluate_prompt(candidate: str) -> float:
    """Score a candidate prompt via Claude-as-judge with ASI logging."""
    try:
        resp = litellm.completion(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"Evaluate this prompt:\n\n```\n{candidate}\n```"},
            ],
            temperature=0.15,
            max_tokens=512,
        )
        text = resp.choices[0].message.content.strip()

        # Extract JSON from possible markdown fences
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        scores = json.loads(text)
        dims = ["clarity", "specificity", "effectiveness", "autonomy", "tool_awareness", "robustness"]
        values = [float(scores.get(d, 0)) for d in dims]
        raw = sum(values) / (len(dims) * 10)

        oa.log(f"Scores: { {d: v for d, v in zip(dims, values)} }")
        oa.log(f"Reasoning: {scores.get('reasoning', 'N/A')}")
        oa.log(f"Weaknesses: {scores.get('weaknesses', 'N/A')}")
        oa.log(f"Aggregate: {raw:.4f}")

        return raw

    except Exception as e:
        oa.log(f"Evaluator error: {e}")
        return 0.0


def run(prompt: str, max_calls: int = 50, run_dir: str | None = None) -> dict:
    """Run GEPA optimize_anything on a prompt string."""
    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=max_calls,
            track_best_outputs=True,
            display_progress_bar=True,
            raise_on_exception=False,
            run_dir=run_dir,
            candidate_selection_strategy="pareto",
            frontier_type="hybrid",
        ),
        reflection=ReflectionConfig(
            reflection_lm=REFLECTION_MODEL,
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

    best = result.best_candidate
    if isinstance(best, dict):
        best = next(iter(best.values()))

    seed_score = result.val_aggregate_scores[0] if result.val_aggregate_scores else 0.0
    best_score = result.val_aggregate_scores[result.best_idx] if result.val_aggregate_scores else 0.0

    return {
        "best_prompt": best,
        "best_score": round(best_score, 4),
        "seed_score": round(seed_score, 4),
        "improvement": round(best_score - seed_score, 4),
        "candidates_explored": result.num_candidates,
        "total_evaluations": result.total_metric_calls,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GEPA prompt optimizer for Claude Code")
    parser.add_argument("prompt", nargs="?", help="Prompt to optimize (or pipe via stdin)")
    parser.add_argument("--max-calls", type=int, default=50, help="Max evaluation calls (default: 50)")
    parser.add_argument("--run-dir", default=None, help="Checkpoint directory for resuming")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    prompt = args.prompt
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        parser.error("No prompt provided. Pass as argument or pipe via stdin.")

    print(f"Optimizing prompt ({args.max_calls} max evals)...\n", file=sys.stderr)

    result = run(prompt, max_calls=args.max_calls, run_dir=args.run_dir)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Score: {result['seed_score']:.4f} -> {result['best_score']:.4f}  "
              f"(+{result['improvement']:.4f})")
        print(f"Candidates explored: {result['candidates_explored']}")
        print(f"Total evaluations: {result['total_evaluations']}")
        print(f"\n{'='*60}")
        print("OPTIMIZED PROMPT:")
        print(f"{'='*60}\n")
        print(result["best_prompt"])


if __name__ == "__main__":
    main()
