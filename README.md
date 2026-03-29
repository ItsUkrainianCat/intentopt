# optimizer

SOTA autonomous prompt optimizer for Claude Code using GEPA's Genetic-Pareto evolutionary search.

**Runs entirely on Claude Code Max plan** — no API keys needed. All LLM calls route through `claude -p`.

## Usage

In Claude Code CLI:
```
/optimize Write a NixOS module that sets up a mesh network
```

Or directly:
```bash
cd ~/autoimprover && source .venv/bin/activate
export LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
echo "your prompt here" | python -m autoimprover --json
```

## How it works

1. GEPA scores your prompt via Claude-as-judge (clarity, specificity, effectiveness, autonomy, tool awareness, robustness)
2. Reflection LM reads execution traces (ASI) and proposes mutations
3. Pareto-optimal candidate selection evolves the prompt over ~30 iterations
4. Returns the highest-scoring variant

## Stack

- **GEPA 0.1.1** — Genetic-Pareto optimizer with `optimize_anything` API
- **Claude CLI** (`claude -p`) — LLM backend via Max plan
