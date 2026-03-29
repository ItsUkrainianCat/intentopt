# autoimprover

SOTA autonomous prompt optimizer for Claude Code using DSPy + GEPA.

Uses GEPA's Genetic-Pareto optimization to evolve prompts via execution trace reflection, integrated with DSPy's programming framework for foundation models.

## Setup

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
```

## Stack

- **DSPy 3.1** — programming framework for foundation models
- **GEPA 0.0.26** — Genetic-Pareto optimizer (trace-driven prompt/agent evolution)
- **Anthropic SDK** — Claude API access
- **LiteLLM** — multi-provider LLM routing
