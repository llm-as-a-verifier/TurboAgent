# Turbo Agent

Claude Code plugin for LLM-as-a-Verifier. It implements an LLM API proxy that improves response quality through concurrent inference, verification, and refinement. It sits between your client (Claude Code, Codex, etc.) and the LLM provider, sending multiple parallel requests and selecting the best response with a **Pivot Preference Tournament (PPT)** scored by a fine-grained logprob verifier.

```
Client request
    │
[Context Refinement]   (optional) rewrite/augment the system prompt for clarity
    │
[Concurrent Inference] send N parallel candidates to the backend model
    │
[Verification]         pivot tournament over the candidates, pick the best one
    │
Best response → Client
```

The verifier scores **directed pairs** (candidate `a` in slot A, `b` in slot B) with Gemini logprobs over a 20-token A–T scale, turns each pair's two fine-grained rewards into a soft Bradley-Terry win, and aggregates them through PPT — a ring pass, pivots = empirical leaders, then pivot rounds — to pick the best of `N` in `O(N·k)` comparisons instead of the `O(N²)` of full round-robin.

## Install

```bash
pip install -e .
```

The verifier uses Gemini logprobs, so set `GEMINI_API_KEY` (or `VERTEX_API_KEY`
with `provider: vertex_ai` in the config) in the environment or a `.env` file
next to `turbo-agent.yaml`.

## Run

```bash
turbo-agent                   # default port 8888
turbo-agent -p 9000           # custom port
```

### Use with Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:8888 claude
```

### Use with OpenAI-compatible clients

```bash
export OPENAI_API_BASE=http://localhost:8888/v1
```

## Configuration

Edit `turbo-agent.yaml`. API keys can reference environment variables with `$VAR_NAME` syntax. See the reference `turbo-agent.yaml` file for reference and usage.

Uncomment the optional `context:` section in `turbo-agent.yaml` to enable
context refinement.

### Model prefixes

| Prefix | Provider |
|--------|----------|
| `gemini/` | Google Gemini |
| `openai/` | OpenAI |
| `anthropic/` | Anthropic |
| (none) | OpenAI-compatible endpoint |

## API endpoints

| Endpoint | Format |
|----------|--------|
| `POST /v1/messages` | Anthropic |
| `POST /v1/chat/completions` | OpenAI |
| `GET /v1/models` | OpenAI |
| `GET /visualizer` | Pipeline visualizer UI |
| `*` | Upstream passthrough to api.anthropic.com |

## Visualizer

A built-in web UI at `http://localhost:8888/visualizer` shows the pipeline DAG for each request — context refinement, all candidate responses, the pairwise tournament comparisons and scores, and the final selection.

To build the frontend (requires Node.js):

```bash
cd frontend
yarn install
yarn build
```