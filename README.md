# Turbo Agent

![Turbo Agent visualizer](screenshot.png)

Turbo Agent is the Claude Code plugin for LLM-as-a-Verifier. It implements an LLM API proxy that improves response quality through concurrent inference, verification, and refinement. It sits between your client (Claude Code, Codex, etc.) and the LLM provider, sending multiple parallel requests and selecting the best response with a **Probabilistic Pivot Tournament (PPT)** scored by a fine-grained logprob verifier.

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

Verification uses the pivot tournament from the [`llm-verifier`](https://pypi.org/project/llm-verifier/) package to pick the best of `N` candidates.

## Install

```bash
pip install turbo-agent
```

Or from source:

```bash
pip install -e .
```

## Setup

For turbo agent to work, you need a `turbo-agent.yaml`. You can copy the reference file in this repo.

`turbo-agent.yaml` references keys with `$VAR_NAME` syntax. The recommended way to provide them is a `.env` file in the project root (next to `turbo-agent.yaml`) — the proxy loads it automatically on startup. Copy the committed template and fill in your keys:

```bash
cp .env.example .env
# then edit .env
```

```bash
# .env
VERTEX_API_KEY=your-vertex-key     # preferred for Gemini 2.5 logprobs (verifier)
# GEMINI_API_KEY=your-gemini-key     # used by gemini/ models (AI Studio)
# OPENAI_API_KEY=...               # only if you route to openai/ models
# ANTHROPIC_API_KEY=...            # only if you route to anthropic/ models
```

`.env` is gitignored; `.env.example` is committed as the template. Keys already
exported in your shell environment work too and take nothing extra. The verifier
and progress monitor use Gemini **logprobs**, which are best served by a Vertex
AI key (`VERTEX_API_KEY` + `provider: vertex_ai` in the config); a plain
`GEMINI_API_KEY` also works for the `gemini/` backend models.

Verify your keys are valid:

```bash
turbo-agent check
```

It checks every supported provider (Gemini, Vertex AI, OpenAI, Anthropic) and reports each with ✅ / ❌ / ⚠️ / ⚪️, flagging which keys your config actually uses.

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

## Caching Support

Turbo Agent supports **prompt caching** for qualifying providers. See [`CACHING_REVIEW.md`](./CACHING_REVIEW.md) for detailed algorithm validation and implementation status.

### Current Status by Provider

| Provider | Support | Status |
|----------|---------|--------|
| **Anthropic** | Prompt Caching | ✅ Pass-through (client-initiated) |
| **OpenAI** | None | ⚠️ Not supported; silently ignored |
| **Google Gemini** | CachedContent API | ❌ Disabled by design |
| **Vertex AI** | CachedContent API | ❌ Disabled by design |

**Anthropic (Claude)**: The proxy preserves `cache_control` metadata from client requests and passes it through to the Anthropic API. Include `cache_control: {type: "ephemeral"}` in your message blocks to enable caching.

**OpenAI**: Prompt caching is not yet supported by OpenAI models. Cache control headers in requests are silently ignored.

**Gemini/VertexAI**: Caching is intentionally disabled due to API incompatibility. Gemini's `CachedContent` API requires a 2-phase workflow (separate resource creation) that conflicts with TurboAgent's single-request model. Implementing this would require significant architectural changes.

### Future Improvements

- [ ] Add `caching` configuration section to `turbo-agent.yaml` for provider-specific settings
- [ ] Implement cache telemetry in request logs (creation/read tokens)
- [ ] Support Gemini caching with 2-phase initialization (high effort, requires redesign)
- [ ] Add warnings when clients request unsupported caching features

See [`CACHING_REVIEW.md`](./CACHING_REVIEW.md) for detailed recommendations and implementation roadmap.

## Visualizer

A built-in web UI at `http://localhost:8888/visualizer` shows the pipeline DAG for each request — context refinement, all candidate responses, the pairwise tournament comparisons and scores, and the final selection.

To build the frontend (requires Node.js):

```bash
cd frontend
yarn install
yarn build
```

## Publish to PyPI

```bash
cd frontend && yarn build && cd ..
pip install build twine
rm -rf dist
python -m build
twine check dist/*
twine upload dist/*
```