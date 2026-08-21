# Caching Implementation Story

## Why We Did This

TurboAgent is a multi-provider LLM proxy that routes requests to Anthropic, OpenAI, Google Gemini, and Vertex AI. Each provider supports prompt caching differently—and they don't all work the same way. We needed to understand which caching algorithms were correctly implemented, which were broken, and what gaps existed.

When users enable concurrent inference (multiple candidate responses), costs multiply. Prompt caching is critical for reducing per-request costs and latency, but implementing it correctly across providers requires understanding their API constraints.

## What We Found

### ✅ Anthropic (Claude): Working
- **Status**: Prompt caching supported via `cache_control` blocks
- **Implementation**: Format conversion correctly preserves cache metadata
- **Gap**: No configuration support in `turbo-agent.yaml`; clients must send cache_control directly
- **Risk**: Cache priming strategy assumes connection pooling; not guaranteed by API

### ⚠️ OpenAI: Broken
- **Status**: No prompt caching support (as of Feb 2025)
- **Implementation**: Cache control fields silently ignored
- **Gap**: Should warn or reject unsupported feature

### ❌ Gemini & Vertex AI: Disabled by Design
- **Status**: Caching intentionally disabled
- **Why**: Gemini's `CachedContent` API requires a 2-phase workflow:
  1. Create a `CachedContent` resource (separate API call)
  2. Reference it in a `generateContent` call (no system_instruction/tools allowed)
- **Problem**: TurboAgent sends single requests with system prompt + tools—incompatible with Gemini's design
- **Impact**: No caching for Gemini backends; would require architectural redesign

## What We Did

1. **Created CACHING_REVIEW.md**
   - Detailed algorithm analysis per provider
   - Identified correctness issues and design gaps
   - Provided 5 priority recommendations (1-5)
   - Included testing checklist

2. **Updated README.md**
   - Added "Caching Support" section
   - Provider status matrix (✅ ⚠️ ❌)
   - Links to detailed review

3. **Implemented cache priming (backend.py)**
   - Sequential primer request per model
   - Parallel followers reuse cached prefix
   - Reduces concurrent cache-write overhead

4. **Preserved cache metadata (conversion.py)**
   - Cache control on text blocks, images, tools
   - Multi-block system prompts with breakpoints
   - Anthropic ↔ OpenAI format conversion

## Next Steps (Priority Order)

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | Warn/reject unsupported cache_control on OpenAI | Low | High clarity |
| 2 | Document cache priming assumptions | Low | Risk mitigation |
| 3 | Add `caching` config to `turbo-agent.yaml` | Medium | User control |
| 4 | Add cache telemetry to request logs | Low | Observability |
| 5 | Redesign Gemini support (2-phase) | High | Not recommended without clear use case |

## File Organization

**Note**: `CACHING_REVIEW.md` can be moved to `/docs` folder for better organization:

```
docs/
  CACHING_REVIEW.md      ← Detailed technical review
  CACHING_SUMMARY.md     ← This file (high-level overview)
```

## Key Takeaway

**Prompt caching works for Anthropic but requires configuration support. OpenAI needs error handling. Gemini/Vertex AI would need architectural changes.** The review provides a roadmap for each provider's implementation level.
