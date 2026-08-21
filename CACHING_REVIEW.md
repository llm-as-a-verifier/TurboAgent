# Caching Algorithm Review: TurboAgent Multi-Provider Support

## Executive Summary

The TurboAgent library currently **preserves cache_control metadata** during format conversion between Anthropic and OpenAI formats, but **does not actively set or configure caching for any provider**. Caching only works if the client initiates it by including `cache_control` in the request.

Each provider has fundamentally different caching mechanisms, and **correct implementation requires provider-specific handling**. The disabled litellm transform (line 19 in `turbo_agent/utils/llm.py`) is a workaround for Gemini's API limitations, not a caching strategy.

---

## Current Implementation Analysis

### 1. **Anthropic (Claude)**
**Status**: ✅ Passthrough support only  
**Location**: `turbo_agent/utils/conversion.py` (lines 23-24, 34-35, 72-73, 147-148, 152-153, 160-163, 221-222, 289-290)

**How it works**:
- `cache_control` fields are preserved during Anthropic → OpenAI format conversion
- If client sends `cache_control: {"type": "ephemeral"}` in text blocks, tools, or messages, it's retained
- litellm passes it to the Anthropic API as-is

**Algorithm correctness**: ✅ **CORRECT** for pass-through
- Anthropic's prompt caching uses `cache_control` blocks
- The conversion properly handles multi-block scenarios and preserves type information
- Works for text, images, and tool definitions

**Issues**:
- ❌ No automatic cache placement strategy (must come from client)
- ❌ No configuration option to enable/tune caching in `turbo-agent.yaml`
- ⚠️ Cache priming strategy in `backend.py:174-200` runs sequential primer before parallel requests — this is **incompatible** with Anthropic's cache write semantics:
  - Anthropic writes cache during prefill, visible in subsequent requests **in the same conversation**
  - But `_gather_completions()` creates **new independent requests**, so cache won't transfer between them
  - Each primer request must hit the same API endpoint (Anthropic has single endpoint), so this *might* work if litellm uses connection pooling, but it's fragile

---

### 2. **OpenAI**
**Status**: ❌ No caching support  
**Location**: `turbo_agent/proxy/backend.py:610-633`, `turbo_agent/utils/llm.py:93-141`

**How it works**:
- OpenAI models (gpt-4o, etc.) do **not** support prompt caching as of Feb 2025
- Cache control fields are **ignored** if sent by client
- The code simply converts and passes through to litellm → OpenAI API

**Algorithm correctness**: ⚠️ **INCOMPLETE**
- No error/warning if client tries to use `cache_control`
- Should either:
  1. Reject requests with `cache_control` (fail-fast)
  2. Silently strip `cache_control` and log a warning (graceful degradation)
  3. Auto-implement a fallback caching layer (out of scope for this review)

**Recommendation**: Add a check in `_build_openai_params()` to warn/reject if `cache_control` is present

---

### 3. **Google Gemini** 
**Status**: ⚠️ Partially broken by disabled transform  
**Location**: `turbo_agent/utils/llm.py:19`, `turbo_agent/verifier/verifier.py`, `turbo_agent/proxy/backend.py:63-64`

**How it works**:
1. Client sends Anthropic format → converted to OpenAI format (via `conversion.py`)
2. Anthropic `cache_control` blocks are converted to OpenAI `cache_control` objects
3. litellm **would** transform these to Gemini's `CachedContent` API calls
4. **BUT**: Line 19 disables this: `litellm.disable_anthropic_gemini_context_caching_transform = True`

**Algorithm correctness**: ❌ **BROKEN**

**Root cause**: 
The comment (lines 14-18 in `llm.py`) explains the disable:
```
Gemini API rejects: "CachedContent can not be used with GenerateContent request
setting system_instruction, tools or tool_config."
```

**This reveals a fundamental design problem**:
- Gemini's `CachedContent` API is a **two-step process**:
  1. Create a `CachedContent` resource with static context
  2. In a **separate** `generateContent` call (without system_instruction/tools), reference the cache
  
- But TurboAgent's workflow:
  1. Converts Anthropic request (which includes system_instruction and tools)
  2. Sends a single `generateContent` call with all that + cache reference
  3. Gemini API rejects this

**What should happen**:
- Gemini caching requires a 2-API-call flow that TurboAgent doesn't support
- The disable is a **necessary workaround**, not a caching strategy
- Caching for Gemini would need:
  1. Separate initialization phase to create `CachedContent` before any inference
  2. Configuration in `turbo-agent.yaml` to specify what to cache (system prompt size, token budget)
  3. Client code to reuse cache IDs across requests
  4. Fallback when cache expires (2 hours with Vertex AI)

---

### 4. **Vertex AI (Gemini via VertexAI)**
**Status**: ⚠️ Same as Gemini (broken by disabled transform)  
**Location**: `turbo_agent/verifier/verifier.py:89-91`, `turbo-agent.yaml:6,14-16`

**How it works**:
- Uses `google.genai.Client(vertexai=True)` with Vertex API key
- Same `CachedContent` API as Gemini, but with different authentication
- litellm passes through to `google.genai` SDK

**Algorithm correctness**: ❌ **SAME BROKEN ARCHITECTURE AS GEMINI**

**Additional constraint**: 
- Vertex AI CachedContent has a **2-hour TTL**
- Cache expiry isn't tracked, so reused cache IDs will fail silently after 2 hours
- Would need refresh logic in config or automatic cache invalidation

---

## Configuration Gap

**Current**: `turbo-agent.yaml` has no caching options
```yaml
backend:
  models:
    - name: anthropic/claude-3-5-sonnet
      temperature: 1
      # ❌ No: enable_caching: true
      # ❌ No: cache_strategy: "system_prompt"
      # ❌ No: cache_ttl: 3600
```

**Should support**:
```yaml
backend:
  models:
    - name: anthropic/claude-3-5-sonnet
      caching:
        enabled: true
        strategy: "system_and_context"  # or "system_only", "none"
        min_cached_tokens: 1024         # only cache if prefix >= N tokens
      
verifier:
  model:
    name: gemini/gemini-2.5-flash
    caching:
      enabled: true
      cache_id: "my-verifier-context"  # reuse across requests
      ttl: 3600                         # refresh if older
```

---

## Algorithm Validation by Provider

| Provider | Caching Type | Status | Issues |
|----------|--------------|--------|--------|
| **Anthropic** | Prompt Caching (ephemeral blocks) | ✅ Passthrough works | ❌ No automatic placement; ⚠️ Cache priming incompatible with multi-request architecture |
| **OpenAI** | None (not supported) | ⚠️ Ignored | ❌ No error; should warn or reject |
| **Gemini** | CachedContent API | ❌ Disabled by workaround | ❌ Requires 2-phase architecture; no config support |
| **VertexAI** | CachedContent API | ❌ Disabled by workaround | ⚠️ 2-hour TTL not tracked; no refresh logic |

---

## Recommendations

### Priority 1: Fix OpenAI (Low effort, high clarity)
```python
# In backend.py:610, add after _build_openai_params:
if openai_body.get("cache_control"):
    logger.warning(
        "Cache control requested but OpenAI does not support prompt caching. "
        "Set ignored."
    )
    # Or: raise ValueError("OpenAI does not support prompt caching")
```

### Priority 2: Document Anthropic cache priming incompatibility (Low effort, clarity)
The cache priming strategy in `_gather_completions()` (lines 174-200) assumes:
- Requests to the same model use the same connection/session
- Cache writes from primer are visible to parallel followers

For Anthropic, this *might* work due to connection pooling, but it's not guaranteed. 

**Recommendation**:
```python
# Add comment in backend.py:174-200
# Cache priming strategy notes:
# - Anthropic: Relies on litellm connection pooling. Cache writes during
#   primer prefill must be visible to parallel followers in same session.
#   Not guaranteed by Anthropic's API; use with caution.
# - Gemini: Cache disabled by design (see llm.py:19); priming has no effect.
# - OpenAI: No prompt caching; priming has no effect.
```

### Priority 3: Add Anthropic cache configuration (Medium effort)
```python
# config.py: Add to ModelConfig
@dataclass
class CachingConfig:
    enabled: bool = False
    strategy: str = "none"  # "none", "system_only", "system_and_context"
    min_tokens: int = 1024
    
# In _parse_model_params():
if model.get("caching"):
    params["cache_control"] = {
        "type": "ephemeral" if model["caching"].get("enabled") else "none"
    }
```

### Priority 4: Redesign Gemini/VertexAI caching (High effort)
**Not recommended** without a clear requirement, because:
1. Requires 2-phase architecture (cache creation, then generation)
2. Token budget management is complex
3. Cache expiry (2h for Vertex) needs refresh logic
4. Single-request isolation model doesn't map to Gemini's design

**If needed**, create a separate `CacheManager` class:
```python
class GeminiCacheManager:
    def __init__(self, client):
        self.client = client
    
    def get_or_create_cache(self, system_prompt, tools, cache_id=None):
        # Check if cache_id exists and is fresh (< 2h old)
        # If not, create new CachedContent resource
        # Return cache_id for use in generateContent
        pass
```

### Priority 5: Add cache telemetry (Low effort, high visibility)
```python
# In request logs, track:
{
  "caching": {
    "strategy": "anthropic_ephemeral",
    "cache_creation_tokens": 1024,
    "cache_read_tokens": 512,
    "cache_hits": 0,
  }
}
```

---

## Testing Checklist

- [ ] Anthropic: Send request with `cache_control: {type: "ephemeral"}` → verify header in logs
- [ ] Anthropic: Send request without `cache_control` → verify no cache header
- [ ] OpenAI: Send request with `cache_control` → verify warning logged, cache ignored
- [ ] Gemini: Send request with `cache_control` → verify transform disabled, no error
- [ ] VertexAI: Same as Gemini
- [ ] Multiple candidates: Verify cache priming runs sequentially, rest in parallel

---

## Summary of Correctness

1. **Anthropic**: ✅ Correctly preserves cache control metadata; ❌ needs auto-placement strategy and config support
2. **OpenAI**: ⚠️ Silently ignores unsupported feature; should warn or reject
3. **Gemini/VertexAI**: ❌ Caching disabled by design; no support for 2-phase architecture

**Recommendation**: Focus on Anthropic first (easiest win), add configuration support, then tackle Gemini if there's a clear use case requiring its 2-phase caching model.
