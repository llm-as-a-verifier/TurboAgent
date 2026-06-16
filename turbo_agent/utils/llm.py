"""
LLM completion wrapper using litellm for multi-provider routing and format
conversion. litellm handles model prefix routing (gemini/, openai/, anthropic/),
provider-specific API formatting, and cross-provider tool call compatibility.
"""

from typing import Any, Dict, List, Optional

import litellm

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


def _build_kwargs(
    model: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[Any] = None,
    stream: Optional[bool] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    tools: Optional[List[Any]] = None,
    tool_choice: Optional[Any] = None,
    response_format: Optional[Any] = None,
    seed: Optional[int] = None,
    n: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[Dict[str, int]] = None,
    stream_options: Optional[Any] = None,
    reasoning_effort: Optional[str] = None,
    thinking_budget: Optional[int] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if api_key is not None:
        kwargs["api_key"] = api_key
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if stop is not None:
        kwargs["stop"] = stop
    if stream is not None:
        kwargs["stream"] = stream
    if logprobs is not None:
        kwargs["logprobs"] = logprobs
    if top_logprobs is not None:
        kwargs["top_logprobs"] = top_logprobs
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format
    if seed is not None:
        kwargs["seed"] = seed
    if n is not None:
        kwargs["n"] = n
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty
    if logit_bias is not None:
        kwargs["logit_bias"] = logit_bias
    if stream_options is not None:
        kwargs["stream_options"] = stream_options
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if thinking_budget is not None:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    return kwargs


async def llm_completion(
    model: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[Any] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    tools: Optional[List[Any]] = None,
    tool_choice: Optional[Any] = None,
    response_format: Optional[Any] = None,
    seed: Optional[int] = None,
    n: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[Dict[str, int]] = None,
    stream_options: Optional[Any] = None,
    reasoning_effort: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    **kwargs: Any,
) -> dict:
    """Non-streaming LLM completion. Returns response as a dict."""
    params = _build_kwargs(
        model, messages,
        api_key=api_key,
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        seed=seed,
        n=n,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        stream_options=stream_options,
        reasoning_effort=reasoning_effort,
        thinking_budget=thinking_budget,
    )
    response = await litellm.acompletion(**params)
    return response.model_dump()


async def llm_stream_completion(
    model: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[Any] = None,
    tools: Optional[List[Any]] = None,
    tool_choice: Optional[Any] = None,
    stream_options: Optional[Any] = None,
    reasoning_effort: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Streaming LLM completion. Returns an async iterable of chunk objects."""
    params = _build_kwargs(
        model, messages,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        stream=True,
        tools=tools,
        tool_choice=tool_choice,
        stream_options=stream_options,
        reasoning_effort=reasoning_effort,
        thinking_budget=thinking_budget,
    )
    return await litellm.acompletion(**params)
