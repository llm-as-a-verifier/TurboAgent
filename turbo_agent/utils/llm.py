"""
LLM completion wrapper using litellm for multi-provider routing and format
conversion. litellm handles model prefix routing (gemini/, openai/, anthropic/),
provider-specific API formatting, and cross-provider tool call compatibility.
"""

import asyncio
import json
import os
import signal
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


_CLI_MODELS = {
    "claude-cli/default": "claude",
    "codex-cli/default": "codex",
}
_CLI_TIMEOUT_SECONDS = 300
_SAFE_ENV_KEYS = frozenset(
    {
        "COLORTERM",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "NO_COLOR",
        "PATH",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "USER",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    }
)
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "response": {"type": "string", "minLength": 1},
    },
    "required": ["response"],
    "additionalProperties": False,
}


def _child_env(provider: str) -> Dict[str, str]:
    """Return a default-deny environment with only subscription auth state."""
    allowed = set(_SAFE_ENV_KEYS)
    if provider == "claude":
        allowed.update({"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR"})
    elif provider == "codex":
        allowed.add("CODEX_HOME")
    return {key: value for key, value in os.environ.items() if key in allowed}


def _serialize_messages(messages: List[Dict[str, Any]]) -> str:
    """Render text-only chat messages into one stateless CLI prompt."""
    sections = [
        "You are a read-only advisory model in a best-of-N comparison.",
        "Inspect the current checkout only when useful. Do not modify files.",
        "Return only the next assistant response.",
    ]
    for message in messages:
        role = str(message.get("role") or "unknown").upper()
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    raise NotImplementedError(
                        "CLI providers support text message blocks only"
                    )
                parts.append(str(block.get("text", "")))
            content = "\n".join(parts)
        if not isinstance(content, str):
            raise NotImplementedError("CLI providers support text messages only")
        if content:
            sections.append(f"{role}:\n{content}")
    if len(sections) == 3:
        raise ValueError("CLI completion requires at least one non-empty message")
    return "\n\n".join(sections) + "\n"


def _safe_error(stderr: bytes, env: Dict[str, str]) -> str:
    rendered = stderr.decode("utf-8", errors="replace").strip()
    for key in ("CLAUDE_CODE_OAUTH_TOKEN",):
        secret = env.get(key)
        if secret:
            rendered = rendered.replace(secret, "[REDACTED_SECRET]")
    return rendered[:1000]


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    """Terminate and reap a CLI process group if it is still running."""
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        await process.wait()
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=1)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def _run_cli(
    command: List[str], prompt: str, env: Dict[str, str]
) -> tuple[bytes, bytes]:
    """Run one subscription CLI and terminate its process group on timeout."""
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=os.getcwd(),
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(prompt.encode("utf-8")),
            timeout=_CLI_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        await _terminate_process_group(process)
        raise TimeoutError(
            f"{command[0]} timed out after {_CLI_TIMEOUT_SECONDS} seconds"
        ) from exc
    except asyncio.CancelledError:
        cleanup = asyncio.create_task(_terminate_process_group(process))
        await asyncio.shield(cleanup)
        raise
    if process.returncode != 0:
        detail = _safe_error(stderr or stdout, env)
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"{command[0]} exited {process.returncode}{suffix}")
    return stdout, stderr


def _usage(prompt_tokens: int = 0, completion_tokens: int = 0) -> Dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _response(model: str, content: str, usage: Dict[str, int]) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-cli-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def _parse_claude(stdout: bytes, model: str) -> Dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Claude returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("is_error") is not False:
        raise RuntimeError("Claude did not return a successful result envelope")
    if payload.get("type") != "result" or payload.get("subtype") != "success":
        raise RuntimeError("Claude result envelope was not successful")
    model_usage = payload.get("modelUsage")
    if not isinstance(model_usage, dict) or not model_usage:
        raise RuntimeError("Claude response omitted modelUsage proof")
    structured = payload.get("structured_output")
    content = structured.get("response") if isinstance(structured, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Claude response omitted structured_output.response")
    prompt_tokens = 0
    completion_tokens = 0
    for raw in model_usage.values():
        if not isinstance(raw, dict):
            continue
        prompt_tokens += sum(
            int(raw.get(key, 0) or 0)
            for key in ("inputTokens", "cacheReadInputTokens", "cacheCreationInputTokens")
        )
        completion_tokens += int(raw.get("outputTokens", 0) or 0)
    return _response(model, content.strip(), _usage(prompt_tokens, completion_tokens))


def _parse_codex(
    stdout: bytes, output_path: Path, model: str
) -> Dict[str, Any]:
    events = []
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Codex returned invalid JSONL: {exc}") from exc
        if isinstance(event, dict):
            events.append(event)
    if any(event.get("type") == "turn.failed" for event in events):
        raise RuntimeError("Codex reported turn.failed")
    completed = [event for event in events if event.get("type") == "turn.completed"]
    if not completed:
        raise RuntimeError("Codex response omitted turn.completed proof")
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Codex returned invalid structured output: {exc}") from exc
    content = payload.get("response") if isinstance(payload, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Codex response omitted response")
    raw_usage = completed[-1].get("usage")
    raw_usage = raw_usage if isinstance(raw_usage, dict) else {}
    prompt_tokens = int(raw_usage.get("input_tokens", 0) or 0)
    completion_tokens = int(raw_usage.get("output_tokens", 0) or 0)
    return _response(model, content.strip(), _usage(prompt_tokens, completion_tokens))


def _validate_cli_options(
    *,
    api_key: Optional[str],
    tools: Optional[List[Any]],
    tool_choice: Optional[Any],
    response_format: Optional[Any],
    logprobs: Optional[bool],
    top_logprobs: Optional[int],
    n: Optional[int],
    unsupported_controls: Dict[str, Any],
) -> None:
    if api_key:
        raise ValueError("CLI providers reject API keys")
    if tools or tool_choice is not None:
        raise NotImplementedError("CLI providers do not expose client tool calls")
    if response_format is not None:
        raise NotImplementedError("CLI providers use a fixed response schema")
    if logprobs or top_logprobs is not None:
        raise NotImplementedError("CLI candidate providers do not expose logprobs")
    if n not in (None, 1):
        raise ValueError("TurboAgent must create CLI candidates with num_candidates")
    enabled = [name for name, value in unsupported_controls.items() if value is not None]
    if enabled:
        raise NotImplementedError(
            "CLI providers cannot honor these completion controls: "
            + ", ".join(sorted(enabled))
        )


async def _cli_completion(
    *,
    model: str,
    provider: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str],
    tools: Optional[List[Any]],
    tool_choice: Optional[Any],
    response_format: Optional[Any],
    logprobs: Optional[bool],
    top_logprobs: Optional[int],
    n: Optional[int],
    unsupported_controls: Dict[str, Any],
) -> Dict[str, Any]:
    _validate_cli_options(
        api_key=api_key,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        n=n,
        unsupported_controls=unsupported_controls,
    )
    prompt = _serialize_messages(messages)
    env = _child_env(provider)
    compact_schema = json.dumps(_RESPONSE_SCHEMA, separators=(",", ":"))
    if provider == "claude":
        command = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--json-schema",
            compact_schema,
            "--permission-mode",
            "plan",
            "--safe-mode",
            "--tools",
            "Read,Grep,Glob",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--no-chrome",
        ]
        stdout, _ = await _run_cli(command, prompt, env)
        return _parse_claude(stdout, model)

    with tempfile.TemporaryDirectory(prefix="turbo-agent-codex-") as temp_dir:
        root = Path(temp_dir)
        schema_path = root / "response-schema.json"
        output_path = root / "response.json"
        schema_path.write_text(compact_schema, encoding="utf-8")
        command = [
            "codex",
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--disable",
            "hooks",
            "--disable",
            "multi_agent",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "-c",
            'approval_policy="never"',
            "-c",
            'model_provider="openai"',
            "-c",
            'web_search="disabled"',
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "--json",
            "--color",
            "never",
            "-C",
            os.getcwd(),
            "-",
        ]
        stdout, _ = await _run_cli(command, prompt, env)
        return _parse_codex(stdout, output_path, model)


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
    provider = _CLI_MODELS.get(model)
    if provider is not None:
        return await _cli_completion(
            model=model,
            provider=provider,
            messages=messages,
            api_key=api_key,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            n=n,
            unsupported_controls={
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
                "seed": seed,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "logit_bias": logit_bias,
                "stream_options": stream_options,
                "reasoning_effort": reasoning_effort,
                "thinking_budget": thinking_budget,
            },
        )
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
    if model in _CLI_MODELS:
        raise NotImplementedError(
            f"{model} supports verified non-streaming completions only"
        )
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
