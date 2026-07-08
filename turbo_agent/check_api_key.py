#!/usr/bin/env python3
"""
Check every API provider TurboAgent supports and report whether its key works.

For each provider it loads the key from the environment (or the ``.env`` next to
``turbo-agent.yaml``), makes one minimal live request, and prints a status:

    ✅ working      key authenticated and the call succeeded
    ❌ failing      key is missing/expired/invalid (auth error)
    ⚠️  unverified   key looks set but the test call failed for another reason
    ⚪️ not set      no key in the environment

The Vertex AI check also confirms token logprobs are returned, since the
pivot-tournament verifier depends on them.

Usage:
    turbo-agent check
"""

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import httpx

# .env and turbo-agent.yaml are resolved from the directory the user runs in.
ROOT = Path.cwd()

# Keywords that mark an error as an authentication / key problem (vs. anything
# else, like a wrong model name or a transient network error).
_AUTH_HINTS = (
    "api key", "api_key", "apikey", "expired", "invalid", "unauthorized",
    "permission", "authenticat", "credential", "401", "403", "api_key_invalid",
)

# Status -> emoji
_EMOJI = {"ok": "✅", "fail": "❌", "warn": "⚠️ ", "skip": "⚪️"}
_LABEL = {"ok": "working", "fail": "failing", "warn": "unverified",
          "skip": "not set"}


def _looks_like_auth_error(message: str) -> bool:
    low = message.lower()
    return any(hint in low for hint in _AUTH_HINTS)


def _redact(text: str) -> str:
    """Hide anything that looks like a secret so the report is safe to share."""
    return re.sub(r"(sk|key|AIza)[-_A-Za-z0-9]{6,}", "<redacted>", text)


def _short_error(resp: httpx.Response) -> str:
    """One-line, redacted summary of an error response body."""
    detail = resp.text
    try:
        err = resp.json().get("error", {})
        detail = err.get("message", detail) if isinstance(err, dict) else detail
    except Exception:
        pass
    detail = " ".join(str(detail).split())[:160]
    return f"HTTP {resp.status_code}: {_redact(detail)}"


@dataclass
class CheckResult:
    name: str
    env_var: str
    status: str           # "ok" | "fail" | "warn" | "skip"
    detail: str
    roles: Tuple[str, ...] = ()

    def line(self) -> str:
        emoji = _EMOJI[self.status]
        label = _LABEL[self.status]
        role = f"  ({', '.join(self.roles)})" if self.roles else ""
        return (f"{emoji} {self.name:<14} [{self.env_var}] — "
                f"{label}: {self.detail}{role}")


class ProviderChecker:
    """Base class: resolve a key from the environment and validate it."""

    name: str = ""
    env_var: str = ""

    def __init__(self, roles: Tuple[str, ...] = ()):
        self.roles = roles

    def get_key(self) -> Optional[str]:
        key = os.environ.get(self.env_var, "").strip()
        return key or None

    def validate(self, key: str) -> Tuple[str, str]:
        """Return (status, detail). Override per provider."""
        raise NotImplementedError

    def run(self) -> CheckResult:
        key = self.get_key()
        if not key:
            return CheckResult(self.name, self.env_var, "skip",
                               "no key in environment", self.roles)
        try:
            status, detail = self.validate(key)
        except Exception as e:  # never let one provider abort the whole script
            msg = _redact(" ".join(f"{type(e).__name__}: {e}".split()))[:200]
            status = "fail" if _looks_like_auth_error(msg) else "warn"
            detail = msg
        return CheckResult(self.name, self.env_var, status, detail, self.roles)

    # -- shared helper for REST-based checks --------------------------------

    @staticmethod
    def _classify_http(resp: httpx.Response, ok_detail: str) -> Tuple[str, str]:
        if resp.status_code == 200:
            return "ok", ok_detail
        summary = _short_error(resp)
        if resp.status_code in (401, 403) or _looks_like_auth_error(resp.text):
            return "fail", summary
        return "warn", summary


class GeminiChecker(ProviderChecker):
    """Google Gemini API (the backend's ``gemini/`` models use this)."""

    name = "Gemini"
    env_var = "GEMINI_API_KEY"

    def validate(self, key: str) -> Tuple[str, str]:
        resp = httpx.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": key, "pageSize": 1},
            timeout=30.0,
        )
        return self._classify_http(resp, "models list reachable")


class OpenAIChecker(ProviderChecker):
    name = "OpenAI"
    env_var = "OPENAI_API_KEY"

    def validate(self, key: str) -> Tuple[str, str]:
        resp = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30.0,
        )
        return self._classify_http(resp, "models list reachable")


class AnthropicChecker(ProviderChecker):
    name = "Anthropic"
    env_var = "ANTHROPIC_API_KEY"

    def validate(self, key: str) -> Tuple[str, str]:
        resp = httpx.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            timeout=30.0,
        )
        return self._classify_http(resp, "models list reachable")


class VertexChecker(ProviderChecker):
    """Vertex AI via google-genai — the verifier's logprob path.

    A successful generation alone is not enough: the verifier needs token
    logprobs, so this also confirms they come back."""

    name = "Vertex AI"
    env_var = "VERTEX_API_KEY"

    def validate(self, key: str) -> Tuple[str, str]:
        from google import genai
        from google.genai.types import GenerateContentConfig, ThinkingConfig

        client = genai.Client(vertexai=True, api_key=key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="ping",
            config=GenerateContentConfig(
                max_output_tokens=1,
                temperature=0.0,
                response_logprobs=True,
                logprobs=5,
                thinking_config=ThinkingConfig(thinking_budget=0),
            ),
        )
        candidate = resp.candidates[0] # type: ignore
        has_logprobs = bool(
            getattr(candidate, "logprobs_result", None)
            and candidate.logprobs_result.top_candidates # type: ignore
        )
        if has_logprobs:
            return "ok", "generation + logprobs OK"
        return "warn", "generation OK but no logprobs returned (verifier needs them)"


def _roles_from_config() -> dict:
    """Map env-var name -> roles (which config slots reference it), so the
    report shows what each key is actually used for. Best-effort."""
    roles: dict = {}
    try:
        import yaml
        raw = yaml.safe_load((ROOT / "turbo-agent.yaml").read_text()) or {}
    except Exception:
        return roles

    def note(api_key_ref, role):
        if isinstance(api_key_ref, str) and api_key_ref.startswith("$"):
            roles.setdefault(api_key_ref[1:], []).append(role)

    for m in raw.get("backend", {}).get("models", []):
        note(m.get("api_key"), "backend")
    vmodel = raw.get("verifier", {}).get("model", {})
    note(vmodel.get("api_key"), "verifier")
    ctx = raw.get("context", {}).get("refinement_model", {})
    note(ctx.get("api_key"), "context")
    return roles


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(str(env_path), override=False)
    except Exception:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(),
                                      v.strip().strip('"').strip("'"))


def main() -> int:
    _load_dotenv()
    config_roles = _roles_from_config()

    checkers: List[ProviderChecker] = [
        GeminiChecker(),
        VertexChecker(),
        OpenAIChecker(),
        AnthropicChecker(),
    ]
    for c in checkers:
        c.roles = tuple(config_roles.get(c.env_var, ()))

    print("Checking TurboAgent provider API keys\n" + "─" * 60)
    results = [c.run() for c in checkers]
    for r in results:
        print(r.line())

    print("─" * 60)
    n_ok = sum(r.status == "ok" for r in results)
    n_fail = sum(r.status == "fail" for r in results)
    n_warn = sum(r.status == "warn" for r in results)
    n_skip = sum(r.status == "skip" for r in results)
    print(f"Summary: {n_ok} ✅  {n_fail} ❌  {n_warn} ⚠️   {n_skip} ⚪️")

    # Exit non-zero if any *configured* key (one a config slot references) is
    # failing — handy for scripting / CI.
    configured_fail = any(r.status == "fail" and r.roles for r in results)
    return 1 if configured_fail else 0


if __name__ == "__main__":
    sys.exit(main())
