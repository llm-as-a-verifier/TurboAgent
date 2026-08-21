"""Build the llm-verifier client for a configured verifier model.

llm-verifier scores with token logprobs, and every backend exposes them
differently: Gemini through Vertex AI, DeepSeek through its own sampled score
tags, and vLLM / SGLang / OpenAI through the OpenAI-compatible ``logprobs``
field. Routing on the model-name prefix lets the verifier and the progress
monitor point at any of them instead of assuming Gemini.

The model id and the client are resolved separately so the id is available
without paying for a client the caller may never use.
"""

from typing import Any, Optional

from .config import ModelConfig

_PREFIXES = ("deepseek/", "openai/", "gemini/", "anthropic/")


def _ensure_verifiable(cfg: ModelConfig) -> None:
    """Reject a backend that cannot score.

    The Messages API exposes no logprobs, and the fine-grained reward is an
    expectation over the score-token distribution — there is nothing to take
    an expectation over. Raising here rather than in the lazy client property
    means a bad config fails at startup, not on the first request.
    """
    if cfg.name.startswith("anthropic/"):
        raise ValueError(
            f"'{cfg.name}' cannot be a verifier: the Anthropic Messages API "
            "does not return token logprobs, which the fine-grained reward "
            "needs. Claude works as a backend model — put it under "
            "`backend.models` and verify with a logprob backend "
            "(deepseek/, openai/, or gemini/ on Vertex).")


def verifier_model_id(cfg: ModelConfig) -> str:
    """The bare model name to send to the verifier backend."""
    _ensure_verifiable(cfg)
    for prefix in _PREFIXES:
        if cfg.name.startswith(prefix):
            return cfg.name.removeprefix(prefix)
    return cfg.name


def build_verifier_client(cfg: ModelConfig) -> Optional[Any]:
    """The llm-verifier client for ``cfg``.

    ``None`` means "no client configured here" and lets llm-verifier build one
    from the environment, which is the pre-existing behaviour for Gemini.
    """
    model_id = verifier_model_id(cfg)

    if cfg.name.startswith("deepseek/"):
        from llm_verifier.fine_grained_reward import create_deepseek_client
        # api_key=None falls back to DEEPSEEK_API_KEY in the environment.
        return create_deepseek_client(api_key=cfg.api_key, model=model_id)

    if cfg.name.startswith("openai/") or cfg.base_url:
        from llm_verifier.fine_grained_reward import create_openai_client
        # base_url=None falls back to OPENAI_BASE_URL in the environment.
        return create_openai_client(base_url=cfg.base_url,
                                    api_key=cfg.api_key)

    if not cfg.api_key:
        return None

    from google import genai
    if cfg.provider == "vertex_ai":
        return genai.Client(vertexai=True, api_key=cfg.api_key)
    return genai.Client(api_key=cfg.api_key)
