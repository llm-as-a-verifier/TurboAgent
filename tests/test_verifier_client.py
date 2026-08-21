"""Verifier-backend routing: the right client and the right model id.

Run: python tests/test_verifier_client.py   (or: pytest tests/)
No network — every client constructor here only builds an object.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from turbo_agent.utils import (  # noqa: E402
    Config, ModelConfig, build_verifier_client, verifier_model_id,
)
from turbo_agent.verifier.verifier import Verifier  # noqa: E402


def test_model_id_strips_the_backend_prefix():
    assert verifier_model_id(ModelConfig("deepseek/deepseek-v4-flash")) == \
        "deepseek-v4-flash"
    assert verifier_model_id(ModelConfig("gemini/gemini-2.5-flash")) == \
        "gemini-2.5-flash"
    assert verifier_model_id(ModelConfig("openai/gpt-4o")) == "gpt-4o"
    # A bare name (a local vLLM server's served id) is passed through.
    assert verifier_model_id(ModelConfig("Qwen/Qwen3.5-9B")) == "Qwen/Qwen3.5-9B"


def test_deepseek_prefix_builds_a_tagged_deepseek_client():
    client = build_verifier_client(
        ModelConfig("deepseek/deepseek-v4-flash", api_key="test-key"))
    # The tag is what makes llm-verifier read DeepSeek's sampled score tags
    # instead of the vLLM-only prefill trick.
    assert getattr(client, "_llm_verifier_deepseek", False) is True
    # The prefix must be stripped before it reaches the API.
    assert client._llm_verifier_model == "deepseek-v4-flash"
    assert "api.deepseek.com" in str(client.base_url)


def test_base_url_builds_an_openai_compatible_client():
    client = build_verifier_client(
        ModelConfig("Qwen/Qwen3.5-9B", api_key="test-key",
                    base_url="http://localhost:8000/v1"))
    assert getattr(client, "_llm_verifier_deepseek", False) is False
    assert "localhost:8000" in str(client.base_url)


def test_claude_is_rejected_as_a_verifier():
    """The Messages API returns no logprobs, so a Claude verifier must fail
    loudly rather than silently reaching the Gemini branch."""
    cfg = ModelConfig("anthropic/claude-opus-4-5", api_key="test-key")
    for call in (lambda: verifier_model_id(cfg),
                 lambda: build_verifier_client(cfg)):
        try:
            call()
        except ValueError as e:
            assert "logprob" in str(e)
            assert "backend model" in str(e)
        else:
            raise AssertionError("a Claude verifier should raise")

    # It must fail when the Verifier is constructed — at proxy startup — not
    # on the first scoring call.
    try:
        Verifier(_verifier_config("anthropic/claude-opus-4-5"))
    except ValueError:
        pass
    else:
        raise AssertionError("constructing a Claude verifier should raise")


def test_gemini_without_a_key_defers_to_the_environment():
    assert build_verifier_client(ModelConfig("gemini/gemini-2.5-flash")) is None


def test_verifier_model_id_is_correct_before_the_client_is_built():
    """Regression: `llm_verifier.select(model=..., client=...)` evaluates its
    keyword arguments left to right, so the model id must not depend on the
    client property having run first."""
    verifier = Verifier(_verifier_config("deepseek/deepseek-v4-flash"))
    assert verifier.model_id == "deepseek-v4-flash"  # no client access yet


def test_config_resolves_base_url_from_the_environment():
    os.environ["TEST_VERIFIER_BASE_URL"] = "http://localhost:9001/v1"
    yaml_text = """
backend:
  models:
    - name: gemini/gemini-2.5-flash
      api_key: test
verifier:
  model:
    name: openai/local-model
    base_url: $TEST_VERIFIER_BASE_URL
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "turbo-agent.yaml"
        path.write_text(yaml_text)
        cfg = Config(str(path))
        assert cfg.verifier_config.model.base_url == "http://localhost:9001/v1"


def _verifier_config(model_name):
    from turbo_agent.utils import (CriterionConfig, PivotTournamentConfig,
                                   VerifierConfig)
    return VerifierConfig(
        model=ModelConfig(model_name, api_key="test-key"),
        method=PivotTournamentConfig(
            criteria=[CriterionConfig(name="Task Success", description="d")]),
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
