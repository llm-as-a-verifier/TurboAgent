import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from turbo_agent.utils.config import Config, CriterionConfig, ModelConfig, PivotTournamentConfig
from turbo_agent.verifier import verifier as verifier_module
from turbo_agent.verifier.verifier import StrictGeminiClient, Verifier


def _gemini_response(*, valid_score_logprobs=True, include_logprobs=True):
    tokens = [
        "analysis",
        "<score_A>",
        "A",
        "</score_A>",
        "<score_B>",
        "T",
        "</score_B>",
    ]
    chosen = [SimpleNamespace(token=token) for token in tokens]
    top = []
    for index, token in enumerate(tokens):
        alternatives = [SimpleNamespace(token=token, log_probability=0.0)]
        if index in (2, 5) and not valid_score_logprobs:
            alternatives = [SimpleNamespace(token="not-a-score", log_probability=0.0)]
        top.append(SimpleNamespace(candidates=alternatives))
    result = (
        SimpleNamespace(chosen_candidates=chosen, top_candidates=top)
        if include_logprobs
        else None
    )
    candidate = SimpleNamespace(logprobs_result=result)
    return SimpleNamespace(candidates=[candidate], text="scored")


class _Models:
    def __init__(self, response):
        self.response = response

    def generate_content(self, **_kwargs):
        return self.response


class StrictGeminiClientTests(unittest.TestCase):
    def test_counts_actual_calls_with_both_score_tag_distributions(self):
        inner = SimpleNamespace(models=_Models(_gemini_response()))
        client = StrictGeminiClient(inner)

        client.models.generate_content(model="gemini-3.5-flash")

        self.assertEqual(client.validated_calls, 1)

    def test_rejects_missing_or_unusable_score_logprobs(self):
        invalid = (
            _gemini_response(include_logprobs=False),
            _gemini_response(valid_score_logprobs=False),
        )
        for response in invalid:
            with self.subTest(response=response):
                client = StrictGeminiClient(SimpleNamespace(models=_Models(response)))
                with self.assertRaisesRegex(RuntimeError, "score-tag logprobs"):
                    client.models.generate_content(model="gemini-3.5-flash")


class StrictVerifierTests(unittest.TestCase):
    def _config(self):
        return verifier_module.VerifierConfig(
            model=ModelConfig(
                name="gemini/gemini-3.5-flash",
                api_key="fake-gemini-key",
            ),
            method=PivotTournamentConfig(
                pivots=1,
                n_verifications=1,
                seed=0,
                criteria=[
                    CriterionConfig(
                        name="Task Success",
                        description="Prefer the answer that solves the task.",
                    )
                ],
            ),
            majority_voting=False,
            require_logprobs=True,
        )

    def test_run_select_requires_raise_mode_and_returns_actual_call_count(self):
        verifier = Verifier(self._config())
        verifier._client = SimpleNamespace(models=_Models(_gemini_response()))
        seen = {}

        def fake_select(_history, _actions, **kwargs):
            seen.update(kwargs)
            kwargs["client"].models.generate_content(model=kwargs["model"])
            Path(kwargs["cache"]).write_text(
                json.dumps(
                    {
                        "task_success|task|0,1|0": {
                            "score_A": 0.8,
                            "score_B": 0.2,
                        }
                    }
                )
            )
            return SimpleNamespace(index=0, scores=[0.8, 0.2], n_comparisons=1)

        with mock.patch.object(verifier_module.llm_verifier, "select", fake_select):
            _result, pair_scores, logprob_calls = verifier._run_select(
                "history", ["answer a", "answer b"]
            )

        self.assertEqual(seen["on_error"], "raise")
        self.assertIsInstance(seen["client"], StrictGeminiClient)
        self.assertEqual(logprob_calls, 1)
        self.assertTrue(pair_scores)

    def test_config_parses_opt_in_strict_logprob_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "turbo-agent.yaml"
            path.write_text(
                """
backend:
  models:
    - name: claude-cli/default
    - name: codex-cli/default
verifier:
  model:
    name: gemini/gemini-3.5-flash
    api_key: fake
  require_logprobs: true
  method:
    name: pivot_tournament
""".strip()
            )

            config = Config(str(path))

        self.assertTrue(config.verifier_config.require_logprobs)


if __name__ == "__main__":
    unittest.main()
