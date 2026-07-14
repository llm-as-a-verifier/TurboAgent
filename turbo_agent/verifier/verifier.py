"""
Verifier: best-of-N selection over candidate agent responses, delegated to
the `llm-verifier` package.

Given the conversation history and N candidate responses, `llm_verifier.select`
scores directed pairs (candidate `a` in slot A, `b` in slot B) with the
fine-grained logprob reward and aggregates them through a Probabilistic Pivot
Tournament (PPT) to pick the best candidate in O(N·k) comparisons rather than
the O(N^2) of full round-robin. This module wraps that call with TurboAgent's
config, majority-voting shortcut, and the per-comparison records the
visualizer displays.
"""

import asyncio
import json
import math
import os
import tempfile
import threading
from typing import List, Optional

import llm_verifier
from llm_verifier.fine_grained_reward import build_prompt, directed_reward
from llm_verifier.prompts import normalize_criteria

from ..utils import VerifierConfig, create_logger

_logger = create_logger("verifier")

_SCORE_TOKENS = frozenset("ABCDEFGHIJKLMNOPQRSTabcdefghijklmnopqrst")


def _has_score_tag_distribution(tokens, positions, tag: str) -> bool:
    text_so_far = ""
    for index, token in enumerate(tokens):
        text_so_far += str(token)
        if not text_so_far.rstrip().endswith(tag):
            continue
        if index + 1 >= len(positions):
            return False
        candidates = getattr(positions[index + 1], "candidates", None) or []
        for candidate in candidates:
            score_token = str(getattr(candidate, "token", "")).strip()
            logprob = getattr(candidate, "log_probability", None)
            if score_token in _SCORE_TOKENS:
                try:
                    if math.isfinite(float(logprob)):
                        return True
                except (TypeError, ValueError):
                    pass
        return False
    return False


class _StrictGeminiModels:
    def __init__(self, inner, owner):
        self._inner = inner
        self._owner = owner

    def generate_content(self, *args, **kwargs):
        response = self._inner.generate_content(*args, **kwargs)
        self._owner.record(response)
        return response


class StrictGeminiClient:
    """Request-local Gemini proxy that rejects llm-verifier text fallbacks."""

    def __init__(self, inner):
        self.models = _StrictGeminiModels(inner.models, self)
        self._lock = threading.Lock()
        self._validated_calls = 0

    @property
    def validated_calls(self) -> int:
        with self._lock:
            return self._validated_calls

    def record(self, response) -> None:
        candidates = getattr(response, "candidates", None) or []
        result = (
            getattr(candidates[0], "logprobs_result", None)
            if candidates else None
        )
        chosen = getattr(result, "chosen_candidates", None) if result else None
        positions = getattr(result, "top_candidates", None) if result else None
        tokens = [getattr(candidate, "token", "") for candidate in chosen or []]
        if not tokens or not positions or not all(
            _has_score_tag_distribution(tokens, positions, tag)
            for tag in ("<score_A>", "<score_B>")
        ):
            raise RuntimeError(
                "Gemini verifier response omitted required score-tag logprobs"
            )
        with self._lock:
            self._validated_calls += 1


class Comparison:
    """A single directed comparison (candidate i in slot A, j in slot B)."""

    def __init__(self, i: int, j: int, reward_a: float, reward_b: float,
                 prompt: str, text: str):
        self.i = i
        self.j = j
        self.reward_a = reward_a
        self.reward_b = reward_b
        self.text = text
        self.prompt = prompt

    def to_dict(self) -> dict:
        if self.reward_a > self.reward_b:
            winner = "A"
        elif self.reward_b > self.reward_a:
            winner = "B"
        else:
            winner = "tie"
        return {
            "i": self.i,
            "j": self.j,
            "rating_A": self.reward_a,
            "rating_B": self.reward_b,
            "winner": winner,
            "request": [{"role": "user", "content": self.prompt}],
            "text": self.text,
        }


class SelectionResult:
    def __init__(self, best_index: int, scores: List[float],
                 comparisons: List[Comparison], logprob_calls: int = 0):
        self.best_index = best_index
        self.scores = scores
        self.comparisons = comparisons
        self.logprob_calls = logprob_calls


class Verifier:
    def __init__(self, cfg: VerifierConfig):
        self.cfg = cfg
        self.method = cfg.method
        self.model_id = cfg.model.name.removeprefix("gemini/")
        self.require_logprobs = cfg.require_logprobs
        if self.require_logprobs:
            if not cfg.model.name.startswith("gemini/"):
                raise ValueError("require_logprobs currently supports Gemini only")
            if not cfg.model.api_key:
                raise ValueError("require_logprobs needs an explicit verifier api_key")
        self.criteria = normalize_criteria(
            [{"name": c.name, "description": c.description}
             for c in self.method.criteria]
        )
        self._client = None  # created lazily on first scoring call
        _logger.info(
            f"Verifier: model={cfg.model.name}, method=pivot_tournament, "
            f"pivots={self.method.pivots}, K={self.method.n_verifications}, "
            f"criteria={[c['name'] for c in self.criteria]}"
        )

    @property
    def client(self):
        """A google-genai client built from the config's api_key/provider;
        None lets llm-verifier create one from the environment."""
        if self._client is None and self.cfg.model.api_key:
            from google import genai
            if self.cfg.model.provider == "vertex_ai":
                self._client = genai.Client(vertexai=True,
                                            api_key=self.cfg.model.api_key)
            else:
                self._client = genai.Client(api_key=self.cfg.model.api_key)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def select_best(
        self, history: str, actions: List[str],
    ) -> SelectionResult:
        n = len(actions)
        if n == 0:
            return SelectionResult(0, [], [])
        if n == 1:
            return SelectionResult(0, [1.0], [])

        majority = self._try_majority_voting(actions)
        if majority is not None:
            return majority

        result, pair_scores, logprob_calls = await asyncio.to_thread(
            self._run_select, history, actions)
        comparisons = self._build_comparisons(history, actions, pair_scores)

        _logger.info(
            f"PPT: N={n} comparisons={result.n_comparisons} "
            f"scores=[{', '.join(f'{s:.3f}' for s in result.scores)}] "
            f"best={result.index}"
        )
        return SelectionResult(
            result.index, result.scores, comparisons, logprob_calls)

    # ------------------------------------------------------------------
    # llm-verifier tournament
    # ------------------------------------------------------------------

    def _run_select(self, history: str, actions: List[str]):
        """Run llm_verifier.select with a per-request score cache, returning
        (VerifierResult, raw pair scores) so the comparisons the tournament
        actually ran can be reconstructed for the request log."""
        m = self.method
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "scores.json")
            client = self.client
            strict_client = (
                StrictGeminiClient(client) if self.require_logprobs else client
            )
            result = llm_verifier.select(
                history,
                actions,
                criteria=self.criteria,
                ground_truth_note=m.note,
                n_evaluations=m.n_verifications,
                pivots=m.pivots,
                seed=m.seed,
                model=self.model_id,
                client=strict_client,
                cache=cache,
                progress=False,
                on_error="raise" if self.require_logprobs else "tie",
            )
            pair_scores = {}
            if os.path.exists(cache):
                with open(cache) as f:
                    pair_scores = json.load(f)
            logprob_calls = (
                strict_client.validated_calls if self.require_logprobs else 0
            )
            if self.require_logprobs and logprob_calls == 0:
                raise RuntimeError("Verifier produced no validated logprob calls")
        return result, pair_scores, logprob_calls

    def _build_comparisons(
        self, history: str, actions: List[str], pair_scores: dict,
    ) -> List[Comparison]:
        """One Comparison per directed pair in the score cache, with rewards
        averaged over criteria and repeats (matching the tournament's
        aggregation) and the slot-A/slot-B prompt of the first criterion."""
        m = self.method
        criteria_ids = [c["id"] for c in self.criteria]
        pairs = {}
        for key in pair_scores:
            _, task, pair, _ = key.split("|")
            a, b = (int(x) for x in pair.split(","))
            pairs[(a, b)] = task
        comparisons = []
        for (a, b), task in sorted(pairs.items()):
            ra, rb = directed_reward(
                pair_scores, task, a, b, criteria_ids, m.n_verifications)
            prompt = build_prompt(
                history, actions[a], actions[b], self.criteria[0], m.note)
            comparisons.append(Comparison(a, b, ra, rb, prompt, ""))
        return comparisons

    # ------------------------------------------------------------------
    # Majority voting
    # ------------------------------------------------------------------

    def _try_majority_voting(
        self, actions: List[str],
    ) -> Optional[SelectionResult]:
        if not self.cfg.majority_voting:
            return None
        counts = {}
        for action in actions:
            counts[action] = counts.get(action, 0) + 1
        majority_action, majority_count = "", 0
        for action, count in counts.items():
            if count > majority_count:
                majority_count, majority_action = count, action
        if majority_count <= len(actions) / 2:
            return None

        _logger.info(
            f"Majority voting: {majority_count}/{len(actions)} responses are "
            f"identical, skipping tournament"
        )
        best = next(i for i, a in enumerate(actions) if a == majority_action)
        scores = [1.0 if a == majority_action else 0.0 for a in actions]
        return SelectionResult(best, scores, [])
