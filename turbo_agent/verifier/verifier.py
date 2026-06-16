"""
Verifier: best-of-N selection over candidate agent responses using the Pivot
Preference Tournament (PPT) with a fine-grained logprob reward.

Given the conversation history and N candidate responses, the verifier scores
directed pairs (candidate `a` in slot A, `b` in slot B) with Gemini logprobs
over a 20-token A-T scale, turns each pair's two rewards into a soft
Bradley-Terry win, and aggregates them through PPT to pick the best candidate
in O(N·k) comparisons rather than the O(N^2) of full round-robin.
"""

import asyncio
import random
from typing import Dict, List, Optional, Tuple

from ..utils import VerifierConfig, create_logger
from . import pivot_tournament as ppt
from .fine_grained_reward import (
    build_prompt,
    call_gemini,
    create_gemini_client,
    extract_score,
)

_logger = create_logger("verifier")

Pair = Tuple[int, int]


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
                 comparisons: List[Comparison], pivots: List[int]):
        self.best_index = best_index
        self.scores = scores
        self.comparisons = comparisons
        self.pivots = pivots


class Verifier:
    def __init__(self, cfg: VerifierConfig):
        self.cfg = cfg
        self.method = cfg.method
        self.model_id = cfg.model.name.removeprefix("gemini/")
        self._client = None  # created lazily on first scoring call
        _logger.info(
            f"Verifier: model={cfg.model.name}, method=pivot_tournament, "
            f"pivots={self.method.pivots}, K={self.method.n_verifications}, "
            f"criteria={[c.name for c in self.method.criteria]}"
        )

    @property
    def client(self):
        if self._client is None:
            self._client = create_gemini_client(
                api_key=self.cfg.model.api_key,
                provider=self.cfg.model.provider,
            )
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def select_best(
        self, history: str, actions: List[str],
    ) -> SelectionResult:
        n = len(actions)
        if n == 0:
            return SelectionResult(0, [], [], [])
        if n == 1:
            return SelectionResult(0, [1.0], [], [])

        majority = self._try_majority_voting(actions)
        if majority is not None:
            return majority

        k = self.method.pivots
        rng = random.Random(self.method.seed)

        # ---- step 1: ring pass ----
        ring = ppt.ring_cycle(n, rng)
        rewards: Dict[Pair, Tuple[float, float]] = {}
        comps: Dict[Pair, Comparison] = {}
        await self._score_pairs(history, actions, ring, rewards, comps)

        # ---- step 2: pivots = ring-pass empirical leaders ----
        w, c = [0.0] * n, [0] * n
        ppt.accumulate(ring, rewards, w, c)
        pivots = ppt.select_pivots(w, c, k)

        # ---- step 3: pivot rounds ----
        pr_pairs = ppt.pivot_round_pairs(n, pivots)
        await self._score_pairs(history, actions, pr_pairs, rewards, comps)

        # ---- step 4: aggregate ring + pivot rounds, select winner ----
        w, c = [0.0] * n, [0] * n
        ppt.accumulate(ring, rewards, w, c)
        ppt.accumulate(pr_pairs, rewards, w, c)
        best = ppt.best_index(w, c)
        scores = ppt.mean_preferences(w, c)

        ordered_pairs: List[Pair] = []
        seen = set()
        for pair in [*ring, *pr_pairs]:
            if pair not in seen:
                seen.add(pair)
                ordered_pairs.append(pair)
        comparisons = [comps[p] for p in ordered_pairs if p in comps]

        _logger.info(
            f"PPT: N={n} pivots={pivots} comparisons={len(ordered_pairs)} "
            f"scores=[{', '.join(f'{s:.3f}' for s in scores)}] best={best}"
        )
        return SelectionResult(best, scores, comparisons, pivots)

    # ------------------------------------------------------------------
    # Majority voting
    # ------------------------------------------------------------------

    def _try_majority_voting(
        self, actions: List[str],
    ) -> Optional[SelectionResult]:
        if not self.cfg.majority_voting:
            return None
        counts: Dict[str, int] = {}
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
        return SelectionResult(best, scores, [], [])

    # ------------------------------------------------------------------
    # Directed-pair scoring (fine-grained logprob reward)
    # ------------------------------------------------------------------

    async def _score_pairs(
        self, history: str, actions: List[str], pairs: List[Pair],
        rewards: Dict[Pair, Tuple[float, float]],
        comps: Dict[Pair, Comparison],
    ) -> None:
        """Score every directed pair not already scored, in parallel, and
        merge the fine-grained rewards + a representative comparison record."""
        todo = [p for p in pairs if p not in rewards]
        if not todo:
            return
        results = await asyncio.gather(
            *[self._score_pair(history, actions[a], actions[b])
              for (a, b) in todo]
        )
        for (a, b), (ra, rb, prompt, text) in zip(todo, results):
            rewards[(a, b)] = (ra, rb)
            comps[(a, b)] = Comparison(a, b, ra, rb, prompt, text)

    async def _score_pair(
        self, history: str, action_a: str, action_b: str,
    ) -> Tuple[float, float, str, str]:
        """Fine-grained rewards (R_a, R_b) for one directed pair, averaged over
        all criteria and K repeated verifications."""
        m = self.method
        jobs = [(crit, rep)
                for crit in m.criteria
                for rep in range(m.n_verifications)]

        async def one(crit) -> Tuple[float, float, str, str]:
            prompt = build_prompt(
                history, action_a, action_b, crit.name, crit.description,
                m.note)
            text, tokens, position_logprobs = await call_gemini(
                self.client, self.model_id, prompt,
                top_logprobs=self.cfg.model.max_top_logprobs,
                temperature=m.temperature, max_tokens=m.max_tokens)
            ra = extract_score(text, tokens, position_logprobs, "<score_A>")
            rb = extract_score(text, tokens, position_logprobs, "<score_B>")
            return ra, rb, prompt, text

        results = await asyncio.gather(*[one(crit) for crit, _ in jobs])
        sum_a = sum(r[0] for r in results)
        sum_b = sum(r[1] for r in results)
        n = len(results)
        prompt0, text0 = results[0][2], results[0][3]
        return sum_a / n, sum_b / n, prompt0, text0
