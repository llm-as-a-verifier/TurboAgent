"""
Progress monitor: a post-hoc, observability-only score for the selected
trajectory, computed with the EXACT fine-grained reward of `pre-release`.

It runs *after* the API response has been selected and returned — it never
changes which response the client receives. The score is the pre-release
fine-grained reward applied pointwise to the trajectory:

    R(t, tau) = (1 / C K) * sum_c sum_k  extract_score( <score> | t, c, tau )

  C = number of evaluation criteria
  K = number of repeated verifications (n_verifications)

Each (criterion, repetition) is one verifier call on the granularity-20 A-T
scale; `extract_score` reads the token logprobs and takes the expectation over
the ordered score tokens, normalized to [0, 1]. This is the same SCALE,
`extract_score`, and C*K averaging the pivot-tournament verifier uses — just on
a single trajectory instead of a directed pair. The result is recorded in the
request log and surfaced as a node in the visualizer.
"""

import asyncio
from typing import List

from ..utils import ProgressMonitorConfig, create_logger
from ..verifier.fine_grained_reward import (
    build_pointwise_prompt,
    call_gemini,
    create_gemini_client,
    extract_score,
)

_logger = create_logger("progress_monitor")


class ProgressMonitorResult:
    def __init__(self, score: float, criterion_scores: List[dict],
                 generated_text: str):
        self.score = score
        self.criterion_scores = criterion_scores
        self.generated_text = generated_text

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "criterionScores": self.criterion_scores,
            "generatedText": self.generated_text,
        }


class ProgressMonitor:
    def __init__(self, cfg: ProgressMonitorConfig):
        self.cfg = cfg
        self.model_id = cfg.model.name.removeprefix("gemini/")
        self._client = None
        _logger.info(
            f"ProgressMonitor: model={cfg.model.name}, "
            f"criteria={[c.name for c in cfg.criteria]}, "
            f"K={cfg.n_verifications}"
        )

    @property
    def client(self):
        if self._client is None:
            self._client = create_gemini_client(
                api_key=self.cfg.model.api_key,
                provider=self.cfg.model.provider,
            )
        return self._client

    async def evaluate(self, trajectory: str) -> ProgressMonitorResult:
        """Fine-grained reward of the trajectory, averaged over every criterion
        and repeated verification — exactly the pre-release R(t, tau)."""
        cfg = self.cfg
        jobs = [(crit, rep)
                for crit in cfg.criteria
                for rep in range(cfg.n_verifications)]

        async def one(crit) -> tuple:
            prompt = build_pointwise_prompt(
                trajectory, crit.name, crit.description, cfg.note)
            text, tokens, position_logprobs = await call_gemini(
                self.client, self.model_id, prompt,
                top_logprobs=cfg.model.max_top_logprobs,
                temperature=cfg.temperature, max_tokens=cfg.max_tokens)
            score = extract_score(text, tokens, position_logprobs, "<score>")
            return score, text

        results = await asyncio.gather(*[one(crit) for crit, _ in jobs])
        scores = [r[0] for r in results]
        reward = sum(scores) / len(scores) if scores else 0.5

        criterion_scores = [
            {"criterion": crit.name, "rep": rep, "score": score}
            for (crit, rep), (score, _) in zip(jobs, results)
        ]
        _logger.info(
            f"Progress reward: {reward:.3f} "
            f"(C={len(cfg.criteria)} K={cfg.n_verifications}, "
            f"{len(jobs)} verifier calls)"
        )
        return ProgressMonitorResult(reward, criterion_scores, results[0][1])
