"""
Progress monitor: a post-hoc, observability-only score for the selected
trajectory, computed with `llm_verifier.track`.

It runs *after* the API response has been selected and returned — it never
changes which response the client receives. The conversation history is the
task and the selected response is the trajectory step to score; `track` asks
the verifier whether the agent's current state would satisfy the task's
hidden grader, decodes the answer-letter logprob distribution into a
continuous score in [0, 1], and averages it over `n_verifications` repeats.
The result is recorded in the request log and surfaced as a node in the
visualizer.
"""

import asyncio
from typing import List, Optional

import llm_verifier

from ..utils import ProgressMonitorConfig, create_logger

_logger = create_logger("progress_monitor")


class ProgressMonitorResult:
    def __init__(self, score: float, rep_scores: List[Optional[float]]):
        self.score = score
        self.rep_scores = rep_scores

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "criterionScores": [
                {"criterion": "Task Progress", "rep": rep, "score": score}
                for rep, score in enumerate(self.rep_scores)
                if score is not None
            ],
        }


class ProgressMonitor:
    def __init__(self, cfg: ProgressMonitorConfig):
        self.cfg = cfg
        self.model_id = cfg.model.name.removeprefix("gemini/")
        self._client = None
        _logger.info(
            f"ProgressMonitor: model={cfg.model.name}, "
            f"K={cfg.n_verifications}"
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

    async def evaluate(self, problem: str, response: str) -> ProgressMonitorResult:
        """Progress score of the selected response on the task, averaged over
        `n_verifications` repeated verifications."""
        cfg = self.cfg
        result = await asyncio.to_thread(
            llm_verifier.track,
            problem,
            [response],
            n_evaluations=cfg.n_verifications,
            model=self.model_id,
            client=self.client,
        )
        rep_scores = [rep[0] for rep in result.per_rep_scores]
        _logger.info(
            f"Progress reward: {result.final:.3f} "
            f"(K={cfg.n_verifications} verifier calls)"
        )
        return ProgressMonitorResult(result.final, rep_scores)
