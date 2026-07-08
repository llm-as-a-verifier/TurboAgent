import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    name: str
    provider: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class CriterionConfig:
    name: str
    description: str = ""


@dataclass
class PivotTournamentConfig:
    """Parameters for the Probabilistic Pivot Tournament selection method."""
    pivots: int = 2            # k: number of pivot (empirical-leader) candidates
    n_verifications: int = 4   # K: repeated verifications per directed pair
    seed: int = 0             # seed for the random ring pass (reproducible)
    note: str = ""             # ground-truth note injected into the prompt
    criteria: List[CriterionConfig] = field(default_factory=list)


@dataclass
class VerifierConfig:
    model: ModelConfig
    method: PivotTournamentConfig
    majority_voting: bool = False


@dataclass
class ContextConfig:
    model_name: str
    api_key: str
    refinement_prompt: str


@dataclass
class ProgressMonitorConfig:
    """A post-hoc progress score for the selected trajectory, computed with
    `llm_verifier.track` (K repeated verifications, averaged). Observability
    only — it never changes the response."""
    model: ModelConfig
    n_verifications: int = 4   # K: repeated verifications


# Default holistic criterion used when the config declares none.
_DEFAULT_CRITERIA = [
    CriterionConfig(
        name="Task Success",
        description=(
            "How likely the agent correctly and completely solved the task. "
            "The strongest signal is the agent verifying its solution against "
            "the task's specific requirements. Trajectory length, number of "
            "steps, and apparent confidence do not predict correctness."
        ),
    ),
]


class Config:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(Path.cwd() / "turbo-agent.yaml")
            if not Path(config_path).exists():
                raise FileNotFoundError(
                    f"No turbo-agent.yaml found in {Path.cwd()}. "
                    "Run turbo-agent from a directory containing a "
                    "turbo-agent.yaml config file."
                )

        # Load .env from the same directory as the config file.
        env_path = Path(config_path).parent / ".env"
        if env_path.exists():
            load_dotenv(str(env_path), override=True)

        with open(config_path, "r") as f:
            self._raw: Dict[str, Any] = yaml.safe_load(f) or {}

        self._expand_env_vars()

    @staticmethod
    def _resolve_env(value: str) -> str:
        if isinstance(value, str) and value.startswith("$"):
            return os.environ.get(value[1:], "")
        return value

    def _expand_env_vars(self) -> None:
        for model in self.models:
            key = model.get("api_key", "")
            if isinstance(key, str) and key.startswith("$"):
                model["api_key"] = self._resolve_env(key)

    # ------------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------------

    @property
    def models(self) -> List[Dict[str, Any]]:
        return self._raw.get("backend", {}).get("models", [])

    @property
    def default_model(self) -> Dict[str, Any]:
        if not self.models:
            raise ValueError("No models configured under backend.models")
        return self.models[0]

    @property
    def total_candidates(self) -> int:
        return sum(m.get("num_candidates", 1) for m in self.models)

    # ------------------------------------------------------------------
    # Context refinement (optional)
    # ------------------------------------------------------------------

    @property
    def context_config(self) -> Optional[ContextConfig]:
        raw_ctx = self._raw.get("context")
        if not raw_ctx:
            return None
        raw_model = raw_ctx.get("refinement_model")
        prompt = raw_ctx.get("refinement_prompt")
        if not raw_model or not raw_model.get("name") or not prompt:
            return None
        return ContextConfig(
            model_name=raw_model["name"],
            api_key=self._resolve_env(raw_model.get("api_key", "")),
            refinement_prompt=prompt,
        )

    # ------------------------------------------------------------------
    # Verifier
    # ------------------------------------------------------------------

    @property
    def verifier_config(self) -> Optional[VerifierConfig]:
        raw_v = self._raw.get("verifier")
        if not raw_v:
            return None

        raw_model = raw_v.get("model")
        if not raw_model or not raw_model.get("name"):
            return None
        raw_api_key = raw_model.get("api_key", "")
        model_cfg = ModelConfig(
            name=raw_model["name"],
            provider=raw_model.get("provider"),
            api_key=self._resolve_env(raw_api_key) if raw_api_key else None,
        )

        raw_method = raw_v.get("method", {})
        method_name = raw_method.get("name", "pivot_tournament")
        if method_name != "pivot_tournament":
            raise ValueError(
                f"Unknown verifier method '{method_name}'. "
                f"Only 'pivot_tournament' is supported."
            )

        criteria = [
            CriterionConfig(
                name=c.get("name", ""),
                description=c.get("description", ""),
            )
            for c in raw_method.get("criteria", [])
        ] or list(_DEFAULT_CRITERIA)

        method_cfg = PivotTournamentConfig(
            pivots=raw_method.get("pivots", 2),
            n_verifications=raw_method.get("n_verifications", 4),
            seed=raw_method.get("seed", 0),
            note=raw_method.get("note", ""),
            criteria=criteria,
        )

        return VerifierConfig(
            model=model_cfg,
            method=method_cfg,
            majority_voting=raw_v.get("majority_voting", False),
        )

    # ------------------------------------------------------------------
    # Progress monitor (optional, post-hoc observability)
    # ------------------------------------------------------------------

    @property
    def progress_monitor_config(self) -> Optional[ProgressMonitorConfig]:
        raw_pm = self._raw.get("progress_monitor")
        if not raw_pm:
            return None
        raw_model = raw_pm.get("model")
        if not raw_model or not raw_model.get("name"):
            return None
        raw_api_key = raw_model.get("api_key", "")
        model_cfg = ModelConfig(
            name=raw_model["name"],
            provider=raw_model.get("provider"),
            api_key=self._resolve_env(raw_api_key) if raw_api_key else None,
        )
        return ProgressMonitorConfig(
            model=model_cfg,
            n_verifications=raw_pm.get("n_verifications", 4),
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @property
    def log_dir(self) -> str:
        dir_name = self._raw.get("log_dir", "default")
        return str(Path(".turbo-agent") / dir_name)

    @property
    def raw_config(self) -> Dict[str, Any]:
        return self._raw
