"""PPO model loading and action-probability helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from envs.patrol_env import ACTION_NAMES


class PPOExecutor:
    """Thin wrapper around a Stable-Baselines3 PPO checkpoint."""

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self.model = self._load_model(self.model_path)

    def predict(self, observation, deterministic: bool = True) -> int:
        action, _state = self.model.predict(observation, deterministic=deterministic)
        return int(np.asarray(action).item())

    def action_probabilities(self, observation) -> Optional[List[float]]:
        return stable_baselines_action_probabilities(self.model, observation)

    @staticmethod
    def _load_model(model_path: Path):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError(
                "stable-baselines3 is required to load PPO checkpoints. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc
        if not model_path.exists():
            raise FileNotFoundError(f"PPO checkpoint not found: {model_path}")
        return PPO.load(str(model_path))


def stable_baselines_action_probabilities(model, observation) -> Optional[List[float]]:
    """Return action probabilities for SB3 categorical policies when possible."""

    try:
        import torch

        obs_tensor, _ = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.detach().cpu().numpy()[0]
        return [float(x) for x in probs]
    except Exception:
        return None


def empty_action_probabilities() -> List[float]:
    return [0.0 for _ in ACTION_NAMES]
