"""Uniform random baseline policy."""

from __future__ import annotations

import random
from typing import List

from envs.patrol_env import ACTION_NAMES, PatrolEnv


class RandomPolicy:
    """Uniform random action baseline."""

    policy_name = "random"

    def select_action(self, env: PatrolEnv) -> int:
        return int(env.action_space.sample())

    def predict(self, observation, deterministic: bool = True):
        return random.randrange(len(ACTION_NAMES)), None

    def action_probabilities(self, action=None) -> List[float]:
        return [1.0 / len(ACTION_NAMES) for _ in ACTION_NAMES]
