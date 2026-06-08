"""Optional callbacks for future PPO experiments."""

from __future__ import annotations


def require_sb3_callback_base():
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for callback classes.") from exc
    return BaseCallback


def make_episode_info_callback():
    """Create a callback class lazily so this file imports without SB3."""

    BaseCallback = require_sb3_callback_base()

    class EpisodeInfoCallback(BaseCallback):
        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                if "completed_targets" in info:
                    self.logger.record("env/completed_targets", info["completed_targets"])
                if "remaining_battery" in info:
                    self.logger.record("env/remaining_battery", info["remaining_battery"])
            return True

    return EpisodeInfoCallback
