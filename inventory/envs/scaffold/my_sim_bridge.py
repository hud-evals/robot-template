"""SCAFFOLD bridge — wrap your simulator. Copy, fill the TODOs, delete the rest.

You implement exactly three methods; the framework owns the WebSocket serve loop,
the openpi/0 wire codec, the single-agent connection, and stepping at the control
rate. Reference: `environment/libero_sim_bridge.py`.
Docs: https://docs.hud.ai/v6/core/robots#environment-side
"""

from __future__ import annotations

import numpy as np

from hud.environment.robot import RobotBridge


class MySimBridge(RobotBridge):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sim = None      # TODO: your simulator handle
        self._obs = None      # latest raw sim observation

    async def reset(self, task_id: str, seed: int = 0) -> str:
        """Build the episode for `task_id`; return the task prompt (the agent's instruction).

        The base wrapper already zeroes success/total_reward/terminated and pushes
        the first frame after this returns — just build the scene and return the prompt.
        """
        # self._sim = make_sim(task_id, seed); self._obs = self._sim.reset()  # TODO
        self.task_description = "TODO: the natural-language task for this episode"
        return self.task_description

    def step(self, action: np.ndarray) -> None:
        """Apply one action and advance one tick (SYNCHRONOUS). Set scoring fields.

        `action` is already a decoded numpy array in your env's action space.
        success = goal achieved (a horizon timeout must NOT count as success).
        """
        # self._obs, reward, done = self._sim.step(action)  # TODO
        reward, done, goal_reached = 0.0, False, False      # TODO
        self.total_reward += reward
        self.success = bool(goal_reached)
        self.terminated = bool(done) or self.success

    def get_observation(self) -> tuple[dict[str, np.ndarray], bool] | None:
        """Return (data, terminated), or None if not ready.

        `data` keys must match the contract's observation feature names EXACTLY.
        Images: raw uint8 HWC. State: a single 1-D float array in contract order.
        """
        if self._obs is None:
            return None
        data = {
            "observation/image": np.zeros((128, 128, 3), dtype=np.uint8),   # TODO: real RGB frame
            "observation/state": np.zeros(4, dtype=np.float32),             # TODO: real proprio
        }
        return data, self.terminated


__all__ = ["MySimBridge"]
