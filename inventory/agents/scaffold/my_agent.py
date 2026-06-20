"""SCAFFOLD — define your own policy agent. Copy this file, fill the TODOs, delete the rest.

An agent is three seams; you usually only touch one or two:

    RobotAgent   the episode loop (connect, observe, act, stop) — the SDK owns it
      ├─ Model    *how to run* your policy: preprocess -> forward -> postprocess
      └─ Adapter  translate env<->policy spaces, learned from the env contract

Pick the path that matches your policy. Docs: https://docs.hud.ai/v6/core/robots
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from hud.agents.robot.adapter import Adapter, LeRobotAdapter
from hud.agents.robot.agent import RobotAgent
from hud.agents.robot.model import LeRobotModel, Model

# ── Path A: a stock LeRobot checkpoint — no custom code ───────────────────────
# If your policy is a LeRobot policy, this is the whole agent (see smolvla_libero.py).

DEFAULT_CHECKPOINT = "TODO/your-lerobot-checkpoint"


class MyAgent(RobotAgent):
    max_steps = 400  # per-episode cap

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None) -> None:
        from lerobot.policies.factory import make_pre_post_processors

        # from lerobot.policies.<family>.modeling_<family> import YourPolicy  # TODO
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        policy = ...  # TODO: YourPolicy.from_pretrained(checkpoint).to(device).eval()
        preprocess, postprocess = make_pre_post_processors(
            policy.config, checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.model = LeRobotModel(policy, preprocess, postprocess)
        self.adapter = LeRobotAdapter(model_image_keys=list(policy.config.image_features))


# ── Path B: a non-LeRobot policy — subclass Model, implement infer() ──────────
# `infer` is batch-shaped and stateless: one batch dict in, an [N, T, A] chunk out
# (keep the leading N even for N=1). Per-episode state lives on the agent, not here.


class MyModel(Model):
    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def infer(self, batch: Any) -> np.ndarray:
        chunk = self.policy(batch)                       # TODO: run your policy
        return np.asarray(chunk, dtype=np.float32)       # [N, T, A] in the env's action space


# ── Path C: custom env<->policy wiring — subclass Adapter ─────────────────────
# Override `bind` is rarely needed (the base splits images/state from the contract).
# `adapt_observation` is the one you usually write; `adapt_action` defaults to identity.


class MyAdapter(Adapter):
    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> Any:
        data = obs["data"]                               # env obs, keyed by contract feature name
        # self.image_keys / self.state_key were filled from the contract in bind()
        return {                                          # TODO: shape this for your policy
            "image": data[self.image_keys[0]],
            "state": data[self.state_key],
            "task": prompt,
        }

    def adapt_action(self, action: np.ndarray, obs: dict[str, Any]) -> np.ndarray:
        return action                                    # TODO: map policy action -> env action


__all__ = ["MyAgent"]
