"""pi0.5 on LIBERO — same surface as `smolvla_libero.py`, bigger policy.

A vision-language-action policy conditioned on the scenario prompt and the env's
observations. The env→policy wiring comes from the contract, so the agent carries
no env-specific key names — only the policy class (`PI05Policy`) and checkpoint
differ from the SmolVLA agent.

pi0.5 specifics handled transparently by the shared surface:
  - the checkpoint exposes three model image slots (``observation.images.image``,
    ``observation.images.image2`` and an ``empty_camera_0`` pad). The env's two
    views map positionally onto the first two; the unfilled empty-camera slot is
    auto zero-padded by the policy (masked out), so no extra wiring is needed.
  - the policy embeds the (normalized, discretized) state into the language prompt
    as state tokens; ``make_pre_post_processors`` builds that step from the
    checkpoint config, so the agent still only feeds ``observation.state``.
"""

from __future__ import annotations

import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

from hud.agents.robot.adapter import LeRobotAdapter
from hud.agents.robot.agent import RobotAgent
from hud.agents.robot.model import LeRobotModel

DEFAULT_CHECKPOINT = "lerobot/pi05_libero_finetuned"


class PI05LiberoAgent(RobotAgent):
    max_steps = 400

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None) -> None:
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[agent] loading policy: {checkpoint} (device={self.device})", flush=True)
        policy = PI05Policy.from_pretrained(checkpoint).to(self.device).eval()
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            checkpoint,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        self.model = LeRobotModel(policy, preprocess, postprocess)
        self.adapter = LeRobotAdapter(model_image_keys=list(policy.config.image_features))
        print(
            f"[agent] policy ready on {self.device} | "
            f"image_features={self.adapter.model_image_keys}",
            flush=True,
        )


__all__ = ["PI05LiberoAgent"]
