"""SmolVLA on LIBERO — the smallest complete agent.

A stock LeRobot checkpoint wired with the batteries-included `LeRobotModel` +
`LeRobotAdapter`. There is no env-specific code here: the env→policy wiring
(which camera feeds which model slot, the state layout) is read from the env's
contract at connect time, so the same agent drives any LIBERO-shaped env.

This is the reference to copy for any other stock LeRobot checkpoint: swap the
policy class + checkpoint and you're done (see `pi05_libero.py`, `vla_jepa_libero.py`).
"""

from __future__ import annotations

import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from hud.agents.robot.adapter import LeRobotAdapter
from hud.agents.robot.agent import RobotAgent
from hud.agents.robot.model import LeRobotModel

DEFAULT_CHECKPOINT = "lerobot/smolvla_libero"


class SmolVLALiberoAgent(RobotAgent):
    max_steps = 400  # episode cap; LIBERO suites terminate well before this

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[agent] loading {checkpoint} (device={self.device})", flush=True)
        policy = SmolVLAPolicy.from_pretrained(checkpoint).to(self.device).eval()
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            checkpoint,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        self.model = LeRobotModel(policy, preprocess, postprocess)
        # The model's own image-slot names; the adapter maps the env's cameras onto them.
        self.adapter = LeRobotAdapter(model_image_keys=list(policy.config.image_features))


__all__ = ["SmolVLALiberoAgent"]
