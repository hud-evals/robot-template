"""vla_jepa_libero policy agent for libero over the robot transport.

VLA-JEPA (arXiv:2602.10098): Qwen3-VL-2B backbone + V-JEPA2 latent world model
+ DiT-B flow-matching action head; the LeRobot port of ginwind/VLA-JEPA. The
world model is training-only — at inference only the Qwen backbone and action
head run.

Stock LeRobot wiring (same surface as ``pi05_libero.py``): the checkpoint's two
image slots (``image``, ``image2``) take the env's agentview + wrist views
positionally, the 8-dim eef state feeds ``observation.state``, and the
checkpoint's own processors handle MEAN_STD state / MIN_MAX action
normalization plus the gripper snap+binarize (output gripper is exactly
{-1 open, +1 close} — LIBERO's native convention, so ``LeRobotAdapter``
passes actions straight through). ``select_action`` pops a 7-step chunk
(n_action_steps=7), matching the starVLA LIBERO eval client's chunk caching.

Requires a lerobot build that ships ``lerobot.policies.vla_jepa`` (git main as
of 2026-06; not in the 0.5.x PyPI releases):

    pip install "lerobot @ git+https://github.com/huggingface/lerobot.git"
"""

from __future__ import annotations

import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

from hud.agents.robot.adapter import LeRobotAdapter
from hud.agents.robot.agent import RobotAgent
from hud.agents.robot.model import LeRobotModel

DEFAULT_CHECKPOINT = "lerobot/VLA-JEPA-LIBERO"


class VLAJEPALiberoAgent(RobotAgent):
    max_steps = 400

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None) -> None:
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[agent] loading policy: {checkpoint} (device={self.device})", flush=True)
        policy = VLAJEPAPolicy.from_pretrained(checkpoint).to(self.device).eval()
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


__all__ = ["VLAJEPALiberoAgent"]
