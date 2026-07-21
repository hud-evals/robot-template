"""SmolVLA x LIBERO: run the benchmark and report a success rate.

    MUJOCO_GL=egl python run.py

With HUD_API_KEY set, rollouts stream to the trace viewer on hud.ai.
See README for the Docker path (sim in a container, no local LIBERO).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from hud.agents.robot import LeRobotAdapter, LeRobotModel, RobotAgent
from hud.eval import LocalRuntime, Task, Taskset

CHECKPOINT = "lerobot/smolvla_libero"
SUITE = "libero_spatial"  # LIBERO task suite
N_TASKS = 3  # task ids 0..N_TASKS-1
ENV_MODULE = Path(__file__).parent / "environment" / "env.py"


class SmolVLAAgent(RobotAgent):
    """Stock LeRobot checkpoint: model + adapter, harness owns the rest."""

    def __init__(self, checkpoint: str = CHECKPOINT) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.model = LeRobotModel(policy, preprocess, postprocess)
        self.adapter = LeRobotAdapter(model_image_keys=list(policy.config.image_features))


async def main() -> None:
    tasks = [
        Task(env="libero", id="episode", args={"task_suite": SUITE, "task_id": t})
        for t in range(N_TASKS)
    ]
    # LocalRuntime runs the sim in-process; swap in DockerRuntime for the container path (README).
    job = await Taskset("smolvla-libero", tasks).run(
        SmolVLAAgent(),
        runtime=LocalRuntime(ENV_MODULE, ready_timeout=600.0),
    )

    rewards = [run.reward or 0.0 for run in job.runs]
    for task, reward in zip(tasks, rewards, strict=True):
        print(f"  {task.args} -> reward={reward}")
    n = len(rewards) or 1
    print(f"\nsuccess_rate={sum(rewards) / n:.2f} ({sum(rewards):.0f}/{n})")
    if getattr(job, "id", None):
        print(f"job=https://hud.ai/jobs/{job.id}")


if __name__ == "__main__":
    asyncio.run(main())
