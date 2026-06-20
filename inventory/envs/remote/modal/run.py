"""Drive LIBERO with the env on Modal — one sandbox per rollout.

Each rollout's LIBERO env runs in its own Modal sandbox (booted from the
`hud-libero-env` image; see `deploy.py`); the SmolVLA policy runs in this process
on your GPU and drives all of them at once. `BatchedAgent` clones the agent per
rollout (isolated episode state) and coalesces the concurrent per-step `infer`
calls into one stacked forward, so `BATCH_SIZE` sandboxes share a single GPU pass.

    modal run inventory/envs/remote/modal/deploy.py     # once: build + publish the image
    python inventory/envs/remote/modal/run.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root on the path

from environment.tasks import make_tasks
from hud.agents.robot.batching import BatchedAgent
from hud.eval import ModalRuntime, Taskset
from inventory.agents.smolvla_libero import SmolVLALiberoAgent

IMAGE_NAME = "hud-libero-env"  # published by deploy.py
BATCH_SIZE = 8                 # concurrent sandboxes coalesced per forward (== max_concurrent)

TASKS = make_tasks(suite="libero_spatial", n=BATCH_SIZE)


async def main() -> None:
    agent = BatchedAgent(SmolVLALiberoAgent(), batch_size=BATCH_SIZE)
    job = await Taskset("modal_libero_batched", TASKS).run(
        agent,
        # LIBERO's CPU sim wants the RAM headroom; the image serves env.py at its WORKDIR.
        runtime=ModalRuntime(IMAGE_NAME, runtime_config={"resources": {"memory_mb": 8192}}),
        max_concurrent=BATCH_SIZE,
    )
    rewards = [run.reward for run in job.runs]
    print(f"success_rate={sum(rewards) / len(rewards):.2f} ({sum(rewards):.0f}/{len(rewards)})")


if __name__ == "__main__":
    asyncio.run(main())
