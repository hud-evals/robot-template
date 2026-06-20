"""Drive LIBERO with the env on Daytona — one sandbox per rollout.

Each rollout's LIBERO env runs in its own Daytona sandbox (booted from the
`hud-libero-env` snapshot; see `deploy.py`); the SmolVLA policy runs in this
process on your GPU and drives all of them at once via `BatchedAgent` (one stacked
forward serves `BATCH_SIZE` sandboxes).

    python inventory/envs/remote/daytona/deploy.py      # once: build + register the snapshot
    python inventory/envs/remote/daytona/run.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root on the path

from environment.tasks import make_tasks
from hud.agents.robot.batching import BatchedAgent
from hud.eval import DaytonaRuntime, Taskset
from inventory.agents.smolvla_libero import SmolVLALiberoAgent

SNAPSHOT_NAME = "hud-libero-env"  # registered by deploy.py
BATCH_SIZE = 8

TASKS = make_tasks(suite="libero_spatial", n=BATCH_SIZE)


async def main() -> None:
    agent = BatchedAgent(SmolVLALiberoAgent(), batch_size=BATCH_SIZE)
    job = await Taskset("daytona_libero_batched", TASKS).run(
        agent,
        # Dockerfile.hud sets WORKDIR /app and serves environment/env:env from there.
        runtime=DaytonaRuntime(SNAPSHOT_NAME, workdir="/app"),
        max_concurrent=BATCH_SIZE,
    )
    rewards = [run.reward for run in job.runs]
    print(f"success_rate={sum(rewards) / len(rewards):.2f} ({sum(rewards):.0f}/{len(rewards)})")


if __name__ == "__main__":
    asyncio.run(main())
