"""Batched run: many LIBERO episodes off one stacked GPU forward.

Identical to `examples/local.py` but the agent is wrapped in `BatchedAgent`, so
`BATCH_SIZE` rollouts run at once and their per-step `infer` calls coalesce into one
forward. You write nothing extra — `BatchedAgent` clones the same shared agent per
rollout (isolated episode state) and shares one batched model.

Run:  python examples/batched.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on the path

from environment.tasks import make_tasks
from hud.agents.robot.batching import BatchedAgent
from hud.eval import DockerRuntime, Taskset
from inventory.agents.smolvla_libero import SmolVLALiberoAgent

BATCH_SIZE = 8  # concurrent rollouts coalesced per forward (== max_concurrent)


async def main() -> None:
    agent = BatchedAgent(SmolVLALiberoAgent(), batch_size=BATCH_SIZE)
    tasks = make_tasks(suite="libero_spatial", n=BATCH_SIZE)

    start = time.perf_counter()
    job = await Taskset("smolvla_libero_batched", tasks).run(
        agent, runtime=DockerRuntime("hud-libero-env"), max_concurrent=BATCH_SIZE
    )
    elapsed = time.perf_counter() - start

    rewards = [run.reward or 0.0 for run in job.runs]
    print(f"success_rate={sum(rewards) / len(rewards):.2f} ({sum(rewards):.0f}/{len(rewards)})")
    print(f"[timing] {len(tasks)} episodes (batch_size={BATCH_SIZE}) in {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
