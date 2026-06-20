"""Local run: SmolVLA in this process, each LIBERO episode in its own Docker container.

The smallest end-to-end robot run — load a stock checkpoint, point it at the
containerized env, get a success rate. Same as `python run.py`, spelled out.

Prereqs (see QUICKSTART.md): the harness (`hud-python[robot]` + `lerobot[smolvla]`)
and the env image built once from Dockerfile.hud (`docker build -t hud-libero-env .`).

Run:  python examples/local.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on the path

from environment.tasks import make_tasks
from hud.eval import DockerRuntime, Taskset
from inventory.agents.smolvla_libero import SmolVLALiberoAgent


async def main() -> None:
    agent = SmolVLALiberoAgent()  # the one shared agent every run uses
    tasks = make_tasks(suite="libero_spatial", n=3)
    job = await Taskset("smolvla_libero", tasks).run(
        agent, runtime=DockerRuntime("hud-libero-env"), max_concurrent=1
    )
    rewards = [run.reward or 0.0 for run in job.runs]
    for task, reward in zip(tasks, rewards, strict=False):
        print(f"{task.id} {task.args} -> reward={reward}")
    print(f"success_rate={sum(rewards) / len(rewards):.2f} ({sum(rewards):.0f}/{len(rewards)})")


if __name__ == "__main__":
    asyncio.run(main())
