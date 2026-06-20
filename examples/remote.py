"""Fully remote run: the policy is served on a GPU, the env runs on Modal.

Your machine is *just the harness* — no GPU, no sim. Two things live elsewhere:
  - the policy: a served openpi/0 server (`RemoteModel` ships each observation and
    gets back an action chunk).
  - the env: each episode runs in a Modal sandbox (`ModalRuntime`).

So this process only runs the episode loop — it works on a laptop.

Prereqs:
  1. Serve the policy:  modal run inventory/agents/remote/smolvla/smolvla_serve.py
     (prints ws://HOST:PORT)  ->  export POLICY_HOST=...  POLICY_PORT=...
  2. Publish the env:   modal run inventory/envs/remote/modal/deploy.py   (builds hud-libero-env)
  3. modal token new    (one-time)

Run:  python examples/remote.py

(Point POLICY_HOST/PORT at any openpi/0 server instead — stock openpi, Cosmos, a
hosted pi0.5 — the harness doesn't change.)
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on the path

from environment.tasks import make_tasks
from hud.eval import ModalRuntime, Taskset
from inventory.agents.remote.smolvla.smolvla_libero_remote import SmolVLARemoteAgent

IMAGE_NAME = "hud-libero-env"  # published by inventory/envs/remote/modal/deploy.py


async def main() -> None:
    host, port = os.environ["POLICY_HOST"], int(os.environ["POLICY_PORT"])
    agent = SmolVLARemoteAgent(host, port)
    tasks = make_tasks(suite="libero_spatial", n=3)
    # RemoteModel is one request per env (not batchable), so one rollout at a time.
    job = await Taskset("smolvla_libero_remote", tasks).run(
        agent,
        runtime=ModalRuntime(IMAGE_NAME, runtime_config={"resources": {"memory_mb": 8192}}),
        max_concurrent=1,
    )
    rewards = [run.reward or 0.0 for run in job.runs]
    for task, reward in zip(tasks, rewards, strict=False):
        print(f"{task.id} {task.args} -> reward={reward}")
    print(f"success_rate={sum(rewards) / len(rewards):.2f} ({sum(rewards):.0f}/{len(rewards)})")


if __name__ == "__main__":
    asyncio.run(main())
