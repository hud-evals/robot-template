"""SmolVLA x LIBERO: the whole benchmark from one Taskset.run.

    docker build -f Dockerfile.hud -t hud-libero-env .
    python run.py                     # 3 tasks of libero_spatial
    python run.py --suite libero_goal -n 10

The sim stack lives in the image: each rollout boots a fresh container
(DockerRuntime), and the task travels as template args over the wire. Only the
policy runs here, so the local install needs lerobot for the checkpoint but no
MuJoCo/LIBERO. With HUD_API_KEY set, rollouts stream to the trace viewer on hud.ai.
"""

from __future__ import annotations

import argparse
import asyncio

from hud.agents.robot import LeRobotAgent
from hud.eval import DockerRuntime, Task, Taskset

CHECKPOINT = "lerobot/smolvla_libero"
IMAGE = "hud-libero-env"  # built from Dockerfile.hud (see README)


async def main(args: argparse.Namespace) -> None:
    tasks = [
        Task(env="libero", id="episode", args={"task_suite": args.suite, "task_id": t})
        for t in range(args.n)
    ]
    job = await Taskset("smolvla-libero", tasks).run(
        LeRobotAgent(CHECKPOINT),
        runtime=DockerRuntime(IMAGE),
        max_concurrent=args.max_concurrent,
    )

    rewards = [run.reward or 0.0 for run in job.runs]
    for task, reward in zip(tasks, rewards):
        print(f"  {task.args} -> reward={reward}")
    n = len(rewards) or 1
    print(f"\nsuccess_rate={sum(rewards) / n:.2f} ({sum(rewards):.0f}/{n})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SmolVLA x LIBERO; report a success rate.")
    ap.add_argument("--suite", default="libero_spatial", help="LIBERO task suite")
    ap.add_argument("-n", type=int, default=3, help="tasks (task ids 0..n-1)")
    ap.add_argument("--max-concurrent", type=int, default=1,
                    help="parallel rollouts (each boots its own container)")
    asyncio.run(main(ap.parse_args()))
