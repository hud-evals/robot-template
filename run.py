"""The central run: SmolVLA x LIBERO, end to end. Reports a success rate.

One runner, one agent definition (`inventory/agents/smolvla_libero.py`), one
taskset (`environment/tasks.py`). The flags only change *where* the policy and the
env run — the thing being evaluated is identical, so results are comparable. The
`examples/` scripts are the same three modes spelled out one-per-file.

    python run.py                          # local: policy here, env in Docker
    python run.py --batched 8              # many episodes off one batched GPU forward
    python run.py --full                   # the whole libero_spatial suite (500 episodes)
    python run.py --full --suite all       # the entire 6500-episode benchmark
    python run.py --remote HOST:PORT       # policy served on a GPU box; env in Docker
    python run.py --remote HOST:PORT --modal   # ...and the env in a Modal sandbox

`--remote` points at any openpi/0 policy server (serve one from
`inventory/agents/remote/`, or a stock openpi / Cosmos / hosted pi0.5). With
`HUD_API_KEY` set, every rollout streams to the trace viewer on hud.ai.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from environment.tasks import load_bench, make_tasks
from hud.eval import Taskset

IMAGE_NAME = "hud-libero-env"  # built from Dockerfile.hud (see QUICKSTART.md)


def _build_agent(args: argparse.Namespace):
    """Pick the agent for the requested mode (imported lazily: --remote needs no torch)."""
    if args.remote:
        from inventory.agents.remote.smolvla.smolvla_libero_remote import SmolVLARemoteAgent

        host, _, port = args.remote.rpartition(":")
        return SmolVLARemoteAgent(host=host or "localhost", port=int(port)), f"remote://{args.remote}"

    from inventory.agents.smolvla_libero import SmolVLALiberoAgent

    agent = SmolVLALiberoAgent()
    if args.batched:
        from hud.agents.robot.batching import BatchedAgent

        return BatchedAgent(agent, batch_size=args.batched), f"smolvla x{args.batched} (batched)"
    return agent, "smolvla"


def _runtime(args: argparse.Namespace):
    """Where the env runs: a fresh Docker container, or a Modal sandbox, per rollout."""
    if args.modal:
        from hud.eval import ModalRuntime

        # LIBERO's CPU sim wants the RAM headroom.
        return ModalRuntime(IMAGE_NAME, runtime_config={"resources": {"memory_mb": 8192}})
    from hud.eval import DockerRuntime

    return DockerRuntime(IMAGE_NAME)


async def main(args: argparse.Namespace) -> None:
    agent, policy = _build_agent(args)
    if args.full:  # the whole benchmark from the checked-in full_bench_tasks.json
        tasks = load_bench(None if args.suite == "all" else args.suite)
    else:
        # Default to filling the batch when --batched is set, else 3 (`--group` overrides).
        n = args.group if args.group is not None else (args.batched or 3)
        tasks = make_tasks(suite=args.suite, n=n, init_state_id=args.init_state)
    # RemoteModel is one request per env (not batchable); local runs can fill a batch.
    max_concurrent = args.batched if (args.batched and not args.remote) else 1
    print(f"{len(tasks)} rollout(s) of {args.suite} (policy: {policy})\n", flush=True)

    job = await Taskset("smolvla_libero", tasks).run(
        agent, runtime=_runtime(args), max_concurrent=max_concurrent
    )

    rewards = [run.reward or 0.0 for run in job.runs]
    for task, reward in zip(tasks, rewards, strict=False):
        print(f"  {task.id} {task.args} -> reward={reward}")
    n = len(rewards) or 1
    print(f"\nsuccess_rate={sum(rewards) / n:.2f} ({sum(rewards):.0f}/{n})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SmolVLA x LIBERO; report a success rate.")
    ap.add_argument("--suite", default="libero_spatial", help="LIBERO task suite, or 'all' with --full (see environment/tasks.py)")
    ap.add_argument("--group", type=int, default=None, metavar="N", help="episodes (task ids 0..N-1); default 3, or the batch size with --batched; ignored with --full")
    ap.add_argument("--full", action="store_true", help="run the entire suite from full_bench_tasks.json (500 episodes; 6500 with --suite all)")
    ap.add_argument("--init-state", type=int, default=0, metavar="I", help="saved init state id (0..49)")
    ap.add_argument("--batched", type=int, default=0, metavar="N", help="run N rollouts off one batched GPU forward")
    ap.add_argument("--remote", default=None, metavar="HOST:PORT", help="drive a served openpi/0 policy (see inventory/agents/remote/)")
    ap.add_argument("--modal", action="store_true", help="run the env in a Modal sandbox instead of local Docker")
    asyncio.run(main(ap.parse_args()))
