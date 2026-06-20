"""Drive LIBERO with SmolVLA served remotely — the local side is a thin harness.

The policy weights live on a GPU somewhere (here, the Modal server in
`smolvla_serve.py`); locally you run only the episode loop. The agent is the same
`RobotAgent` as always, but the model is a weightless `RemoteModel(host, port)`
that ships each observation to the openpi server and gets back an action chunk —
so this runs on a laptop with no GPU.

`OpenPIAdapter` (instead of `LeRobotAdapter`) passes the env's raw obs straight
to the server under the openpi wire keys; all pre/post-processing lives on the
server. Point `RemoteModel` at any openpi server (Cosmos, pi0.5, ...) the same way.

    modal run inventory/agents/remote/smolvla/smolvla_serve.py     # prints ws://HOST:PORT
    export POLICY_HOST=... POLICY_PORT=...
    python inventory/agents/remote/smolvla/smolvla_libero_remote.py
"""

from __future__ import annotations

import asyncio
import os

from hud.agents.robot.adapter import OpenPIAdapter
from hud.agents.robot.agent import RobotAgent
from hud.agents.robot.model import RemoteModel
from hud.eval import DockerRuntime, Task, Taskset


class SmolVLARemoteAgent(RobotAgent):
    max_steps = 400

    def __init__(self, host: str, port: int) -> None:
        self.model = RemoteModel(host, port)  # weightless: chunks come from the server
        self.adapter = OpenPIAdapter()        # raw obs -> openpi wire keys; server does pre/post


TASKS = [
    Task(env="libero", id="libero_spatial", args={"libero_task_id": t, "init_state_id": 0})
    for t in range(3)
]


async def main() -> None:
    host, port = os.environ["POLICY_HOST"], int(os.environ["POLICY_PORT"])
    agent = SmolVLARemoteAgent(host, port)
    job = await Taskset("smolvla_libero_remote", TASKS).run(
        agent, runtime=DockerRuntime("hud-libero-env"), max_concurrent=1
    )
    rewards = [run.reward for run in job.runs]
    print(f"success_rate={sum(rewards) / len(rewards):.2f} ({sum(rewards):.0f}/{len(rewards)})")


if __name__ == "__main__":
    asyncio.run(main())
