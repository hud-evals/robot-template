"""SCAFFOLD — drive an env with a remotely-served policy. Fill the TODOs.

The local side is just the episode loop: `RemoteModel(host, port)` ships each
observation to your openpi policy server and gets back an action chunk, so this
runs anywhere (no GPU needed locally). Working example:
`inventory/agents/remote/smolvla/smolvla_libero_remote.py`.

    export POLICY_HOST=...  POLICY_PORT=...     # from your serve script
    python inventory/agents/remote/scaffold/my_remote_agent.py
"""

from __future__ import annotations

import asyncio
import os

from hud.agents.robot.adapter import OpenPIAdapter
from hud.agents.robot.agent import RobotAgent
from hud.agents.robot.model import RemoteModel
from hud.eval import DockerRuntime, Task, Taskset


class MyRemoteAgent(RobotAgent):
    max_steps = 400

    def __init__(self, host: str, port: int) -> None:
        # response_key="actions" is stock openpi; pass response_key="action" for Cosmos.
        self.model = RemoteModel(host, port)
        self.adapter = OpenPIAdapter()  # raw obs -> openpi wire keys; server does pre/post


# TODO: your env id + task args (see inventory/envs/<env>/env.py for the signature).
TASKS = [Task(env="TODO-env", id="TODO-suite", args={})]


async def main() -> None:
    host, port = os.environ["POLICY_HOST"], int(os.environ["POLICY_PORT"])
    agent = MyRemoteAgent(host, port)
    job = await Taskset("my_remote_run", TASKS).run(
        agent,
        runtime=DockerRuntime("TODO-env-image"),  # or ModalRuntime / DaytonaRuntime
        max_concurrent=1,
    )
    rewards = [run.reward for run in job.runs]
    print(f"success_rate={sum(rewards) / len(rewards):.2f} ({sum(rewards):.0f}/{len(rewards)})")


if __name__ == "__main__":
    asyncio.run(main())
