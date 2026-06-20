"""SCAFFOLD env — the declarative authoring surface. Copy, fill the TODOs.

Serve it with:   python -m hud.environment.server env.py --port 9001

`RobotEndpoint` is the single handle the env drives the bridge through
(start/stop/reset/result/url) — identical whether the sim is local (shown here)
or in another process (`RobotEndpoint.remote(...)`; see inventory/envs/robolab).
Reference: `environment/env.py`. Docs: https://docs.hud.ai/v6/core/robots
"""

from __future__ import annotations

from config import CONTRACT, TASKS
from my_sim_bridge import MySimBridge

from hud.capabilities import Capability
from hud.environment import Environment
from hud.environment.robot import RobotEndpoint

bridge = MySimBridge()          # the env owns the contract, NOT the bridge
endpoint = RobotEndpoint(bridge)
env = Environment(name="my-env")  # TODO: name it


@env.initialize
async def _up() -> None:
    await endpoint.start()  # binds the bridge's ephemeral robot WebSocket
    # Publish the robot capability once the URL exists (after start()).
    env.add_capability(Capability.robot(name="robot", url=await endpoint.url(), contract=CONTRACT))


@env.shutdown
async def _down() -> None:
    await endpoint.stop()


def _make_task(task_id: str):
    # One async generator per task: yield the prompt, then yield the result.
    async def task(seed: int = 0):
        prompt = await endpoint.reset(task_id=task_id, seed=seed)
        yield {"prompt": prompt}
        yield await endpoint.result()  # {"score", "success", "total_reward"}

    return task


for task_id in TASKS:
    env.template(id=task_id, description=f"my-env {task_id}")(_make_task(task_id))
