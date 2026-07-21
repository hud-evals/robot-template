"""LIBERO as one declarative HUD env — the task is an episodic template arg.

``env.gym(make_env)`` spawns the sim child; templates drive episodes through the
returned ``RobotEndpoint`` (``reset`` → prompt, ``result`` → grade). Serve with
``hud serve environment/env.py``, or point a runtime at this module from a runner.
"""

from pathlib import Path

from lerobot_sim import make_env

from hud import Environment

env = Environment(name="libero")
sim = env.gym(make_env, contract=str(Path(__file__).parent / "contract_lerobot.json"))


@env.template()
async def episode(task_suite: str = "libero_spatial", task_id: int = 0, seed: int = 0):
    """One LIBERO episode; the sim provides the instruction as the prompt."""
    ep = await sim.reset(task_suite=task_suite, task_id=task_id, seed=seed)
    yield {"prompt": ep["prompt"]}  # single env: the slot token is optional
    yield await sim.result()
