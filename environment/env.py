"""LIBERO as one declarative HUD env — the task is an episodic template arg.

The sim defers lerobot's ``make_env`` to first reset and rebuilds only when
(task_suite, task_id) changes (see ``lerobot_sim.py``), so this one module-level
declaration serves the whole benchmark: ``hud serve environment/env.py``, or
``LocalRuntime("environment/env.py")`` from a runner.
"""

from pathlib import Path

from lerobot_sim import LeRobotSim

from hud import Environment

env = Environment(name="libero")
sim = env.gym(LeRobotSim(), contract=Path(__file__).parent / "contract_lerobot.json")


@env.template()
async def episode(task_suite: str = "libero_spatial", task_id: int = 0, seed: int = 0):
    """One LIBERO episode; the sim provides the instruction as the prompt."""
    prompt = await sim.reset(task_suite=task_suite, task_id=task_id, seed=seed)
    yield {"prompt": prompt}
    yield await sim.result()
