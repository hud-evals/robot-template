"""RoboLab — declarative HUD env (canonical authoring surface).

RoboLab's simulator (Isaac/Omniverse) must own a process main thread, so it runs
in a *separate* process (``sim_process.py``). This module is the thin control
surface over it: ``@env.initialize`` connects to that process and a
``RobotEndpoint.remote(...)`` drives its reset/result over JSON-RPC — the same
endpoint a same-process env would use, just pointed at another process. The
contract lives here (env-side); only the bridge address is fetched over the link.
The agent's step/observation loop tunnels straight through the control channel to
the sim's bridge — it never touches this process.

``sim_process.py`` needs Isaac's interpreter (``isaaclab`` + the kit runtime),
which is usually *not* the interpreter serving this env. So by default the env
does NOT spawn it — run the sim yourself in the Isaac env, then serve this::

    # 1) in the Isaac interpreter, bring the sim up (binds the endpoint):
    python sim_process.py
    # 2) serve the env; it connects to the running sim:
    python -m hud.environment.server env.py --port 9001

Set ``ROBOLAB_SIM_AUTOSPAWN=1`` to have the env spawn ``sim_process.py`` itself
(only works when this interpreter can import Isaac, e.g. a container whose
``ROBOLAB_SIM_PYTHON`` points at Isaac's ``python.sh``).

    BENCH_CONTROL=joint_pos|ee_abs|ee_del   action mode (default joint_pos)
    ROBOLAB_SIM_AUTOSPAWN=1                   env spawns sim_process.py (default: off)
    ROBOLAB_SIM_PYTHON=...                    interpreter used when autospawning
                                            (default sys.executable)
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from config import CONTROL, ENDPOINT_HOST, ENDPOINT_PORT, build_contract

from hud.capabilities import Capability
from hud.environment import Environment
from hud.environment.robot import RobotEndpoint

_DIR = Path(__file__).resolve().parent
_SIM_PYTHON = os.environ.get("ROBOLAB_SIM_PYTHON", sys.executable)
_AUTOSPAWN = os.environ.get("ROBOLAB_SIM_AUTOSPAWN", "0").lower() in ("1", "true", "yes")
CONTRACT = build_contract({"control": CONTROL})  # env owns the contract (matches the sim's --control)
endpoint = RobotEndpoint.remote(ENDPOINT_HOST, ENDPOINT_PORT)  # connected in @initialize
env = Environment(name="robolab")
_sim: asyncio.subprocess.Process | None = None


@env.initialize
async def _up() -> None:
    global _sim
    if _AUTOSPAWN:
        # Spawn the Isaac process (inherits env vars: BENCH_CONTROL, HUD_RECORD_DIR, ...).
        # Only works if _SIM_PYTHON can import Isaac (isaaclab + kit runtime).
        _sim = await asyncio.create_subprocess_exec(_SIM_PYTHON, "sim_process.py", cwd=str(_DIR))
        print(f"[env] spawned sim process pid={_sim.pid}; waiting for it to boot...", flush=True)
    else:
        print(f"[env] connecting to externally-run sim_process at {ENDPOINT_HOST}:{ENDPOINT_PORT} "
              f"(set ROBOLAB_SIM_AUTOSPAWN=1 to spawn it here)", flush=True)
    await endpoint.connect()  # retries through the ~2 min Omniverse boot
    url = await endpoint.url()  # the only thing the env needs off the link (contract is local)
    print(f"[env] linked sim: bridge={url} "
          f"action='{CONTRACT['features']['action']['type']}'", flush=True)
    env.add_capability(Capability.robot(name="robot", url=url, contract=CONTRACT))


@env.shutdown
async def _down() -> None:
    await endpoint.close()
    if _sim is not None and _sim.returncode is None:
        _sim.terminate()
        await _sim.wait()


@env.template(
    id="robolab",
    description="A RoboLab benchmark task (parametrized by task_name/seed/instruction_type)",
)
async def robolab(task_name: str, seed: int = 0, instruction_type: str = "default"):
    print(f"[env] task start: {task_name} seed={seed} instruction={instruction_type}", flush=True)
    prompt = await endpoint.reset(task_name=task_name, seed=seed, instruction_type=instruction_type)
    yield {"prompt": prompt}
    yield await endpoint.result()
