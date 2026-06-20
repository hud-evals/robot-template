"""LIBERO — declarative HUD env (canonical authoring surface).

This is the served environment: ``Dockerfile.hud``'s CMD runs
``hud serve environment/env:env``, and ``hud deploy`` reads the env name from the
``Environment(...)`` call below. Serve it directly the same way::

    hud serve environment/env:env                      # from the repo root
    python -m hud.environment.server environment/env.py --port 9001

Then open  http://localhost:8080/  to watch the live MJPEG debug stream.

Launch-time decision variables travel as environment variables (argparse has
no place in a declare-only module):

    BENCH_CONTROL=delta|absolute   eef action mode: selects the contract branch
                                   + the sim controller (default: delta)
    BENCH_STREAM_PORT=8080         MJPEG live-view HTTP port
    HUD_RECORD_DIR=...             record episodes as a LeRobot v3 dataset

The robot bridge binds an ephemeral loopback port and publishes its concrete
address from the ``@env.initialize`` hook; agents reach it through the control
channel's capability tunnel, so the control port is the only port to expose.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # serve from repo root or here

from config import STREAM_FPS, STREAM_PORT, TASK_SUITES, build_contract
from env_streamer import MJPEGStreamer
from libero_sim_bridge import LiberoSimBridge

from hud.capabilities import Capability
from hud.environment import Environment
from hud.environment.robot import RobotEndpoint

PARAMS = {"use_delta": os.environ.get("BENCH_CONTROL", "delta") == "delta"}
CONTRACT = build_contract(PARAMS)  # env owns the contract

streamer = MJPEGStreamer(host="0.0.0.0", port=STREAM_PORT, fps=STREAM_FPS)
# The endpoint drives the (same-process) bridge: start/stop/reset/result/url.
bridge = LiberoSimBridge(stream_sink=streamer, use_delta=PARAMS["use_delta"])
endpoint = RobotEndpoint(bridge)

env = Environment(name="libero")


@env.initialize
async def _up() -> None:
    print(f"[env] params={PARAMS} -> action '{CONTRACT['features']['action']['type']}'", flush=True)
    await streamer.start()
    print(f"[env] live view: http://localhost:{STREAM_PORT}/", flush=True)
    await endpoint.start()
    env.add_capability(Capability.robot(name="robot", url=await endpoint.url(), contract=CONTRACT))


@env.shutdown
async def _down() -> None:
    await endpoint.stop()
    await streamer.stop()


def _make_task(task_suite: str):
    async def task(libero_task_id: int, init_state_id: int = 0):
        print(f"[env] task start: {task_suite}:{libero_task_id}:{init_state_id}", flush=True)
        prompt = await endpoint.reset(
            task_suite=task_suite, task_id=libero_task_id, init_state_id=init_state_id
        )
        yield {"prompt": prompt}
        yield await endpoint.result()

    return task


for suite in TASK_SUITES:
    env.template(id=suite, description=f"LIBERO {suite} task suite")(_make_task(suite))
