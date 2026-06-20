"""SimplerEnv (WidowX/Bridge on ManiSkill3) — declarative HUD env.

Serve with::

    python -m hud.environment.server env.py

Then open  http://localhost:8080/  to watch the live MJPEG debug stream.

Launch-time decision variables (environment variables):

    BENCH_CONTROL=delta|absolute   EE action mode: contract branch + WidowX sim
                                   control mode (default: delta; absolute mirrors
                                   the X-VLA authors' Simpler-WidowX setup)
    BENCH_STREAM_PORT=8080         MJPEG live-view HTTP port
    HUD_RECORD_DIR=...             record episodes as a LeRobot v3 dataset
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # envs/ for the shared streamer

from config import STREAM_FPS, STREAM_PORT, TASKS, build_contract
from env_streamer import MJPEGStreamer
from simpler_sim_bridge import SimplerSimBridge

from hud.capabilities import Capability
from hud.environment import Environment
from hud.environment.robot import RobotEndpoint

PARAMS = {"use_delta": os.environ.get("BENCH_CONTROL", "delta") == "delta"}
CONTRACT = build_contract(PARAMS)  # env owns the contract

streamer = MJPEGStreamer(host="0.0.0.0", port=STREAM_PORT, fps=STREAM_FPS)
bridge = SimplerSimBridge(stream_sink=streamer, use_delta=PARAMS["use_delta"])
endpoint = RobotEndpoint(bridge)  # drives the same-process bridge

env = Environment(name="simpler")


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


def _make_task(task: str):
    async def task_fn(episode_id: int = 0):
        print(f"[env] task start: {task} episode_id={episode_id}", flush=True)
        prompt = await endpoint.reset(task=task, task_id=episode_id)
        yield {"prompt": prompt}
        yield await endpoint.result()

    return task_fn


for task in TASKS:
    env.template(id=task, description=f"SimplerEnv {task}")(_make_task(task))
