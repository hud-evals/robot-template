"""RoboLab sim process — Omniverse + the robot bridge, alone in its own process.

Isaac/Kit owns the main thread it booted on **and** the asyncio event loop:
``omni.kit.async_engine`` creates one main-thread loop, monkeypatches the loop class
globally, and ticks it from ``app.update``. So we don't start our own loop — running a
second one (even on another thread) collides with Kit's global patches. Instead HUD's
servers run on *Kit's* loop, and the main thread ticks the loop and drains sim touches:

    main thread (Kit's loop)   HUD's servers: the robot WebSocket (agent step/obs) + a
                               RobotEndpoint serving the control surface over JSON-RPC.
    main thread (between ticks) reset/step/close run here via ``MainThreadSimRunner.drain``,
                               *outside* any task — so Isaac's reset (USD load ->
                               ``run_until_complete``) doesn't nest inside a running task.

``env.py`` drives this from another process via ``RobotEndpoint.remote(...)``; the wire
contract stays env-side. ``env.py`` connects to this (it does not spawn it by default),
so run it yourself in the Isaac interpreter:

    python sim_process.py        # control mode via BENCH_CONTROL (default joint_pos)
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from config import CONTROL, ENDPOINT_HOST, ENDPOINT_PORT, STREAM_FPS, STREAM_PORT

sys.path.append(str(Path(__file__).resolve().parent.parent))  # envs/ for the shared streamer
from env_streamer import MJPEGStreamer  # noqa: E402  DEBUG ONLY (live MJPEG view)

from hud.environment.robot import MainThreadSimRunner, RobotEndpoint  # noqa: E402

# Booting Omniverse MUST happen before the loop is touched (Kit's ``async_engine`` owns
# it). Importing the bridge boots the SimulationApp at module top; it also re-exports the
# app handle we tick below.
from robolab_sim_bridge import RobolabSimBridge, simulation_app  # noqa: E402


def main() -> None:
    print(f"[sim] control={CONTROL}", flush=True)
    runner = MainThreadSimRunner()  # sim touches run on this (main) thread, outside tasks
    streamer = MJPEGStreamer(host="0.0.0.0", port=STREAM_PORT, fps=STREAM_FPS)  # DEBUG ONLY
    bridge = RobolabSimBridge(stream_sink=streamer, control=CONTROL, sim_runner=runner)
    endpoint = RobotEndpoint(bridge)  # contract is env-side, not here

    loop = asyncio.get_event_loop()  # Kit's loop (omni.kit.async_engine set it at boot)

    async def _bringup() -> None:
        await streamer.start()  # DEBUG ONLY
        print(f"[sim] live view: http://localhost:{STREAM_PORT}/", flush=True)
        await bridge.start()  # the robot WebSocket the agent steps
        await endpoint.serve(ENDPOINT_HOST, ENDPOINT_PORT)  # start/stop/reset/result/url for env.py

    loop.create_task(_bringup())
    try:
        while simulation_app.is_running():
            loop.run_once()  # service HUD's servers (no app.update -> no sim advance)
            runner.drain()   # run queued reset/step/close on main, outside any task
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
