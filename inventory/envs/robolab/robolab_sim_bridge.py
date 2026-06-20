"""RoboLab (Isaac Lab) sim bridge — owns the simulator, serves it over robot.

Analogous to ``environment/libero_sim_bridge.py`` but for RoboLab
(NVIDIA Isaac Lab, DROID embodiment). The single heavy dependency — booting the
Omniverse ``SimulationApp`` — happens once at import time (RoboLab requires the
app up before any ``isaaclab`` / ``robolab`` import), so importing this module
*is* the sim boot. Keep it out of the lightweight harness / contract paths.

Per the RoboLab README, ``OMNI_KIT_ACCEPT_EULA=Y`` must be set before launch.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

from config import DEVICE  # noqa: E402  (cheap, app-free)

import cv2  # noqa: E402  RoboLab requires cv2 imported before isaaclab — do not move
import numpy as np  # noqa: E402

# Pre-import the full websockets stack BEFORE the app boots — do not move.
# Kit prepends its extension prebundles (which carry websockets 12) to sys.path
# during app startup; importing here pins the site-packages websockets (>=15,
# the one hud's bridge uses) into sys.modules so a later lazy import inside
# ``RobotBridge.start()`` cannot split-brain across the two copies.
import websockets.asyncio.server  # noqa: E402, F401
import websockets.exceptions  # noqa: E402, F401

from isaaclab.app import AppLauncher  # noqa: E402

# Boot Omniverse once (headless; cameras on so the tiled RGB sensors render).
_APP = AppLauncher(headless=True, enable_cameras=True, device=DEVICE)
simulation_app = _APP.app

import robolab.constants  # noqa: E402
import torch  # noqa: E402
from robolab.core.environments.factory import auto_discover_and_create_cfgs, get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.logging.results import get_all_env_subtask_infos  # noqa: E402
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_proprio_obs  # noqa: E402

from debug_view import stitched_view  # noqa: E402  DEBUG ONLY (live MJPEG view)
from hud.environment.robot import RobotBridge  # noqa: E402

# Fractional subtask progress (0..1) is the primary reward; off by default in RoboLab.
robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True

# Contract feature key -> RoboLab image-obs camera term (see camera_presets.WRIST_LEFT).
EXTERIOR_CAM = "over_shoulder_left_camera"
WRIST_CAM = "wrist_cam"
LOG_EVERY = 20


def _action_cfg(control: str):
    """Map the ``control`` decision variable to a RoboLab DROID action config."""
    from robolab.robots.droid import (
        DroidIKActionCfg,
        DroidJointPositionActionCfg,
        DroidRelIKActionCfg,
    )

    return {
        "joint_pos": DroidJointPositionActionCfg,
        "ee_abs": DroidIKActionCfg,
        "ee_del": DroidRelIKActionCfg,
    }[control]()


# Cap the long edge of policy frames sent on the wire. Aspect ratio is PRESERVED
# (no square squash): the policy does the aspect-preserving resize+pad to 224 itself
# (openpi `image_tools.resize_with_pad`), so squashing here would distort the view.
# 640 is comfortably above the 224 model size, so the policy-side resize is lossless
# in aspect while keeping the wire payload small.
POLICY_FRAME_LONG_EDGE = 640


def _resize(frame: np.ndarray) -> np.ndarray:
    """RGB uint8 -> aspect-preserving frame (long edge <= POLICY_FRAME_LONG_EDGE).

    Deliberately does NOT force a square: distortion-free aspect ratio is required
    for the policy's resize_with_pad to match the official benchmark. The square
    contract ``shape`` is advisory metadata only; the agent reads the real array shape.
    """
    h, w = frame.shape[:2]
    long_edge = max(h, w)
    if long_edge > POLICY_FRAME_LONG_EDGE:
        s = POLICY_FRAME_LONG_EDGE / long_edge
        frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame, dtype=np.uint8)


class RobolabSimBridge(RobotBridge):
    """Owns a RoboLab (Isaac Lab) env and serves it over robot.

    One ``control`` mode for the bridge's lifetime (the launch-time decision
    variable, paired with the contract): joint-position (native), absolute-IK,
    or relative-IK end-effector control.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        stream_sink=None,
        control: str = "joint_pos",
        sim_runner=None,
    ) -> None:
        # Isaac/Kit is pinned to the thread that booted it (the process main thread) and
        # drives the event loop itself during reset (USD asset loading), so every sim
        # touch (env create/reset/step/close, ``app.update``) must run there — off the
        # HUD asyncio loop. sim_process injects a MainThreadSimRunner for that; all sim
        # dispatch goes through ``self._sim_runner`` (base bridge).
        # The contract isn't held here — the serving RobotEndpoint carries it.
        super().__init__(host=host, port=port, sim_runner=sim_runner)
        self._stream_sink = stream_sink  # DEBUG ONLY: live MJPEG sink (.add_stream_frame); None disables
        self._control = control
        self._action_cfg = _action_cfg(control)
        self._registered: set[str] = set()  # task names registered with the factory this process
        self._env = None
        self._env_cfg = None
        self._obs: dict | None = None
        # Base inits task_description/total_reward/success/terminated; score is robolab-specific.
        self.score = 0.0          # fractional subtask progress (0..1) — the primary reward
        self._action_dim = 0
        self._step_count = 0

    async def stop(self) -> None:
        print("[env] stopping sim bridge", flush=True)
        await super().stop()
        await self._sim_runner.call(self._close_sim)  # sim teardown on the Isaac thread

    def _close_sim(self) -> None:
        if self._env is not None:
            self._env.close()
        simulation_app.close()

    # ── task setup ───────────────────────────────────────────────────────
    def _register(self, task_name: str) -> str:
        """Register ``task_name`` against the chosen action config; return its env name.

        Registers lazily (only the requested task — far faster than the full
        120-task discovery) and caches the result. Mirrors the RoboLab DROID
        registration (WRIST_LEFT cameras + proprio + mirrored viewport) but with
        the action config selected by the ``control`` mode (``BENCH_CONTROL``).
        """
        if task_name not in self._registered:
            from robolab.core.observations.observation_utils import (
                generate_image_obs_from_cameras,
                generate_obs_cfg,
            )
            from robolab.registrations.droid.camera_presets import WRIST_LEFT
            from robolab.robots.droid import DroidCfg, ProprioceptionObservationCfg, WristCameraCfg, contact_gripper
            from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
            from robolab.variations.camera import EgocentricMirroredCameraCfg
            from robolab.variations.lighting import SphereLightCfg

            ImageObsCfg = generate_image_obs_from_cameras(WRIST_LEFT)
            ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
            obs_cfg = generate_obs_cfg({
                "image_obs": ImageObsCfg(),
                "proprio_obs": ProprioceptionObservationCfg(),
                "viewport_cam": ViewportCameraCfg(),
            })
            scene_cameras = [c for c in WRIST_LEFT if c is not WristCameraCfg]
            auto_discover_and_create_cfgs(
                task_subdirs=robolab.constants.DEFAULT_TASK_SUBFOLDERS,
                tasks=[task_name],
                observations_cfg=obs_cfg(),
                actions_cfg=self._action_cfg,
                robot_cfg=DroidCfg,
                camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
                lighting_cfg=SphereLightCfg,
                background_cfg=HomeOfficeBackgroundCfg,
                contact_gripper=contact_gripper,
                dt=1 / (60 * 2),
                render_interval=8,
                decimation=8,
                seed=1,
            )
            self._registered.add(task_name)
        # env_postfix="" => env name == task class name; resolve via the factory to be safe.
        return get_envs(task=[task_name])[0]

    async def reset(self, task_name: str, seed: int = 0, instruction_type: str = "default") -> str:
        """Build the selected RoboLab task, settle it, return the prompt.

        The simulator work runs on the Isaac thread (via the runner); the base
        ``_reset`` wrapper resets scoring and broadcasts the first obs on the HUD loop.
        """
        await self._sim_runner.call(self._reset_sim, task_name, seed, instruction_type)
        return self.task_description

    def _reset_sim(self, task_name: str, seed: int, instruction_type: str) -> None:
        env_name = self._register(task_name)
        print(f"[env] loading task {env_name} (seed={seed}, instruction={instruction_type}, control={self._control})", flush=True)

        if self._env is not None:
            self._env.close()
        self._env, self._env_cfg = create_env(
            env_name, device=DEVICE, num_envs=1, use_fabric=True,
            seed=seed, instruction_type=instruction_type,
        )
        self.task_description = self._env_cfg.instruction
        self._action_dim = int(self._env.action_manager.total_action_dim)

        # Two cold resets to stabilize the scene + populate the tiled camera sensors.
        self._obs, _ = self._env.reset()
        self._obs, _ = self._env.reset()
        self._wait_until_playing()

        self.score = 0.0  # robolab-specific scoring (not reset by the base)
        self._step_count = 0
        print(f"[env] reset done: {self.task_description!r}", flush=True)
        self._emit_stream_frame()  # DEBUG ONLY (live view)

    def result(self) -> dict:
        """Episode score: fractional subtask progress (primary) + binary success."""
        return {
            "score": self.score,
            "success": bool(self.success),
            "total_reward": float(self.total_reward),
        }

    # ── robot hooks (run on the sim thread; dispatched by the base bridge) ──
    def step(self, action: np.ndarray) -> None:
        if self.terminated or self._env is None:
            return
        self._wait_until_playing()
        actions = torch.zeros(1, self._action_dim, device=self._env.device)
        a = torch.as_tensor(np.asarray(action, dtype=np.float32), device=self._env.device)
        actions[0, : a.shape[0]] = a[: self._action_dim]

        self._obs, reward, term, trunc, _ = self._env.step(actions)
        self.total_reward += float(reward[0].item())
        self.success = bool(term[0].item())
        self.terminated = bool(self._env.all_terminated)
        self.score = self._subtask_score()
        self._step_count += 1

        self._emit_stream_frame()  # DEBUG ONLY (live view)
        if self.terminated or self._step_count % LOG_EVERY == 0:
            print(
                f"[env] step {self._step_count} | score={self.score:.3f} "
                f"total={self.total_reward:.3f} terminated={self.terminated} success={self.success}",
                flush=True,
            )

    def get_observation(self) -> tuple[dict[str, np.ndarray], bool] | None:
        if self._obs is None:
            return None
        images = unpack_image_obs(self._obs, env_id=0)
        proprio = unpack_proprio_obs(self._obs, env_id=0)
        # Keys match the contract's observation feature names (see config.CONTRACT):
        #   observation/exterior_image_1_left / observation/wrist_image_left : the two policy
        #     RGB views (aspect-preserving, long edge capped; the policy resize_with_pads to 224)
        #   observation/state : 8-dim joint proprio = 7 Panda arm joints + gripper opening [0,1]
        data = {
            "observation/exterior_image_1_left": _resize(images[EXTERIOR_CAM]),
            "observation/wrist_image_left": _resize(images[WRIST_CAM]),
            "observation/state": np.concatenate([
                np.asarray(proprio["arm_joint_pos"], dtype=np.float32).reshape(-1),
                np.asarray(proprio["gripper_pos"], dtype=np.float32).reshape(-1),
            ]),
        }
        return data, self.terminated

    # ── internals ─────────────────────────────────────────────────────────
    def _subtask_score(self) -> float:
        """Latest fractional subtask progress (0..1) for env 0; falls back to success."""
        infos = get_all_env_subtask_infos(self._env)
        if infos and infos[0] is not None and infos[0].get("score") is not None:
            return float(infos[0]["score"])
        return 1.0 if self.success else 0.0

    def _wait_until_playing(self) -> None:
        """RoboLab requires the Omniverse timeline to be playing before stepping."""
        import omni.kit.app
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        app = omni.kit.app.get_app()
        while not timeline.is_playing():
            app.update()

    # ── DEBUG ONLY (live MJPEG view; not the robot protocol) ───────────────
    # Called from reset()/step() purely to feed the http://localhost:8080 stream.
    # Drop _stream_sink + these two callsites and the bridge is unchanged.
    def _emit_stream_frame(self) -> None:
        if self._stream_sink is None or self._obs is None:
            return
        frame = stitched_view(self._obs, EXTERIOR_CAM, WRIST_CAM)
        if frame is not None:
            self._stream_sink.add_stream_frame(frame)


__all__ = ["RobolabSimBridge", "simulation_app"]
