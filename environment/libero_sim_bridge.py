from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa") # default to software rendering so the bridge runs on a CPU-only container

sys.path.insert(0, str(Path(__file__).resolve().parent))  # siblings resolve from repo root or here

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from hud.environment.robot import RobotBridge
from config import IMAGE_SIZE
from env_streamer import stitch
from utilities import quat2axisangle

# No-op action used to let objects settle after a reset (gripper open).
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
SETTLE_STEPS = 10
# How often (in env steps) to log a progress line.
LOG_EVERY = 20


class LiberoSimBridge(RobotBridge):
    """Owns a LIBERO env and serves it over robot."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, stream_sink=None, use_delta: bool = True) -> None:
        super().__init__(host=host, port=port)
        self._stream_sink = stream_sink   # optional debug stream sink with .add_stream_frame(frame); None disables
        # Decision variable (paired with the contract): delta vs absolute eef control.
        self._use_delta = use_delta
        self._env: OffScreenRenderEnv | None = None
        self._obs: dict | None = None
        self.task_description = ""
        self.total_reward = 0.0
        self.terminated = False
        self.success = False  # task goal actually satisfied (check_success), not just episode end
        self._step_count = 0

    async def stop(self) -> None:
        print("[env] stopping sim bridge", flush=True)
        await super().stop()
        if self._env is not None:
            self._env.close()

    async def reset(self, task_suite: str, task_id: int, init_state_id: int = 0) -> str:
        """Build the selected LIBERO task + initial state, settle it, broadcast the first obs."""
        suite = benchmark.get_benchmark_dict()[task_suite]()
        task = suite.get_task(task_id)
        self.task_description = task.language
        print(f"[env] loading task [{task_suite}:{task_id}:{init_state_id}] {task.language!r}", flush=True)

        if self._env is not None:
            self._env.close()
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(f"[env] bddl: {bddl}", flush=True)
        
        # the main env object
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl, camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE
        )
        self._env.reset()
        self._env.set_init_state(suite.get_task_init_states(task_id)[init_state_id])

        # standard procedure: do 10 cold steps to stabilize the scene
        for _ in range(SETTLE_STEPS):
            obs, *_ = self._env.step(DUMMY_ACTION)

        # set the controller's control mode (delta vs absolute eef targets) to
        # match the action branch this env advertised in its contract.
        for robot in self._env.robots:
            robot.controller.use_delta = self._use_delta

        self._obs = obs
        self._step_count = 0
        print("[env] reset done", flush=True)
        self._emit_stream_frame()  # debug video sink (not the robot protocol obs)
        return self.task_description


    ''' purely local functions '''
    def step(self, action: np.ndarray) -> None:
        if self.terminated:
            return
        self._obs, reward, done, _ = self._env.step(np.asarray(action, dtype=np.float64))
        self.total_reward += reward
        # Score is dished out on success only; `done` alone (e.g. horizon timeout)
        # must NOT count as a win. `terminated` still ends the episode on either.
        # CAVEAT: check_success() is an instantaneous, per-step predicate with no
        # dwell requirement, and we latch the FIRST step it's True (terminate +
        # early-return). So a single transient frame where the goal momentarily
        # holds (e.g. the bowl brushing the plate mid-trajectory, even while still
        # gripped and never stably placed) counts as a win. This is faithful to
        # LIBERO's standard "success on first satisfaction" eval convention, so it
        # is intentionally lenient — not a bug. For a stricter placement metric,
        # require the predicate to hold for N consecutive steps (and after gripper
        # release) before setting success.
        self.success = bool(self._env.check_success())
        self.terminated = bool(done) or self.success
        self._step_count += 1
   
        self._emit_stream_frame()
        if self.terminated or self._step_count % LOG_EVERY == 0:
            print(
                f"[env] step {self._step_count} | reward={reward:.3f} "
                f"total={self.total_reward:.3f} terminated={self.terminated} success={self.success}",
                flush=True,
            )

    def get_observation(self) -> tuple[dict[str, np.ndarray], bool] | None:
        if self._obs is None:
            return None
        # Keys match the contract's observation feature names (see config.CONTRACT),
        # which use the OpenPI LIBERO naming; the camera reads below still use the
        # sim-native robosuite keys:
        #   observation/image / observation/wrist_image : the two 256x256 RGB views
        #   observation/state : 8-dim proprio = eef pos + axis-angle orient + gripper
        #           qpos (ee_abs; from robot0_eef_pos / _eef_quat / _gripper_qpos)
        data = {
            "observation/image": self._get_frame("agentview_image"),
            "observation/wrist_image": self._get_frame("robot0_eye_in_hand_image"),
            "observation/state": np.concatenate([
                np.asarray(self._obs["robot0_eef_pos"], dtype=np.float32),
                quat2axisangle(np.asarray(self._obs["robot0_eef_quat"], dtype=np.float32)),
                np.asarray(self._obs["robot0_gripper_qpos"], dtype=np.float32),
            ]),
        }
        return data, self.terminated

    def _get_frame(self, camera: str) -> np.ndarray:
        # Rotate 180 deg (flip H and W) to match the LeRobot/LIBERO camera convention.
        return np.ascontiguousarray(np.asarray(self._obs[camera])[::-1, ::-1], dtype=np.uint8)

    def _emit_stream_frame(self) -> None:
        """Push a side-by-side [agentview | wrist] debug frame to the stream sink (if any)."""
        if self._stream_sink is None or self._obs is None:
            return
        self._stream_sink.add_stream_frame(stitch([
            self._get_frame("agentview_image"),
            self._get_frame("robot0_eye_in_hand_image"),
        ]))


__all__ = ["LiberoSimBridge", "IMAGE_SIZE"]
