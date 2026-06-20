"""Env process sim bridge for SimplerEnv (WidowX/Bridge on ManiSkill3).

Analogous to ``environment/libero_sim_bridge.py``: owns one sim env
and serves it over ``robot``. Each received action advances the sim one control
step; observations are emitted under the contract's feature names (``image``,
``state``) so the agent wires itself up purely from the manifest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# gymnasium is the env API; importing bridge_dataset_eval REGISTERS the WidowX
# task ids (PutCarrotOnPlateInScene-v1, ...) with gym.
import gymnasium as gym
from mani_skill.envs.tasks.digital_twins import bridge_dataset_eval  # noqa: F401  (registers envs)

from config import CAMERA, OBS_MODE
from hud.environment.robot import RobotBridge

# benchmarks/ holds shared helpers across demos (envs/<env>/ -> benchmarks/).
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utilities import quat2axisangle  # noqa: E402

# How often (in env steps) to log a progress line.
LOG_EVERY = 10


class SimplerSimBridge(RobotBridge):
    """Owns a SimplerEnv (ManiSkill3) env and serves it over robot."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = 0, stream_sink=None, use_delta: bool = True,
    ) -> None:
        super().__init__(host=host, port=port)
        self._stream_sink = stream_sink   # optional debug stream sink with .add_stream_frame(frame); None disables
        self._use_delta = use_delta       # delta vs absolute EE-target control (paired with the contract)
        self._env = None
        self._obs: dict | None = None
        self.task_description = ""
        self.total_reward = 0.0
        self.terminated = False
        self.success = False  # task goal actually satisfied (info["success"]), not just episode end
        self._step_count = 0

    async def stop(self) -> None:
        print("[env] stopping sim bridge", flush=True)
        await super().stop()
        if self._env is not None:
            self._env.close()
            self._env = None

    async def reset(self, task: str, task_id: int) -> str:
        """Build the selected WidowX/Bridge task, reset it, and broadcast the first obs.

        ``task`` is a gym env id (a scenario); ``task_id`` is the deterministic
        ``episode_id`` — it indexes the task's predefined object pose/orientation
        variants (``len(xyz_configs) * len(quat_configs)`` of them, wrapping mod
        that total), so sweeping ``episode_id`` covers every variant exactly once.
        It is also used as the RNG seed (model-scale / settle randomness) so a
        given ``task_id`` is fully reproducible.
        """
        print(f"[env] loading task [{task}] episode_id={task_id}", flush=True)

        # ManiSkill envs are GPU-backed; rebuild per scenario for a clean scene.
        if self._env is not None:
            self._env.close()
            self._env = None
        self._env = gym.make(task, obs_mode=OBS_MODE, num_envs=1)

        # Optional horizon override. The native SimplerEnv caps (carrot/spoon/stack=60,
        # eggplant=120) are the canonical benchmark and stay the default. But the official
        # GR00T eval lifts the cap (env._max_episode_steps = 10000) so a retry-heavy policy
        # gets enough attempts; opt into that here via SIMPLER_MAX_EPISODE_STEPS. We set it
        # on ManiSkill3's TimeLimitWrapper (._max_episode_steps), which owns truncation.
        max_override = int(os.environ.get("SIMPLER_MAX_EPISODE_STEPS", "0") or 0)
        if max_override > 0:
            w = self._env
            while w is not None:
                if hasattr(w, "_max_episode_steps"):
                    w._max_episode_steps = max_override
                    break
                w = getattr(w, "env", None)

        obs, _info = self._env.reset(seed=int(task_id), options={"episode_id": int(task_id)})
        # Pair the EE controller's control mode (delta vs absolute targets) to the
        # action branch this env advertised in its contract, mirroring libero's
        # `robot.controller.use_delta = ...`.
        self._env.unwrapped.agent.controller.controllers["arm"].config.use_delta = self._use_delta
        # one language goal per (parallel) env; single-env case -> index 0.
        self.task_description = self._env.unwrapped.get_language_instruction()[0]

        self._obs = obs
        self._step_count = 0
        print(f"[env] reset done | instruction={self.task_description!r}", flush=True)
        self._emit_stream_frame()  # debug video sink (not the robot protocol obs)
        return self.task_description

    """ purely local functions """
    def step(self, action: np.ndarray) -> None:
        if self.terminated or self._env is None:
            return
        # ManiSkill expects a batched (num_envs, action_dim) action.
        act = np.asarray(action, dtype=np.float32)[None]
        self._obs, reward, terminated, truncated, info = self._env.step(act)
        self.last_reward = float(np.asarray(reward).reshape(-1)[0])
        self.total_reward += self.last_reward
        # Score is dished out on success only; a horizon timeout (truncated) or a
        # bare `terminated` must NOT count as a win, but either still ends the episode.
        self.success = bool(np.asarray(info["success"]).reshape(-1)[0])
        self.terminated = (
            bool(np.asarray(terminated).reshape(-1)[0])
            or bool(np.asarray(truncated).reshape(-1)[0])
            or self.success
        )
        self._step_count += 1

        self._emit_stream_frame()
        if self.terminated or self._step_count % LOG_EVERY == 0:
            print(
                f"[env] step {self._step_count} | reward={float(np.asarray(reward).reshape(-1)[0]):.3f} "
                f"total={self.total_reward:.3f} terminated={self.terminated} success={self.success}",
                flush=True,
            )

    def get_observation(self) -> tuple[dict[str, np.ndarray], bool] | None:
        if self._obs is None:
            return None
        # Keys match the contract's observation feature names (see config / simpler_*.json):
        #   observation/image : the single external 3rd_view_camera RGB (H x W x 3 uint8)
        #   observation/state : 8-dim ee_abs proprio = base-frame EE pose (pos + axis-angle) + 2 finger qpos
        data = {"observation/image": self._get_frame(), "observation/state": self._get_state()}
        return data, self.terminated

    def _get_state(self) -> np.ndarray:
        # Base-frame EE pose proprio, mirroring LIBERO's ee_abs state:
        # [pos(3), axis-angle(3), gripper finger qpos(2)]. The arm controller tracks
        # the current EE pose in the robot base frame for either control mode.
        ee_pose = self._env.unwrapped.agent.controller.controllers["arm"].ee_pose_at_base
        pos = np.asarray(ee_pose.p[0].cpu().numpy(), dtype=np.float32).reshape(3)
        quat_wxyz = np.asarray(ee_pose.q[0].cpu().numpy(), dtype=np.float32).reshape(4)
        gripper = np.asarray(self._obs["agent"]["qpos"][0, -2:].cpu().numpy(), dtype=np.float32)
        # sapien/ManiSkill quaternion is (w, x, y, z); quat2axisangle wants (x, y, z, w).
        return np.concatenate([pos, quat2axisangle(np.roll(quat_wxyz, -1)), gripper]).astype(np.float32)

    def _get_frame(self) -> np.ndarray:
        # ManiSkill returns batched torch tensors (num_envs, H, W, 3); take env 0.
        frame = self._obs["sensor_data"][CAMERA]["rgb"][0].cpu().numpy()
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def _emit_stream_frame(self) -> None:
        """Push the external-camera RGB frame to the stream sink (if any)."""
        if self._stream_sink is None or self._obs is None:
            return
        self._stream_sink.add_stream_frame(self._get_frame())


__all__ = ["SimplerSimBridge"]
