"""Env configuration: bind address, exposed scenarios, and the env contract.

Mirrors ``environment/config.py``. The env advertises a single
source-of-truth *contract* in its capability manifest (see ``env.py``): one of
the local per-mode contracts (``simpler_ee_del.json`` / ``simpler_ee_abs.json``),
carrying the robot type, control rate, and every action/observation feature with
its layout and value ranges. The agent reads the
contract back from the manifest and splits it into action/observation spaces via
``RobotClient.spaces()``, so it wires observations -> policy inputs without any
shared config (see the agents in ``inventory/agents/``).

Contract feature keys are authoritative on the wire: the bridge emits its
observations under the contract's feature names (``image``, ``state``), so the
env and the agent agree on naming purely via the manifest.

Each feature carries:
- ``role``   : "observation" | "action"   (direction)
- ``type``   : "rgb" | "ee_del" | "joint_pos" | ...  (representation)
- ``dtype`` / ``shape`` / ``names`` : layout
- ``stats``  : per-field value ranges (min/max; mean/std are policy-side for SIMPLER)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ── Scenarios ────────────────────────────────────────────────────────────
# (The control channel is owned by `python -m hud.environment.server`; the
# robot bridge binds an ephemeral loopback port published in the manifest.)
# Which task to run is the runner's choice (see run.py), not env config.

# Every SimplerEnv WidowX/Bridge task we expose as a scenario (one scenario per
# entry). These env ids are registered with gymnasium by importing
# ``mani_skill.envs.tasks.digital_twins.bridge_dataset_eval`` (see the bridge).
TASKS = [
    "PutCarrotOnPlateInScene-v1",
    "PutSpoonOnTableClothInScene-v1",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    "PutEggplantInBasketScene-v1",
]

# Number of deterministic episode_id variants per task, i.e.
# ``len(xyz_configs) * len(quat_configs)`` from each task's definition in
# ``mani_skill.../bridge_dataset_eval/put_on_in_scene.py``. Sweeping episode_id
# over ``range(TASK_VARIANTS[task])`` visits every object pose/orientation combo
# exactly once (the runner uses this to size the benchmark sweep, mirroring how
# ``run.py`` derives the LIBERO task/init-state grid).
TASK_VARIANTS = {
    "PutCarrotOnPlateInScene-v1": 24,                    # 12 xyz x 2 quat
    "PutSpoonOnTableClothInScene-v1": 24,               # 12 xyz x 2 quat
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1": 24,  # 24 xyz x 1 quat
    "PutEggplantInBasketScene-v1": 64,                  # 8 xyz x 8 quat
}

STREAM_PORT = int(os.environ.get("BENCH_STREAM_PORT", "8080"))  # MJPEG live-view HTTP port
STREAM_FPS = 10                      # live-view frame rate

# ── Operational knobs (deliberately NOT in the contract) ─────────────────
# The contract describes the robot's feature spec (what/how); how we drive
# ManiSkill (which camera to read, which obs_mode to request) is the env's job.
CAMERA = "3rd_view_camera"           # WidowX/Bridge has one external view (no wrist cam)
OBS_MODE = "rgb+segmentation"        # which sensor tensors gym.make should return

# ── Contract config (single source of truth) ────────────────────────────
#
# One complete wire contract per control mode (spec_v0 §5: one action space per
# contract). The launch-time decision variable (``use_delta``) selects a FILE.
_DIR = Path(__file__).resolve().parent
CONTRACT_PATHS: dict[str, Path] = {
    "ee_del": _DIR / "simpler_ee_del.json",
    "ee_abs": _DIR / "simpler_ee_abs.json",
}
DEFAULT_MODE = "ee_del"


def load_contract(mode: str = DEFAULT_MODE) -> dict:
    """Load the complete wire contract for one control mode."""
    return json.loads(CONTRACT_PATHS[mode].read_text())


# ── Decision variables ──────────────────────────────────────────────────
#
# Same shape as ``environment/config.py``: the env exposes a
# launch-time decision variable that changes its behavior and the robot it
# presents on the wire. ``use_delta`` picks the WidowX control regime:
#   True  -> native delta EE control  (ee_del action)
#   False -> absolute base-pose EE control (ee_abs action), the regime the
#            X-VLA authors used for Simpler-WidowX.
# Only the action varies; the ee_abs state proprio is fixed (as in LIBERO). The
# bridge flips the matching ManiSkill control mode (see SimplerSimBridge).
DEFAULT_PARAMS: dict = {"use_delta": True}


def build_contract(params: dict | None = None) -> dict:
    """Resolve the env's decision variables (``params``) into one wire contract."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    return load_contract("ee_del" if p["use_delta"] else "ee_abs")


CONTRACT = load_contract()
# Image resolution the contract advertises (H, W), for cross-checking the env.
IMAGE_SIZE = tuple(CONTRACT["features"]["observation/image"]["shape"][:2])

__all__ = [
    "TASKS",
    "TASK_VARIANTS",
    "STREAM_PORT",
    "STREAM_FPS",
    "CAMERA",
    "OBS_MODE",
    "IMAGE_SIZE",
    "CONTRACT_PATHS",
    "DEFAULT_MODE",
    "CONTRACT",
    "DEFAULT_PARAMS",
    "load_contract",
    "build_contract",
]
