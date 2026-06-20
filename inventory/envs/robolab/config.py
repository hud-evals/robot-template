"""Env configuration: bind address, exposed task, and the env contract.

Mirrors ``environment/config.py``. The env builds the contract here
(env-side) and republishes it in its capability manifest; one of the local
per-mode contracts (``robolab_joint_pos.json`` / ``robolab_ee_abs.json`` /
``robolab_ee_del.json``), carrying the robot type, control rate, and every
action/observation feature with its layout and stats. The agent reads the
contract back from the manifest and splits it into action/observation spaces
via ``RobotClient.spaces()``, so it wires observations -> policy inputs without
any shared config (see the agents in ``inventory/agents/``).

Contract feature keys are authoritative on the wire: the bridge emits its
observations under the contract's feature names (``wrist_image``, ``state``), so
the env and the agent agree on naming purely via the manifest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ── Live view ─────────────────────────────────────────────────────────────
# (The env control channel port is `hud.environment.server`'s --port flag; the
# robot bridge binds an ephemeral loopback port in the sim process, published in
# the manifest.) Which RoboLab task to run is the harness's choice (run_robolab.py).

STREAM_PORT = int(os.environ.get("BENCH_STREAM_PORT", "8080"))  # MJPEG live-view HTTP port
STREAM_FPS = 10                      # live-view frame rate

DEVICE = "cuda:0"                    # Isaac Lab sim device

# Episode endpoint (JSON-RPC link between the env process and the sim process):
# the sim serves it, the env dials it. Loopback by convention (both on one host).
ENDPOINT_HOST = os.environ.get("ROBOLAB_ENDPOINT_HOST", "127.0.0.1")
ENDPOINT_PORT = int(os.environ.get("ROBOLAB_ENDPOINT_PORT", "9100"))

# Launch-time control mode (decision variable); selects the contract + action cfg.
CONTROL = os.environ.get("BENCH_CONTROL", "joint_pos")

# ── Default benchmark sweep ──────────────────────────────────────────────
#
# RoboLab ships 120 tasks (under ``tasks/benchmark/``); the env validates *any*
# task name dynamically against ``get_envs()`` at reset, so the harness can call
# any of them by name (``--task-names ...``). This curated subset is just a quick
# default sweep — two simple pick-and-place benchmark tasks for a sanity check.
SANITY_TASKS = ["RubiksCubeTask", "YogurtInBowlTask"]

# Instruction variants to sweep as a parameter axis (RoboLab tasks define
# "default"/"vague"/"specific"). Default sweep uses just "default".
DEFAULT_INSTRUCTION_TYPES = ["default"]

# ── Contract config (single source of truth) ─────────────────────────────
#
# One complete wire contract per control mode (spec_v0 §5: one action space per
# contract). The launch-time decision variable (``control``) selects a FILE.
_DIR = Path(__file__).resolve().parent
CONTRACT_PATHS: dict[str, Path] = {
    "joint_pos": _DIR / "robolab_joint_pos.json",
    "ee_abs": _DIR / "robolab_ee_abs.json",
    "ee_del": _DIR / "robolab_ee_del.json",
}
DEFAULT_MODE = "joint_pos"


def load_contract(mode: str = DEFAULT_MODE) -> dict:
    """Load the complete wire contract for one control mode."""
    return json.loads(CONTRACT_PATHS[mode].read_text())


# ── Decision variables ───────────────────────────────────────────────────
#
# The env exposes one launch-time "decision variable" (``control``) that selects
# both the RoboLab action registration and the contract file advertised on the
# wire:
#   joint_pos (default, native) -> DroidJointPositionActionCfg
#   ee_abs                      -> DroidIKActionCfg     (differential IK, absolute)
#   ee_del                      -> DroidRelIKActionCfg  (differential IK, relative)
DEFAULT_PARAMS: dict = {"control": "joint_pos"}


def build_contract(params: dict | None = None) -> dict:
    """Resolve the env's decision variable (``params``) into one wire contract."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    return load_contract(p["control"])


CONTRACT = load_contract()
# Camera resolution requested from the sim, taken from the contract image feature.
IMAGE_SIZE = CONTRACT["features"]["observation/wrist_image_left"]["shape"][0]

__all__ = [
    "STREAM_PORT",
    "STREAM_FPS",
    "DEVICE",
    "ENDPOINT_HOST",
    "ENDPOINT_PORT",
    "CONTROL",
    "SANITY_TASKS",
    "DEFAULT_INSTRUCTION_TYPES",
    "IMAGE_SIZE",
    "CONTRACT_PATHS",
    "DEFAULT_MODE",
    "CONTRACT",
    "DEFAULT_PARAMS",
    "load_contract",
    "build_contract",
]
