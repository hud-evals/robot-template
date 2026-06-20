"""Env configuration: bind address, exposed scenarios, and the env contract.

The env advertises a single source-of-truth *contract* in its capability
manifest (see ``env.py``): one of the local per-mode contracts
(``libero_ee_del.json`` / ``libero_ee_abs.json``), carrying the robot
type, control rate, and every action/observation feature with its layout and
normalization stats. The agent reads the contract back from the manifest and
splits it into action/observation spaces via ``RobotClient.spaces()``, so it
wires observations -> policy inputs without any shared config (see the agents in
``inventory/agents/``).

Contract feature keys are authoritative on the wire: the bridge emits its
observations under the contract's feature names — the OpenPI LIBERO keys
(``observation/image``, ``observation/wrist_image``, ``observation/state``) — so
the env and the agent agree on naming purely via the manifest.

Each feature carries:
- ``role``   : "observation" | "action"   (direction)
- ``type``   : "rgb" | "ee_abs" | "ee_del" | "joint_pos" | ...  (representation)
- ``dtype`` / ``shape`` / ``names`` : layout
- ``stats``  : per-field normalization (mean/std/min/max)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ── Scenarios + debug stream ─────────────────────────────────────────────
# (The control channel is owned by `python -m hud.environment.server`; the
# robot bridge binds an ephemeral loopback port published in the manifest —
# neither needs configuration here.)

# Every LIBERO task suite we expose as a scenario (one scenario per entry).
TASK_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]

STREAM_PORT = int(os.environ.get("BENCH_STREAM_PORT", "8080"))  # MJPEG live-view HTTP port
STREAM_FPS = 10                      # live-view frame rate

# ── Contract config (single source of truth) ────────────────────────────
#
# One complete wire contract per control mode (spec_v0 §5: one action space per
# contract). The env's launch-time decision variable (``use_delta``) selects a
# FILE — there is no action splicing.
_DIR = Path(__file__).resolve().parent
CONTRACT_PATHS: dict[str, Path] = {
    "ee_del": _DIR / "libero_ee_del.json",
    "ee_abs": _DIR / "libero_ee_abs.json",
}
DEFAULT_MODE = "ee_del"


def load_contract(mode: str = DEFAULT_MODE) -> dict:
    """Load the complete wire contract for one control mode."""
    return json.loads(CONTRACT_PATHS[mode].read_text())


# ── Decision variables ──────────────────────────────────────────────────
#
# The env exposes launch-time "decision variables" (params) that fundamentally
# change its behavior *and* the robot it presents on the wire. ``use_delta``
# (True -> ee_del delta action, False -> ee_abs absolute-pose action) selects
# the contract file and flips the sim controller (see ``LiberoSimBridge``).
# The agent pairs its behavior to the env purely from the contract's single
# ``features.action`` feature (e.g. its ``type``), so the decision variable
# propagates end-to-end through the manifest.
DEFAULT_PARAMS: dict = {"use_delta": True}


def build_contract(params: dict | None = None) -> dict:
    """Resolve the env's decision variables (``params``) into one wire contract."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    return load_contract("ee_del" if p["use_delta"] else "ee_abs")


CONTRACT = load_contract()
# Camera resolution requested from the sim, taken from the contract image feature.
IMAGE_SIZE = CONTRACT["features"]["observation/image"]["shape"][0]

__all__ = [
    "TASK_SUITES",
    "STREAM_PORT",
    "STREAM_FPS",
    "IMAGE_SIZE",
    "CONTRACT_PATHS",
    "DEFAULT_MODE",
    "CONTRACT",
    "DEFAULT_PARAMS",
    "load_contract",
    "build_contract",
]
