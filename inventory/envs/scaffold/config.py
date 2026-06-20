"""SCAFFOLD config — exposed tasks + the env contract (single source of truth).

The env advertises one `contract` (see `example_contract.json`) in its capability
manifest; the agent reads it back and wires itself with no shared config. Mirror
`environment/config.py` for the real thing (per-mode contract files,
launch-time decision variables). Docs: https://docs.hud.ai/v6/core/robots#the-contract
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_DIR = Path(__file__).resolve().parent

# TODO: your task ids (one env.template per entry; see env.py).
TASKS = ["task_a", "task_b"]

STREAM_PORT = int(os.environ.get("BENCH_STREAM_PORT", "8080"))  # optional MJPEG live view
STREAM_FPS = 10


def load_contract() -> dict:
    return json.loads((_DIR / "example_contract.json").read_text())


CONTRACT = load_contract()
# Camera resolution to request from your sim, read off the contract.
IMAGE_SIZE = CONTRACT["features"]["observation/image"]["shape"][0]

__all__ = ["CONTRACT", "IMAGE_SIZE", "STREAM_FPS", "STREAM_PORT", "TASKS", "load_contract"]
