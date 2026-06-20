"""One-time: build the LIBERO env snapshot on Daytona and register it by name.

Builds the root `Dockerfile.hud` into a Daytona snapshot named `hud-libero-env`, so
rollouts boot a fresh sandbox per episode without rebuilding. The Dockerfile COPYs
`environment/` + `pyproject.toml`, so the build context is the template root.

    python inventory/envs/remote/daytona/deploy.py

Requires the Daytona CLI (`daytona`) on PATH and `DAYTONA_API_KEY`. ~15-30 min.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SNAPSHOT_NAME = "hud-libero-env"  # DaytonaRuntime("hud-libero-env") resolves this
TEMPLATE_ROOT = Path(__file__).resolve().parents[4]  # .../remote/daytona -> template root
DOCKERFILE = TEMPLATE_ROOT / "Dockerfile.hud"
REGION = os.environ.get("DAYTONA_REGION", "us")
# Daytona's per-sandbox ceiling; LIBERO's CPU software-rendered sim wants the headroom.
CPU, MEMORY_GB, DISK_GB = 4, 8, 10


def main() -> None:
    if not DOCKERFILE.exists():
        raise SystemExit(f"No Dockerfile.hud at {DOCKERFILE}.")
    cmd = [
        "daytona", "snapshot", "create", SNAPSHOT_NAME,
        "--dockerfile", str(DOCKERFILE),
        "--context", str(TEMPLATE_ROOT),
        "--cpu", str(CPU), "--memory", str(MEMORY_GB), "--disk", str(DISK_GB),
        "--region", REGION,
    ]
    print("building snapshot (uploads context + runs the Dockerfile; ~15-30 min)...")
    subprocess.run(cmd, check=True)
    print(f"registered snapshot: {SNAPSHOT_NAME}  ->  DaytonaRuntime({SNAPSHOT_NAME!r})")


if __name__ == "__main__":
    main()
