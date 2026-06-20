"""Verify your environment is ready to run the template. `python test_install.py`.

Checks the SDK + robot extra (required), then the policy/runtime stack (needed for
the examples). Exits 0 if the required core is present, 1 otherwise. Optional pieces
print a hint instead of failing.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import sys

OK, WARN, BAD = "  ok ", " warn", " MISS"


def _check(label: str, ok: bool, *, hint: str = "", required: bool = True) -> bool:
    tag = OK if ok else (BAD if required else WARN)
    line = f"[{tag}] {label}"
    if not ok and hint:
        line += f"\n         -> {hint}"
    print(line)
    return ok or not required


def _module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def main() -> int:
    print("=== HUD robot template — install check ===\n")
    required_ok = True

    # ── Core SDK + robot extra (required) ────────────────────────────────────
    print("core SDK (required):")
    _git = "pip install 'hud-python[robot] @ git+https://github.com/hud-evals/hud-python.git'"
    required_ok &= _check(
        "hud-python", _module("hud"),
        hint=_git,
    )
    required_ok &= _check(
        "robot extra (numpy, openpi-client)", _module("numpy") and _module("openpi_client"),
        hint=_git,
    )

    # The actual robot symbols the template imports everywhere (submodule paths,
    # matching the example files).
    try:
        from hud.agents.robot.adapter import LeRobotAdapter, OpenPIAdapter  # noqa: F401
        from hud.agents.robot.agent import RobotAgent  # noqa: F401
        from hud.agents.robot.batching import BatchedAgent  # noqa: F401
        from hud.agents.robot.model import LeRobotModel, Model, RemoteModel  # noqa: F401
        from hud.environment.robot import RobotBridge, RobotEndpoint  # noqa: F401
        from hud.eval import DockerRuntime, Task, Taskset  # noqa: F401
        symbols_ok = True
    except Exception as exc:  # noqa: BLE001
        symbols_ok = False
        print(f"         (import error: {exc})")
    required_ok &= _check(
        "robot harness symbols (RobotAgent, LeRobotModel, RemoteModel, ...)", symbols_ok,
        hint="the v6 robot API isn't on PyPI yet — reinstall from git:\n"
             f"            {_git}  --force-reinstall",
    )

    # ── Policy + runtime stack (needed for the examples) ─────────────────────
    print("\npolicy + runtime (for the examples):")
    have_torch = _module("torch")
    _check("torch", have_torch, hint="pip install torch", required=False)
    if have_torch:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _check(f"torch device = {dev}", True, required=False)
        if dev == "cpu":
            print("         -> no GPU detected: local policy inference will be slow; "
                  "prefer the remote example (examples/remote.py).")
    _check("lerobot + smolvla extra (transformers)", _module("lerobot") and _module("transformers"),
           hint="pip install 'lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git'",
           required=False)
    _check("docker CLI", shutil.which("docker") is not None,
           hint="install Docker to run the libero / libero_batching examples", required=False)
    _check("modal", _module("modal"),
           hint="pip install modal && modal token new  (for remote model/env examples)",
           required=False)

    print()
    if required_ok:
        print("CORE OK — you're ready. Next: see QUICKSTART.md")
        return 0
    print("CORE MISSING — install the required pieces above, then re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
