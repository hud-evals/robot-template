"""Module-level ``make_env`` for ``env.gym`` — task selection is a build arg.

``GymBridge`` rebuilds when factory kwargs (``task_suite`` / ``task_id``)
change; ``seed`` stays episodic. Observations are repacked to the flat
two-camera + 8-D state layout smolvla_libero trains on — the same packing
lerobot's ``LiberoProcessorStep`` applies at train time — so the stock
``LeRobotAdapter`` drives the policy with no custom adapter.
"""

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")  # headless render


def ensure_libero_config() -> None:
    """Pre-write LIBERO's config; hf-libero prompts interactively when it is missing."""
    import libero
    import yaml

    cfg = Path.home() / ".libero" / "config.yaml"
    if cfg.exists():
        return
    root = Path(libero.__file__).parent / "libero"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    paths = {"benchmark_root": root, "bddl_files": root / "bddl_files",
             "init_states": root / "init_files", "datasets": root.parent / "datasets",
             "assets": root / "assets"}
    cfg.write_text(yaml.safe_dump({k: str(v) for k, v in paths.items()}))


def pack_observation(obs: dict) -> dict:
    """LIBERO's nested obs → flat contract keys (two cameras + 8-D state)."""
    rs = obs["robot_state"]
    # eef quaternion (xyzw) → axis-angle, as in lerobot's LiberoProcessorStep.
    quat = np.asarray(rs["eef"]["quat"], dtype=np.float32)
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = float(np.sqrt(max(1.0 - w * w, 0.0)))
    axisangle = quat[:3] * (2.0 * np.arccos(w) / den) if den > 1e-10 else np.zeros(3)
    state = np.concatenate([rs["eef"]["pos"], axisangle, rs["gripper"]["qpos"]])
    # Cameras flipped 180°: the HuggingFaceVLA/libero orientation convention.
    return {
        "observation/image": obs["pixels"]["image"][::-1, ::-1].copy(),
        "observation/wrist_image": obs["pixels"]["image2"][::-1, ::-1].copy(),
        "observation/state": state.astype(np.float32),
    }


def make_env(task_suite: str = "libero_spatial", task_id: int = 0):
    """Build one LIBERO task env; ``GymBridge`` rebuilds when suite/id change."""
    ensure_libero_config()
    from gymnasium.wrappers import TransformObservation
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env as lerobot_make_env

    # Unwrap the vec-of-one to the plain gym.Env; 256 = smolvla_libero's camera res.
    env = lerobot_make_env(
        LiberoEnv(task=task_suite, task_ids=[task_id],
                  observation_height=256, observation_width=256),
        n_envs=1,
    )[task_suite][task_id].envs[0]
    return TransformObservation(env, pack_observation, None)
