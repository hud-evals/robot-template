"""A lazy gym.Env over lerobot's ``make_env``: task selection is episodic.

Builds on first reset and rebuilds only when (suite, task_id) changes, so one
declarative env serves the whole benchmark — the task travels as reset args,
not as construction args.
"""

import os
from pathlib import Path

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


class LeRobotSim:
    """Gym-shaped adapter; ``reset(options={"task_suite", "task_id"})`` picks the task."""

    def __init__(self, suite: str = "libero_spatial", task_id: int = 0):
        self._default = (suite, task_id)
        self._key: tuple[str, int] | None = None
        self._env = None
        self.task_description = ""

    def _ensure(self, suite: str, task_id: int) -> None:
        if (suite, task_id) == self._key:
            return
        if self._env is not None:
            self._env.close()

        ensure_libero_config()
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env

        # Unwrap the vec-of-one to the plain gym.Env (task_description, is_success).
        self._env = make_env(LiberoEnv(task=suite, task_ids=[task_id]), n_envs=1)[suite][task_id].envs[0]
        self._key = (suite, task_id)
        self.metadata = getattr(self._env, "metadata", {"render_fps": 10})
        self.task_description = getattr(self._env, "task_description", "")

    def reset(self, seed=None, options=None):
        opts = dict(options or {})
        suite = opts.pop("task_suite", self._default[0])
        task_id = int(opts.pop("task_id", self._default[1]))
        self._ensure(suite, task_id)
        return self._env.reset(seed=seed)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    @property
    def action_space(self):
        return self._env.action_space
