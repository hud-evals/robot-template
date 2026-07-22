# robot-template

SmolVLA driving the [LIBERO](https://libero-project.github.io/) benchmark over HUD's
`robot` capability. `environment/env.py` declares the env, `environment/lerobot_sim.py`
is the module-level sim factory (task is a build arg), and `run.py` is the agent and runner.

```bash
pip install 'hud[robot]' 'lerobot[smolvla,libero]'
MUJOCO_GL=egl python run.py
```

Edit `SUITE` / `N_TASKS` in `run.py` to pick the suite and task count. With
`HUD_API_KEY` set, rollouts stream to the trace viewer on [hud.ai](https://hud.ai).

## Docker (optional)

Run the sim in a container instead of installing LIBERO locally — build the image
from `Dockerfile.hud` and point the runner at a `DockerRuntime`:

```bash
docker build -f Dockerfile.hud -t hud-libero-env .
```

```python
from hud.eval import DockerRuntime
runtime = DockerRuntime("hud-libero-env")  # in place of LocalRuntime in run.py
```

`hud deploy` hosts the env on the platform instead of local Docker.

Docs: [docs.hud.ai/v6/advanced/robots](https://docs.hud.ai/v6/advanced/robots)
