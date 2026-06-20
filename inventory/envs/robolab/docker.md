# Build the RoboLab (Isaac Lab / DROID) image

RoboLab is the **main-thread sim** reference: Isaac/Omniverse must own the process
main thread, so the sim runs in its own `sim_process.py` (owning the robot bridge)
and the declarative `env.py` drives it over `RobotEndpoint.remote(...)`. Study this
pair when your simulator pins the main thread; see also the
[docs](https://docs.hud.ai/v6/core/robots#running-a-sim-in-another-process).

**GPU required** — Isaac Sim renders with RTX (no CPU fallback). Unlike LIBERO /
SimplerEnv, the image is **not** self-contained: it stacks the SDK on top of
RoboLab's own Isaac Lab base image, so you need a RoboLab checkout to build it.

| port | what |
|------|------|
| 8765 | HUD control channel — the only required port |
| 8080 | MJPEG live view — optional debugging |

## Dockerfile (two-stage)

Stage 1 is RoboLab's own image (Isaac Lab base + assets + the robolab package);
stage 2 stacks the SDK + this template's `inventory/` on top:

```bash
# 1) RoboLab's base image (context = your RoboLab checkout; pulls Isaac Lab ~20 GB)
docker build -f <ROBOLAB>/docker/Dockerfile -t robolab:base <ROBOLAB>
# 2) this env image (context = the template root)
docker build -t hud-robolab-env .
```

```dockerfile
FROM robolab:base

# Isaac's kit interpreter is the ONLY python that can import isaaclab + the kit
# runtime, so every install/serve below goes through it (not the system python).
ENV ISAAC_PY=/workspace/isaaclab/_isaac_sim/python.sh

# Installing into kit python is delicate: it dedupes shared packages as symlink
# farms, and a naive upgrade deletes the files those symlinks point at. So:
#  1. shadow-install (--ignore-installed) the few packages hud needs newer;
#  2. freeze everything else to a constraints file so hud can ADD but never upgrade.
# robolab:base already ships an openpi-client (the openpi/0 codec), so install
# plain hud-python (not the [robot] extra) and reuse the base's codec.
RUN ${ISAAC_PY} -m pip install --no-cache-dir --ignore-installed \
        "packaging>=24.0" "pydantic>=2.11.7" "pyperclip>=1.9.0" "uvicorn>=0.35" "websockets>=15.0.1" \
    && ${ISAAC_PY} -m pip list --format=freeze --exclude-editable > /tmp/kit-constraints.txt \
    && ${ISAAC_PY} -m pip install --no-cache-dir -c /tmp/kit-constraints.txt \
        "hud-python @ git+https://github.com/hud-evals/hud-python.git" \
    && ${ISAAC_PY} -c "from openpi_client import msgpack_numpy; print('openpi/0 codec OK')"

COPY inventory /app/inventory

ENV OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y PYTHONUNBUFFERED=1 \
    ROBOLAB_SIM_AUTOSPAWN=1 \
    ROBOLAB_SIM_PYTHON=/workspace/isaaclab/_isaac_sim/python.sh

EXPOSE 8765 8080
WORKDIR /app/inventory/envs/robolab
# robolab:base's entrypoint boots a streaming kit; clear it — sim_process.py owns the boot.
ENTRYPOINT []
CMD ["/workspace/isaaclab/_isaac_sim/python.sh", "-m", "hud.environment.server", \
     "env.py", "--host", "0.0.0.0", "--port", "8765"]
```

## Run

```bash
docker run -d --name robolab-env --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
  --publish 127.0.0.1::8765 hud-robolab-env
```

Boot takes ~2 min (Omniverse). `env.py` autospawns `sim_process.py` itself
(`ROBOLAB_SIM_AUTOSPAWN=1`). Because each fresh container pays the boot cost,
prefer one long-lived container for sweeps and attach to it:

```python
from hud.eval import Task, Taskset
from hud.eval.runtime import Runtime

tasks = [Task(env="robolab", id="robolab", args={"task_name": "RubiksCubeTask", "seed": s})
         for s in range(3)]
job = await Taskset("robolab", tasks).run(agent, runtime=Runtime("tcp://127.0.0.1:8765"))
```

`-e BENCH_CONTROL=ee_abs` (or `ee_del`) switches the action mode. Dataset recording
is not supported in this image (lerobot's deps fight Isaac's); platform telemetry works.

## Running the sim by hand (no Docker)

In an interpreter that can import Isaac:

```bash
python sim_process.py                                   # boots Omniverse + the bridge
python -m hud.environment.server env.py --port 9001     # connects to it
```
