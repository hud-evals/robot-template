# Build the SimplerEnv (WidowX/Bridge on ManiSkill3) image

This template ships the env code but not a `Dockerfile` — recreate it from the
recipe below (self-contained: SDK from PyPI, only this template's `inventory/` is
copied in). Write it to `Dockerfile` at the **template root**.

Renders with Vulkan (SAPIEN) via Mesa **lavapipe** (software ICD), so it runs on
a CPU-only host out of the box — and fast (episodes finish in seconds).

| port | what |
|------|------|
| 8765 | HUD control channel — the only required port |
| 8080 | MJPEG live view (`http://localhost:8080/`) — optional debugging |

## Dockerfile

```dockerfile
FROM python:3.12

# vulkan + mesa-vulkan-drivers (lavapipe): SAPIEN software rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg build-essential libvulkan1 mesa-vulkan-drivers libgl1 libglib2.0-0 \
        libx11-6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA Vulkan ICD manifest (only used with --gpus all; harmless otherwise).
RUN mkdir -p /usr/share/vulkan/icd.d && printf '%s\n' \
    '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}' \
    > /usr/share/vulkan/icd.d/nvidia_icd.json

# CPU-only torch (policy runs agent-side; this image only needs the import chain).
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    "lerobot[dataset] @ git+https://github.com/huggingface/lerobot.git"

# ManiSkill3 sim stack.
RUN pip install --no-cache-dir \
    mani_skill==3.0.1 sapien==3.0.3 gymnasium==1.3.0 \
    msgpack websockets aiohttp requests pyyaml pillow "imageio[ffmpeg]" \
    opencv-python-headless

# WidowX + Bridge real2sim assets (~350 MB), baked in so nothing downloads at runtime.
RUN python -m mani_skill.utils.download_asset bridge_v2_real2sim -y \
    && python -m mani_skill.utils.download_asset widowx250s -y

# From git so the env matches the agent's harness; use the plain
# "hud-python[robot]" PyPI spec once the v6 robot release lands.
RUN pip install --no-cache-dir "hud-python[robot] @ git+https://github.com/hud-evals/hud-python.git" \
    && pip install --no-cache-dir "numpy==2.2.6"

COPY inventory /app/inventory
ENV PYTHONUNBUFFERED=1
EXPOSE 8765 8080
WORKDIR /app/inventory/envs/simpler
CMD ["hud", "serve", "env.py", "--host", "0.0.0.0", "--port", "8765"]
```

## Build & run

From the **template root**:

```bash
docker build -t hud-simpler-env .
```

```python
from hud.eval import DockerRuntime, Taskset
job = await Taskset("simpler", tasks).run(agent, runtime=DockerRuntime("hud-simpler-env"))
```

`-e BENCH_CONTROL=absolute` flips to absolute base-pose EE control (the X-VLA regime).
On a GPU host: `docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all ...`.
