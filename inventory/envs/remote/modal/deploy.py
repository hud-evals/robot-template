"""One-time: build the LIBERO env image on Modal and publish it by name.

Runs the root `Dockerfile.hud` on Modal and tags the result `hud-libero-env`, so
rollouts boot a fresh sandbox per episode without rebuilding. Build context is the
template root (the Dockerfile COPYs `environment/` + `pyproject.toml`).

    modal run inventory/envs/remote/modal/deploy.py

Requires `modal token new` (one-time). ~15-30 min on first build.
"""

from __future__ import annotations

from pathlib import Path

import modal

IMAGE_NAME = "hud-libero-env"  # ModalRuntime("hud-libero-env") resolves this
APP_NAME = "hud-envs"
TEMPLATE_ROOT = Path(__file__).resolve().parents[4]  # .../remote/modal -> template root
DOCKERFILE = TEMPLATE_ROOT / "Dockerfile.hud"

image = modal.Image.from_dockerfile(DOCKERFILE, context_dir=TEMPLATE_ROOT)
app = modal.App(APP_NAME)


@app.local_entrypoint()
def main() -> None:
    if not DOCKERFILE.exists():
        raise SystemExit(f"No Dockerfile.hud at {DOCKERFILE}.")
    sb_app = modal.App.lookup(APP_NAME, create_if_missing=True)
    image.build(app=sb_app)
    image.publish(IMAGE_NAME)
    print(f"published image: {IMAGE_NAME}  ->  ModalRuntime({IMAGE_NAME!r})")
