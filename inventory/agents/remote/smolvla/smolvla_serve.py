"""Serve SmolVLA as an OpenPI-WebSocket policy server on Modal (GPU).

This is the "spin up a policy server" half of the remote story: the model's
weights live on a Modal GPU, and your local harness reaches it with a weightless
`RemoteModel(host, port)` (see `smolvla_libero_remote.py`). The wire is the
**openpi/0** protocol, so the same server works for any openpi client — and,
conversely, your `RemoteModel` can point at *any* openpi server (a stock
`openpi serve_policy.py`, NVIDIA Cosmos, a hosted pi0.5, ...).

Run (prints `ws://HOST:PORT` — set those as POLICY_HOST / POLICY_PORT locally):

    modal run inventory/agents/remote/smolvla/smolvla_serve.py

Requires `modal token new` (one-time). The image installs everything from PyPI,
so nothing local is mounted.
"""

from __future__ import annotations

import modal

CHECKPOINT = "lerobot/smolvla_libero"
PORT = 8000

# Maps the openpi wire keys the env sends -> the policy's own image-slot names.
IMAGE_KEYS = {
    "observation.images.image": "observation/image",
    "observation.images.wrist_image": "observation/wrist_image",
}

# Self-contained GPU image: the SDK (for LeRobotModel), lerobot (the policy), and
# openpi-client (the openpi/0 wire codec). All from PyPI — nothing local mounted.
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    # Separate pip layers: resolving two git+ direct-URL packages in one command
    # trips pip's resolver. lerobot pulls its own pinned torch; the smolvla extra
    # adds transformers + num2words.
    .pip_install("lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git")
    .pip_install("hud-python[robot] @ git+https://github.com/hud-evals/hud-python.git", "openpi-client")
)

app = modal.App("hud-smolvla-serve")


@app.function(image=IMAGE, gpu="L4", timeout=24 * 3600, scaledown_window=900)
def serve() -> None:
    import asyncio

    import numpy as np
    import torch
    import websockets.asyncio.server as wss
    import websockets.exceptions
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from openpi_client import msgpack_numpy

    from hud.agents.robot.model import LeRobotModel

    device = "cuda"
    policy = SmolVLAPolicy.from_pretrained(CHECKPOINT).to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config, CHECKPOINT, preprocessor_overrides={"device_processor": {"device": device}}
    )
    model = LeRobotModel(policy, pre, post)

    def infer(obs: dict) -> dict:
        # Build the policy batch from the openpi obs dict the client sent.
        batch = {
            "observation.state": torch.from_numpy(
                np.asarray(obs["observation/state"], dtype=np.float32)
            ),
            "task": obs["prompt"],
        }
        for model_key, wire_key in IMAGE_KEYS.items():
            batch[model_key] = torch.from_numpy(obs[wire_key]).permute(2, 0, 1).float() / 255.0
        chunk = model.infer(batch)[0]  # [N, T, A] -> this client's [T, A]
        return {"actions": np.asarray(chunk, dtype=np.float32)}

    async def _ws() -> None:
        packer = msgpack_numpy.Packer()

        async def handler(ws):
            await ws.send(packer.pack({}))  # openpi handshake: server metadata first
            try:
                while True:
                    obs = msgpack_numpy.unpackb(await ws.recv())
                    await ws.send(packer.pack(infer(obs)))
            except websockets.exceptions.ConnectionClosed:
                pass

        async with wss.serve(handler, "0.0.0.0", PORT, compression=None, max_size=None) as server:
            await server.serve_forever()

    with modal.forward(PORT, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"[serve] policy server live: ws://{host}:{port}", flush=True)
        print(f"[serve] set:  export POLICY_HOST={host} POLICY_PORT={port}", flush=True)
        asyncio.run(_ws())


@app.local_entrypoint()
def main() -> None:
    serve.remote()
