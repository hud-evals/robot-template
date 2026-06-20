"""SCAFFOLD — serve your policy as an OpenPI-WebSocket server. Fill the TODOs.

Spin your model's weights up on a GPU box (Modal here) and expose them over the
openpi/0 wire protocol. Your local harness then drives it with a weightless
`RemoteModel(host, port)` (see `my_remote_agent.py`). Working example:
`inventory/agents/remote/smolvla/smolvla_serve.py`.

Already have an openpi server (stock `openpi serve_policy.py`, Cosmos, pi0.5)?
Skip this file — just point `RemoteModel` at it.

    modal run inventory/agents/remote/scaffold/my_policy_serve.py
"""

from __future__ import annotations

import modal

PORT = 8000

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install("openpi-client", "torch")
    # TODO: add your policy's deps (e.g. "lerobot @ git+...", "transformers", ...)
)

app = modal.App("my-policy-serve")


@app.function(image=IMAGE, gpu="L4", timeout=24 * 3600, scaledown_window=900)
def serve() -> None:
    import asyncio

    import numpy as np
    import websockets.asyncio.server as wss
    import websockets.exceptions
    from openpi_client import msgpack_numpy

    policy = ...  # noqa: F841  TODO: load your checkpoint onto "cuda" here, once (use it in infer)

    def infer(obs: dict) -> dict:
        # `obs` holds the env's observation under the openpi wire keys + "prompt"
        # (exactly what OpenPIAdapter sends). Build your policy input and run it.
        chunk = ...  # TODO: run your policy -> a [T, A] action chunk (env action space)
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
        asyncio.run(_ws())


@app.local_entrypoint()
def main() -> None:
    serve.remote()
