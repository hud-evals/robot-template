"""Unified live-view streaming for the benchmark envs.

- :class:`MJPEGStreamer` — a tiny MJPEG-over-HTTP frame sink. Push H x W x 3
  uint8 RGB frames with :meth:`add_stream_frame`, then point a browser at the
  served URL to watch live (``multipart/x-mixed-replace``, no client app
  needed). Runs in the host's asyncio loop and is deliberately env-agnostic.
- :func:`stitch` — compose one or more camera views into a single debug frame
  (side-by-side, optionally resized to a common height and labeled), so each
  sim bridge stops re-implementing its own frame layout.

Anything that produces uint8 RGB frames can stream itself for debugging.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Sequence

import numpy as np
from aiohttp import web
from PIL import Image

_PAGE = (
    '<!doctype html><html><body style="margin:0;background:#111">'
    '<img src="/stream" style="width:100vw;height:100vh;object-fit:contain">'
    "</body></html>"
)


def stitch(
    views: Sequence[np.ndarray | None],
    *,
    labels: Sequence[str | None] | None = None,
    height: int | None = None,
) -> np.ndarray | None:
    """Lay RGB uint8 camera views side-by-side into one debug frame.

    - ``views``: frames to compose; ``None`` entries are dropped (a missing cam
      never kills the stream).
    - ``labels``: optional per-view caption drawn top-left (aligned with
      ``views``); any non-empty label triggers the ``cv2`` text path.
    - ``height``: optional common panel height; every view is resized to it
      before concatenation. When omitted and all views already share a height,
      they are concatenated directly (no ``cv2`` needed).

    Returns the composite ``H x W x 3`` uint8 frame, or ``None`` if no view is
    available. ``cv2`` is imported lazily, so the plain equal-height path has no
    OpenCV dependency.
    """
    pairs: list[tuple[np.ndarray, str | None]] = []
    captions = list(labels) if labels is not None else [None] * len(views)
    for view, label in zip(views, captions, strict=False):
        if view is not None:
            pairs.append((np.ascontiguousarray(view, dtype=np.uint8), label))
    if not pairs:
        return None

    needs_cv2 = height is not None or any(label for _, label in pairs)
    if not needs_cv2 and len({v.shape[0] for v, _ in pairs}) == 1:
        return np.ascontiguousarray(np.hstack([v for v, _ in pairs]), dtype=np.uint8)

    import cv2

    target_h = height or max(v.shape[0] for v, _ in pairs)
    tiles: list[np.ndarray] = []
    for view, label in pairs:
        ih, iw = view.shape[:2]
        if ih != target_h:
            tile = cv2.resize(
                view, (max(1, round(iw * target_h / ih)), target_h), interpolation=cv2.INTER_AREA
            )
        else:
            tile = view.copy()  # copy so the optional putText never mutates the source obs
        if label:
            # White text with a dark outline so it reads on any background.
            cv2.putText(tile, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(tile, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.ascontiguousarray(tile, dtype=np.uint8))
    return np.ascontiguousarray(np.concatenate(tiles, axis=1), dtype=np.uint8)


class MJPEGStreamer:
    """Frame sink + tiny HTTP server that streams the latest frame as MJPEG."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, fps: int = 10, quality: int = 80) -> None:
        self._host = host
        self._port = port
        self._fps = fps
        self._quality = quality
        self._latest: bytes | None = None
        self._runner: web.AppRunner | None = None

    def add_stream_frame(self, frame: np.ndarray | None) -> None:
        """Store one H x W x 3 uint8 RGB frame as the current JPEG to stream.

        ``None`` is a no-op (e.g. ``stitch`` found no available view), so callers
        can push straight through without re-guarding.
        """
        if frame is None:
            return
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8)).save(
            buf, format="JPEG", quality=self._quality
        )
        self._latest = buf.getvalue()

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/", lambda _req: web.Response(text=_PAGE, content_type="text/html"))
        app.router.add_get("/stream", self._stream)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, self._host, self._port).start()
        print(f"[stream] MJPEG live view serving on :{self._port} (/, /stream)", flush=True)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _stream(self, request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(
            headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame"}
        )
        await resp.prepare(request)
        try:
            while True:
                if self._latest is not None:
                    await resp.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + self._latest + b"\r\n"
                    )
                await asyncio.sleep(1 / self._fps)
        except (asyncio.CancelledError, ConnectionResetError, ConnectionError):
            pass
        return resp


__all__ = ["MJPEGStreamer", "stitch"]
