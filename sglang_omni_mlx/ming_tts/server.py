# SPDX-License-Identifier: Apache-2.0
"""OpenAI-style speech API with one serialized MLX execution thread."""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from numpy.typing import NDArray
from starlette.background import BackgroundTask

from sglang_omni_mlx.ming_tts.frontend import SpeechRequest

if TYPE_CHECKING:
    from sglang_omni_mlx.ming_tts.pipeline import MingTTSPipeline
else:
    pass

logger = logging.getLogger(__name__)


def pcm_bytes(waveform: NDArray[np.float32]) -> bytes:
    return np.clip(waveform * 32768.0, -32768, 32767).astype("<i2").tobytes()


def create_app(
    pipeline: MingTTSPipeline,
    executor: ThreadPoolExecutor,
    *,
    model_name: str,
    allowed_local_media_path: Path | None = None,
) -> FastAPI:
    busy = asyncio.Lock()
    media_root = (
        allowed_local_media_path.resolve() if allowed_local_media_path else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await asyncio.get_running_loop().run_in_executor(executor, lambda: None)

    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "; ".join(error["msg"] for error in exc.errors()),
                    "type": "invalid_request_error",
                }
            },
        )

    @app.get("/health")
    async def health() -> dict[str, str | bool]:
        return {"status": "healthy", "running": True}

    @app.get("/v1/models")
    async def models() -> dict[str, str | list[dict[str, str]]]:
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/audio/speech", response_model=None)
    async def speech(request: SpeechRequest) -> Response:
        if request.model != model_name:
            raise HTTPException(404, "Unknown model")
        else:
            pass
        if request.ref_audio is not None:
            if not pipeline.reference_audio or media_root is None:
                raise HTTPException(
                    400, "Reference audio requires --allowed-local-media-path"
                )
            else:
                pass
            url = urlsplit(request.ref_audio)
            if url.scheme not in ("", "file") or url.netloc not in ("", "localhost"):
                raise HTTPException(400, "Only local reference audio is supported")
            else:
                pass
            path = Path(unquote(url.path)).resolve()
            if not path.is_relative_to(media_root):
                raise HTTPException(
                    403, "Reference audio is outside the allowed directory"
                )
            elif not path.is_file():
                raise HTTPException(400, "Reference audio file does not exist")
            else:
                request.ref_audio = str(path)
        else:
            pass
        if busy.locked():
            raise HTTPException(429, "Only one active speech request is supported")
        else:
            await busy.acquire()
        loop = asyncio.get_running_loop()
        cancelled = Event()
        generation = pipeline.generate(request, cancelled)

        def finish() -> None:
            if cancelled.is_set():
                return
            else:
                cancelled.set()

            # Note (altale): Close on the model thread after any in-flight Metal work.
            def close_generation() -> None:
                try:
                    generation.close()
                except Exception:
                    logger.exception("Ming generation cleanup failed")
                finally:
                    loop.call_soon_threadsafe(busy.release)

            executor.submit(close_generation)

        try:
            first = await loop.run_in_executor(executor, next, generation, None)
            if first is None:
                raise RuntimeError("Ming produced no audio")
            else:
                pass
        except ValueError as exc:
            finish()
            raise HTTPException(400, str(exc)) from exc
        except BaseException:
            finish()
            raise

        if request.stream:

            async def chunks() -> AsyncIterator[bytes]:
                try:
                    yield pcm_bytes(first)
                    while True:
                        waveform = await loop.run_in_executor(
                            executor, next, generation, None
                        )
                        if waveform is None:
                            break
                        else:
                            yield pcm_bytes(waveform)
                finally:
                    finish()

            return StreamingResponse(
                chunks(),
                media_type="audio/pcm",
                background=BackgroundTask(finish),
                headers={
                    "X-Sample-Rate": str(pipeline.sample_rate),
                    "X-Channels": "1",
                    "X-Bit-Depth": "16",
                },
            )
        else:
            try:
                audio = pcm_bytes(first)
                if request.response_format == "wav":
                    buffer = io.BytesIO()
                    with wave.open(buffer, "wb") as output:
                        output.setnchannels(1)
                        output.setsampwidth(2)
                        output.setframerate(pipeline.sample_rate)
                        output.writeframes(audio)
                    return Response(buffer.getvalue(), media_type="audio/wav")
                else:
                    return Response(
                        audio,
                        media_type="audio/pcm",
                        headers={
                            "X-Sample-Rate": str(pipeline.sample_rate),
                            "X-Channels": "1",
                            "X-Bit-Depth": "16",
                        },
                    )
            finally:
                finish()

    return app
