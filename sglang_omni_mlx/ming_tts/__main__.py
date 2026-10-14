# SPDX-License-Identifier: Apache-2.0
"""Standalone Ming MLX server entry point."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from sglang_omni_mlx.ming_tts.pipeline import MingTTSPipeline
from sglang_omni_mlx.ming_tts.server import create_app


def main(
    model_path: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    model_name: str = "ming-omni-tts",
    quantization: str | None = None,
    context_length: int = 2048,
    allowed_local_media_path: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False)
    ] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ming-mlx") as executor:
        pipeline = executor.submit(
            partial(
                MingTTSPipeline,
                str(model_path),
                context_length=context_length,
                quantization=quantization,
                reference_audio=allowed_local_media_path is not None,
            )
        ).result()
        uvicorn.run(
            create_app(
                pipeline,
                executor,
                model_name=model_name,
                allowed_local_media_path=allowed_local_media_path,
            ),
            host=host,
            port=port,
        )


if __name__ == "__main__":
    typer.run(main)
else:
    pass
