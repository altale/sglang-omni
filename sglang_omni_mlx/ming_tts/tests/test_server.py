# SPDX-License-Identifier: Apache-2.0
"""Standalone speech API contracts without model weights or Metal."""

from __future__ import annotations

import asyncio
import io
import json
import wave
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, get_ident

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from tokenizers import Tokenizer, models

from sglang_omni_mlx.ming_tts.frontend import SpeechRequest, build_prompt
from sglang_omni_mlx.ming_tts.server import create_app


class FakePipeline:
    sample_rate = 44100
    reference_audio = True

    def __init__(self) -> None:
        self.started = Event()
        self.released = Event()
        self.gate: Event | None = None
        self.thread_ids: set[int] = set()
        self.cancelled: Event | None = None
        self.error: ValueError | None = None

    def generate(
        self, request: SpeechRequest, cancelled: Event
    ) -> Generator[NDArray[np.float32], None, None]:
        self.cancelled = cancelled
        self.thread_ids.add(get_ident())
        self.started.set()
        try:
            if self.error is not None:
                raise self.error
            else:
                pass
            if self.gate is not None:
                assert self.gate.wait(5)
            else:
                pass
            for _ in range(3 if request.stream else 1):
                if cancelled.is_set():
                    break
                else:
                    self.thread_ids.add(get_ident())
                    yield np.zeros(128, dtype=np.float32)
        finally:
            self.thread_ids.add(get_ident())
            self.released.set()


@pytest.mark.parametrize("stream", [False, True])
def test_audio_and_cleanup(stream: bool) -> None:
    pipeline = FakePipeline()
    with ThreadPoolExecutor(max_workers=1) as executor:
        with TestClient(create_app(pipeline, executor, model_name="ming")) as client:
            assert client.get("/health").json()["running"]
            response = client.post(
                "/v1/audio/speech",
                json={
                    "model": "ming",
                    "input": "你好",
                    "stream": stream,
                    "response_format": "pcm" if stream else "wav",
                },
            )
            assert response.status_code == 200
            if stream:
                assert response.headers["x-sample-rate"] == "44100"
                assert response.headers["content-type"] == "audio/pcm"
                assert len(response.content) == 3 * 128 * 2
            else:
                with wave.open(io.BytesIO(response.content)) as audio:
                    assert (
                        audio.getnchannels(),
                        audio.getsampwidth(),
                        audio.getframerate(),
                        audio.getnframes(),
                    ) == (1, 2, 44100, 128)
            assert pipeline.released.wait(5)
            assert (
                client.post(
                    "/v1/audio/speech", json={"model": "ming", "input": "again"}
                ).status_code
                == 200
            )
    assert len(pipeline.thread_ids) == 1
    assert get_ident() not in pipeline.thread_ids


@pytest.mark.parametrize(
    "fields,status",
    [
        ({"model": "wrong"}, 404),
        ({"input": " "}, 400),
        ({"stream": True}, 400),
        ({"ref_audio": "/missing.wav"}, 400),
        ({"max_new_tokens": 0}, 400),
        ({"stage_params": {"tts_engine": {"cfg": 1}}}, 400),
    ],
)
def test_invalid_request_before_inference(
    fields: dict[str, object], status: int
) -> None:
    pipeline = FakePipeline()
    with ThreadPoolExecutor(max_workers=1) as executor:
        with TestClient(create_app(pipeline, executor, model_name="ming")) as client:
            response = client.post(
                "/v1/audio/speech", json={"model": "ming", "input": "test", **fields}
            )
            assert response.status_code == status
            assert not pipeline.started.is_set()


def test_reference_path_boundary(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.wav"
    outside.touch()
    (allowed / "escape.wav").symlink_to(outside)
    pipeline = FakePipeline()
    with ThreadPoolExecutor(max_workers=1) as executor:
        with TestClient(
            create_app(
                pipeline, executor, model_name="ming", allowed_local_media_path=allowed
            )
        ) as client:
            response = client.post(
                "/v1/audio/speech",
                json={
                    "model": "ming",
                    "input": "test",
                    "references": [
                        {
                            "audio_path": (allowed / "escape.wav").as_uri(),
                            "text": "reference",
                        }
                    ],
                },
            )
            assert response.status_code == 403
            assert not pipeline.started.is_set()


def test_busy_request_is_rejected() -> None:
    pipeline = FakePipeline()
    pipeline.gate = Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        with TestClient(create_app(pipeline, executor, model_name="ming")) as client:
            with ThreadPoolExecutor(max_workers=1) as callers:
                first = callers.submit(
                    client.post,
                    "/v1/audio/speech",
                    json={"model": "ming", "input": "first"},
                )
                try:
                    assert pipeline.started.wait(5)
                    assert (
                        client.post(
                            "/v1/audio/speech",
                            json={"model": "ming", "input": "second"},
                        ).status_code
                        == 429
                    )
                finally:
                    pipeline.gate.set()
                assert first.result(timeout=5).status_code == 200


def test_inference_error_releases_generation() -> None:
    pipeline = FakePipeline()
    pipeline.error = ValueError("Invalid reference audio")
    with ThreadPoolExecutor(max_workers=1) as executor:
        with TestClient(create_app(pipeline, executor, model_name="ming")) as client:
            request = {"model": "ming", "input": "test"}
            response = client.post("/v1/audio/speech", json=request)
            assert response.status_code == 400
            assert pipeline.released.wait(5)
            pipeline.error = None
            assert client.post("/v1/audio/speech", json=request).status_code == 200


@pytest.mark.asyncio
async def test_stream_disconnect_releases_generation() -> None:
    pipeline = FakePipeline()
    disconnected = asyncio.Event()
    body = json.dumps(
        {"model": "ming", "input": "test", "stream": True, "response_format": "pcm"}
    ).encode()
    sent_request = False

    async def receive() -> dict[str, object]:
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": body, "more_body": False}
        else:
            await disconnected.wait()
            return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            disconnected.set()
        else:
            pass

    with ThreadPoolExecutor(max_workers=1) as executor:
        app = create_app(pipeline, executor, model_name="ming")
        await asyncio.wait_for(
            app(
                {
                    "type": "http",
                    "asgi": {"version": "3.0", "spec_version": "2.0"},
                    "http_version": "1.1",
                    "method": "POST",
                    "scheme": "http",
                    "path": "/v1/audio/speech",
                    "raw_path": b"/v1/audio/speech",
                    "query_string": b"",
                    "headers": [(b"content-type", b"application/json")],
                    "server": ("test", 80),
                    "client": ("test", 1),
                    "root_path": "",
                },
                receive,
                send,
            ),
            timeout=5,
        )
        assert await asyncio.to_thread(pipeline.released.wait, 5)
        assert pipeline.cancelled.is_set()
        assert len(pipeline.thread_ids) == 1


def test_prompt_reference_and_instructions() -> None:
    vocabulary = {
        token: index
        for index, token in enumerate(
            [
                "[UNK]",
                "<role>",
                "</role>",
                "<audio>",
                "<audioPatch>",
                "<spk>",
                "</spk>",
                "<|endoftext|>",
            ]
        )
    }
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.add_special_tokens(list(vocabulary)[1:])
    request = SpeechRequest(
        model="ming",
        input="test",
        instructions="calm",
        ref_audio="voice.wav",
        ref_text="reference",
    )
    prompt = build_prompt(request, tokenizer, reference_tokens=2)
    assert prompt.input_ids[0] == vocabulary["<role>"]
    assert prompt.input_ids[prompt.speaker_positions[0]] == vocabulary["<audioPatch>"]
    assert prompt.input_ids[prompt.reference_start - 2 : prompt.reference_start] == [
        vocabulary["<|endoftext|>"],
        vocabulary["<audio>"],
    ]
    assert (
        prompt.input_ids[prompt.reference_start :] == [vocabulary["<audioPatch>"]] * 2
    )
