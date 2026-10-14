# SPDX-License-Identifier: Apache-2.0
"""Single-threaded Ming MLX speech generation, independent of SGLang."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from threading import Event

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from sglang_omni_mlx.ming_tts.audio_io import MingMlxAudioDecoder
from sglang_omni_mlx.ming_tts.frontend import (
    SamplingParams,
    SpeechRequest,
    build_prompt,
    load_tokenizer,
)
from sglang_omni_mlx.ming_tts.loading import load_ming_audio_vae, load_ming_tts_model
from sglang_omni_mlx.ming_tts.reference import ReferenceEncoder
from sglang_omni_mlx.ming_tts.runner import MingTTSMlxRunner


class MingTTSPipeline:
    def __init__(
        self,
        model_path: str,
        *,
        context_length: int,
        quantization: str | None,
        reference_audio: bool,
    ) -> None:
        self.stream = mx.new_thread_local_stream(mx.gpu)
        with mx.stream(self.stream):
            model = load_ming_tts_model(model_path, quantization=quantization)
            if (
                not 1
                <= context_length
                <= model.config.llm_config.max_position_embeddings
            ):
                raise ValueError(
                    "context_length exceeds the checkpoint's supported range"
                )
            else:
                pass
            self.context_length = context_length
            self.runner = MingTTSMlxRunner(model)
            self.tokenizer = load_tokenizer(Path(model_path))
            self.decoder = MingMlxAudioDecoder(
                load_ming_audio_vae(model_path, component="decoder")
            )
            self.sample_rate = self.decoder.sample_rate
            self.reference_audio = reference_audio
            if reference_audio:
                self.audio_encoder = load_ming_audio_vae(
                    model_path, component="encoder"
                )
                self.reference_encoder = ReferenceEncoder(
                    Path(model_path),
                    sample_rate=self.sample_rate,
                    patch_size=model.config.patch_size,
                )
            else:
                pass

    def generate(
        self, request: SpeechRequest, cancelled: Event
    ) -> Generator[NDArray[np.float32], None, None]:
        with mx.stream(self.stream):
            reference_latents = speaker_embedding = None
            if request.ref_audio is not None:
                if not self.reference_audio:
                    raise ValueError("Reference audio is disabled on this server")
                else:
                    pass
                features = self.reference_encoder.encode(request.ref_audio)
                reference_latents = self.audio_encoder.encode_latent(
                    mx.array(features.waveform)
                )[0]
                speaker_embedding = mx.array(features.speaker_embedding)
                reference_tokens = (
                    reference_latents.shape[0] // self.runner.model.patch_size
                )
            else:
                reference_tokens = 0
            prompt = build_prompt(request, self.tokenizer, reference_tokens)
            if len(prompt.input_ids) + request.max_new_tokens > self.context_length:
                raise ValueError("Prompt and generation exceed context_length")
            else:
                pass
            sampling = request.stage_params.get("tts_engine", SamplingParams())
            temperature = (
                request.temperature
                if sampling.temperature is None
                else sampling.temperature
            )
            pending: list[mx.array] = []
            chunk_patches = 2
            try:
                self.runner.start(
                    "speech",
                    mx.array(prompt.input_ids, dtype=mx.int32),
                    max_steps=request.max_new_tokens,
                    reference_latents=reference_latents,
                    reference_start=prompt.reference_start,
                    speaker_embedding=speaker_embedding,
                    speaker_positions=prompt.speaker_positions,
                )
                while not cancelled.is_set():
                    step = self.runner.step(
                        "speech",
                        cfg=sampling.cfg,
                        sigma=sampling.sigma,
                        temperature=temperature,
                    )
                    pending.append(step.latent)
                    terminal = step.finish_reason is not None
                    if request.stream and (len(pending) == chunk_patches or terminal):
                        waveform = self.decoder.decode_chunk(
                            mx.concatenate(pending), last_chunk=terminal
                        )
                        pending.clear()
                        chunk_patches = 4
                        if waveform.size:
                            yield waveform
                        else:
                            pass
                    else:
                        pass
                    if terminal:
                        if not request.stream:
                            yield self.decoder.decode_full(mx.concatenate(pending))
                        else:
                            pass
                        break
                    else:
                        pass
            finally:
                self.runner.release("speech")
                self.decoder.reset()
