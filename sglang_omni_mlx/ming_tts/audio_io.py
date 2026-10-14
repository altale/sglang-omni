# SPDX-License-Identifier: Apache-2.0
"""Request-local native MLX audio decoding."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from sglang_omni_mlx.ming_tts.audio_vae import AudioDecoderState, AudioVAE


class MingMlxAudioDecoder:
    def __init__(self, audio_vae: AudioVAE) -> None:
        self.audio_vae = audio_vae
        self.sample_rate = int(audio_vae.config["sample_rate"])
        self.state: AudioDecoderState | None = None

    def decode_full(self, latents: mx.array) -> NDArray[np.float32]:
        audio, _ = self.audio_vae.decoder(latents.reshape(1, -1, latents.shape[-1]))
        return np.array(audio[0].astype(mx.float32))

    def decode_chunk(
        self, latents: mx.array, *, last_chunk: bool
    ) -> NDArray[np.float32]:
        try:
            audio, self.state = self.audio_vae.decoder(
                latents.reshape(1, -1, latents.shape[-1]),
                state=self.state,
                streaming=True,
                last_chunk=last_chunk,
            )
            waveform = np.array(audio[0].astype(mx.float32))
            if self.state is not None:
                arrays = [
                    value
                    for cache in self.state.cache
                    for value in (cache.keys, cache.values)
                    if value is not None
                ]
                if self.state.upsample is not None:
                    arrays.append(self.state.upsample.pending)
                    if self.state.upsample.left is not None:
                        arrays.append(self.state.upsample.left)
                    else:
                        pass
                else:
                    pass
                if self.state.overlap is not None:
                    arrays.extend(self.state.overlap)
                else:
                    pass
                mx.eval(arrays)
            else:
                pass
            return waveform
        except Exception:
            self.reset()
            raise

    def reset(self) -> None:
        self.state = None
