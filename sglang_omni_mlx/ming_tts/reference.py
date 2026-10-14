# SPDX-License-Identifier: Apache-2.0
"""CPU reference-audio preprocessing and CampPlus speaker embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from numpy.typing import NDArray
from torchaudio.compliance.kaldi import fbank


@dataclass(kw_only=True)
class ReferenceAudioFeatures:
    waveform: NDArray[np.float32]
    speaker_embedding: NDArray[np.float32]


class ReferenceEncoder:
    def __init__(self, model_path: Path, *, sample_rate: int, patch_size: int) -> None:
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.intra_op_num_threads = 2
        self.session = onnxruntime.InferenceSession(
            str(model_path / "campplus.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.sample_rate = sample_rate
        self.patch_size = patch_size

    def encode(self, path: str) -> ReferenceAudioFeatures:
        try:
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        except sf.LibsndfileError as exc:
            raise ValueError(f"Cannot read reference audio: {exc}") from exc
        if audio.shape[1] != 1 or audio.shape[0] == 0:
            raise ValueError("Reference audio must be nonempty mono audio")
        else:
            pass
        waveform = torch.from_numpy(audio.T.copy())
        speaker_waveform = audio_functional.resample(waveform, sample_rate, 16000)
        if speaker_waveform.shape[-1] < 400:
            raise ValueError("Reference audio must contain at least 25 ms")
        else:
            pass
        features = fbank(speaker_waveform, num_mel_bins=80, sample_frequency=16000)
        features -= features.mean(dim=0, keepdim=True)
        speaker_embedding = self.session.run(
            None, {self.session.get_inputs()[0].name: features.unsqueeze(0).numpy()}
        )[0].reshape(1, 192)
        waveform = audio_functional.resample(waveform, sample_rate, self.sample_rate)
        alignment = int(self.sample_rate * self.patch_size / 12.5)
        padding = (-waveform.shape[-1]) % alignment
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        return ReferenceAudioFeatures(
            waveform=waveform.numpy(),
            speaker_embedding=np.asarray(speaker_embedding, dtype=np.float32),
        )
