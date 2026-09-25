# SPDX-License-Identifier: Apache-2.0
"""Synthetic Torch/MPS integration; no checkpoints or HTTP server."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.ming_tts import apple_runtime, stages
from sglang_omni.models.ming_tts.apple_runtime import MingTtsTorchMpsEngineBuilder


def test_builder_rejects_quantization() -> None:
    builder = MingTtsTorchMpsEngineBuilder()
    builder.context_length = 64
    overrides = builder.generation_defaults(dtype="bfloat16")
    overrides["quantization"] = "mlx_q4"
    with pytest.raises(ValueError, match="does not support quantization"):
        builder.adjust_overrides(overrides)


def test_preprocessing_rejects_streaming_before_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_mps", lambda: True)
    monkeypatch.setattr(apple_runtime, "ming_tts_uses_mlx", lambda: False)
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda _: "unused")
    monkeypatch.setattr(
        stages, "load_ming_tts_config", lambda _: SimpleNamespace(llm_config=None)
    )
    monkeypatch.setattr(stages, "load_ming_tts_tokenizer", lambda *a, **kw: None)
    scheduler = stages.create_preprocessing_executor("unused", context_length=64)
    payload = SimpleNamespace(request=SimpleNamespace(params={"stream": True}))
    with pytest.raises(ValueError, match="non-streaming"):
        scheduler._fn(payload)


def test_audio_factory_uses_nonstream_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang_omni.models.ming_tts.audio_decode import MingTorchAudioDecoder
    from sglang_omni.utils import device

    monkeypatch.setattr(apple_runtime, "ming_tts_uses_mlx", lambda: False)
    monkeypatch.setattr(
        device, "resolve_concrete_device", lambda *a: torch.device("mps")
    )
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda _: "unused")
    monkeypatch.setattr(
        stages, "load_ming_tts_config",
        lambda _: SimpleNamespace(audio_tokenizer_config=None),
    )
    monkeypatch.setattr(
        stages, "resolve_ming_tts_audio_vae_config", lambda *a, **kw: None
    )
    vae = object()
    monkeypatch.setattr(stages, "load_ming_tts_audio_vae", lambda *a, **kw: vae)
    scheduler = stages.create_audio_decode_executor("unused")
    decoder = scheduler._fn.keywords["decoder"]
    assert isinstance(decoder, MingTorchAudioDecoder)
    assert decoder._audio_vae is vae


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple Metal")
def test_mps_audio_vae_encode_and_full_decode() -> None:
    from sglang_omni.models.ming_omni.talker.audio_vae.configuration_audio_vae import (
        AudioVAEconfig,
    )
    from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import (
        AudioVAE,
    )
    from sglang_omni.models.ming_tts.audio_decode import MingTorchAudioDecoder

    backbone = dict(
        vocab_size=8, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
        _attn_implementation="sdpa", use_sliding_window=True, sliding_window=5,
        max_window_layers=0,
    )
    config = AudioVAEconfig(
        sample_rate=44100, patch_size=2,
        enc_kwargs=dict(backbone=backbone, input_dim=8, hop_size=8, latent_dim=4),
        dec_kwargs=dict(backbone=backbone, output_dim=8, latent_dim=4),
    )
    vae = AudioVAE(config).eval().to(device="mps", dtype=torch.bfloat16)
    with torch.inference_mode():
        latents, _ = vae.encode_latent(
            torch.randn(1, 64, device="mps", dtype=torch.bfloat16),
            torch.tensor([64], device="mps"),
        )
    assert latents.shape == (1, 4, 4)
    decoder = MingTorchAudioDecoder(vae)
    waveform = decoder.decode_full(latents.reshape(2, 2, 4))
    assert waveform.shape == (64,)
    assert waveform.device.type == "cpu" and waveform.dtype == torch.float32
    assert torch.isfinite(waveform).all()
    assert decoder.decode_full(torch.empty(0, 2, 4)).numel() == 0
