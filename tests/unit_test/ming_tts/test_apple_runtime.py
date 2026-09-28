# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.models.ming_tts import apple_runtime, stages
from sglang_omni.models.ming_tts.apple_runtime import (
    MingTtsMlxEngineBuilder,
    MingTtsTorchMpsEngineBuilder,
    ming_tts_uses_mlx,
)
from sglang_omni.models.ming_tts.engine_builder import MingTtsEngineBuilder


@pytest.fixture(
    params=[MingTtsMlxEngineBuilder, MingTtsTorchMpsEngineBuilder],
    ids=["mlx", "torch_mps"],
)
def builder_type(request: pytest.FixtureRequest) -> type[MingTtsEngineBuilder]:
    return request.param


@pytest.mark.parametrize("selected,apple,expected", [
    (False, False, False), (True, True, True), (True, False, None), (False, True, False),
])
def test_backend_selection(
    monkeypatch: pytest.MonkeyPatch, selected: bool, apple: bool, expected: bool | None
) -> None:
    from sglang.srt.hardware_backend.mlx import runtime
    from sglang_omni import platforms

    monkeypatch.setattr(runtime, "use_mlx", lambda: selected)
    monkeypatch.setattr(platforms, "current_platform", SimpleNamespace(is_mps=lambda: apple))
    if expected is None:
        with pytest.raises(ValueError):
            ming_tts_uses_mlx()
    else:
        assert ming_tts_uses_mlx() is expected


def test_builder_defaults(builder_type: type[MingTtsEngineBuilder]) -> None:
    builder = builder_type()
    builder.context_length = 2048
    defaults = builder.generation_defaults(dtype="bfloat16")
    builder.adjust_overrides(defaults)
    assert defaults["max_running_requests"] == 1
    assert defaults["max_total_tokens"] == 2048
    assert defaults["attention_backend"] == "torch_native"
    assert defaults["chunked_prefill_size"] == 0
    if isinstance(builder, MingTtsMlxEngineBuilder):
        assert builder.get_model_buffer_bs(None) is None


@pytest.mark.parametrize("key,value", [
    ("max_running_requests", 2), ("disable_cuda_graph", False),
    ("attention_backend", "triton"), ("max_total_tokens", 10),
    ("max_prefill_tokens", 10), ("chunked_prefill_size", 128),
    ("prefill_attention_backend", "triton"),
    ("decode_attention_backend", "triton"), ("speculative_algorithm", "EAGLE"),
])
def test_builder_rejects_unsupported_execution(
    builder_type: type[MingTtsEngineBuilder], key: str, value: Any
) -> None:
    builder = builder_type()
    builder.context_length = 2048
    overrides = builder.generation_defaults(dtype="bfloat16")
    overrides[key] = value
    with pytest.raises(ValueError):
        builder.adjust_overrides(overrides)


def test_builder_rejects_tp(builder_type: type[MingTtsEngineBuilder]) -> None:
    builder = builder_type(tp_size=2, nccl_port=12345)
    builder.context_length = 2048
    with pytest.raises(ValueError, match="TP=1"):
        builder.adjust_overrides(builder.generation_defaults(dtype="bfloat16"))


def test_engine_stage_dispatches_without_loading(
    monkeypatch: pytest.MonkeyPatch, builder_type: type[MingTtsEngineBuilder],
) -> None:
    from sglang_omni.platforms import current_platform

    monkeypatch.setattr(
        apple_runtime, "ming_tts_uses_mlx", lambda: builder_type is MingTtsMlxEngineBuilder
    )
    monkeypatch.setattr(current_platform, "is_mps", lambda: True)
    calls: list[tuple[str, int | None, dict[str, Any]]] = []

    def build(self: Any, model_path: str, **kwargs: Any) -> str:
        calls.append((model_path, self.requested_context_length, kwargs))
        return "scheduler"

    monkeypatch.setattr(builder_type, "build", build)
    assert stages.create_sglang_tts_engine_executor("local-model", context_length=2048) == "scheduler"
    assert calls[0][:2] == ("local-model", 2048)


def test_audio_stage_dispatches_without_importing_torch_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(apple_runtime, "ming_tts_uses_mlx", lambda: True)
    calls = []

    def create(model_path: str, **kwargs: Any) -> str:
        calls.append((model_path, kwargs))
        return "audio-scheduler"

    monkeypatch.setattr(stages, "create_mlx_audio_decode_executor", create)
    assert stages.create_audio_decode_executor("local-model") == "audio-scheduler"
    assert calls[0][1]["initial_chunk_patches"] == 2
    with pytest.raises(ValueError, match="streaming_cuda_graph=false"):
        stages.create_audio_decode_executor("local-model", streaming_cuda_graph=True)


def test_mps_builder_rejects_quantization() -> None:
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
        scheduler.fn(payload)


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
    decoder = scheduler.fn.keywords["decoder"]
    assert isinstance(decoder, MingTorchAudioDecoder)
    assert decoder.audio_vae is vae


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
