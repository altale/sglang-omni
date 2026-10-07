# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Literal

import pytest
from pydantic import JsonValue

from sglang_omni.models.ming_tts.mlx.config import ModelConfig, TextConfig
from sglang_omni.models.ming_tts.mlx.loading import (
    checkpoint_files,
    load_component_weights,
)


def text_config_dict() -> dict[str, JsonValue]:
    return dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        multi_gate=True,
        rope_scaling={"type": "3D", "factor": None, "mrope_section": [1, 1, 2]},
    )


def test_composite_config_accepts_tiny_a3b_structure_without_importing_mlx() -> None:
    config = ModelConfig.from_dict(
        dict(
            llm_config=text_config_dict(),
            ditar_config=dict(
                hidden_size=16, depth=2, num_heads=2, patch_size=2, history_patch_size=4
            ),
            aggregator_config=dict(hidden_size=16, depth=1, num_heads=2),
            audio_tokenizer_config={"enc_kwargs": {"latent_dim": 4}},
            architectures=["BailingMMNativeForConditionalGeneration"],
        )
    )
    assert config.llm_config.mrope_section == (1, 1, 2)
    assert (config.patch_size, config.history_patch_size, config.latent_dim) == (
        2,
        4,
        4,
    )


def test_reject_dense_checkpoint() -> None:
    with pytest.raises(ValueError, match="A3B MoE"):
        TextConfig.from_dict({**text_config_dict(), "num_experts": 0})


def test_official_mrope_sections() -> None:
    config = TextConfig.from_dict(
        {
            **text_config_dict(),
            "head_dim": 128,
            "rope_scaling": {"type": "3D", "factor": None},
        }
    )
    assert config.mrope_section == (16, 24, 24)


def test_checkpoint_index_rejects_escape(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "../outside.safetensors"}})
    )
    with pytest.raises(ValueError, match="within the model directory"):
        checkpoint_files(tmp_path)


@pytest.mark.parametrize(
    "component,expected",
    [
        ("ar", {"model.model.norm.weight": "norm", "stop_head.weight": "head"}),
        ("audio", {"encoder.fc1.weight": "encoder", "decoder.fc1.weight": "decoder"}),
    ],
)
def test_sharded_snapshot_component_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: Literal["ar", "audio"],
    expected: dict[str, str],
) -> None:
    parent = ModuleType("mlx")
    core = ModuleType("mlx.core")
    shards = {
        "part-1.safetensors": {
            "model.model.norm.weight": "norm",
            "audio.encoder.fc1.weight": "encoder",
        },
        "part-2.safetensors": {
            "stop_head.weight": "head",
            "audio.decoder.fc1.weight": "decoder",
        },
    }
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for filename in shards:
        blob = tmp_path / filename
        blob.touch()
        (snapshot / filename).symlink_to(blob)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    name: filename
                    for filename, weights in shards.items()
                    for name in weights
                }
            }
        )
    )
    loaded_paths = []

    def load(path: str) -> dict[str, str]:
        loaded_paths.append(Path(path))
        assert Path(path).is_file()
        return shards[Path(path).name]

    core.load = load
    parent.core = core
    monkeypatch.setitem(sys.modules, "mlx", parent)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    assert load_component_weights(snapshot, component=component) == expected
    assert loaded_paths == [snapshot / filename for filename in shards]
