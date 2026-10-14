# SPDX-License-Identifier: Apache-2.0
"""Speech request validation and Ming's tokenizer prompt format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator
from tokenizers import Tokenizer

DEFAULT_PROMPT = "Please generate speech based on the following description.\n"
BGM_MARKERS = ("Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: ")


class SamplingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cfg: FiniteFloat = Field(default=2.0, ge=1e-5)
    sigma: FiniteFloat = Field(default=0.25, ge=0)
    temperature: FiniteFloat | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_cfg(self) -> SamplingParams:
        if self.cfg == 1.0:
            raise ValueError("cfg=1 is not supported by Ming flow matching")
        else:
            return self


class ReferenceAudio(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio_path: str = Field(min_length=1)
    text: str = Field(min_length=1)


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    input: str = Field(min_length=1)
    response_format: Literal["wav", "pcm"] = "wav"
    stream: bool = False
    voice: Literal["default"] = "default"
    speed: Literal[1.0] = 1.0
    max_new_tokens: int = Field(default=200, ge=1, le=256)
    temperature: FiniteFloat = Field(default=0.0, ge=0)
    prompt: str | None = None
    instructions: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    references: list[ReferenceAudio] | None = Field(default=None, max_length=1)
    stage_params: dict[str, SamplingParams] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_request(self) -> SpeechRequest:
        if not self.input.strip():
            raise ValueError("input must contain text")
        elif self.stream and self.response_format != "pcm":
            raise ValueError("Streaming requires response_format=pcm")
        elif self.stage_params.keys() - {"tts_engine"}:
            raise ValueError("Only tts_engine sampling parameters are supported")
        else:
            pass
        if self.references:
            if self.ref_audio is not None or self.ref_text is not None:
                raise ValueError("Use references or ref_audio/ref_text, not both")
            else:
                self.ref_audio = self.references[0].audio_path
                self.ref_text = self.references[0].text
        else:
            pass
        if (self.ref_audio is None) != (self.ref_text is None):
            raise ValueError("Reference audio and transcript must be supplied together")
        elif self.ref_text is not None and not self.ref_text.strip():
            raise ValueError("Reference transcript must contain text")
        else:
            return self


@dataclass(kw_only=True)
class Prompt:
    input_ids: list[int]
    speaker_positions: list[int]
    reference_start: int | None


def load_tokenizer(model_path: Path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    for token in (
        "<role>",
        "</role>",
        "<audioPatch>",
        "<audio>",
        "<spk>",
        "<|endoftext|>",
    ):
        token_id = tokenizer.token_to_id(token)
        if token_id is None or tokenizer.encode(
            token, add_special_tokens=False
        ).ids != [token_id]:
            raise ValueError(
                f"Checkpoint tokenizer requires a single token for {token}"
            )
        else:
            pass
    return tokenizer


def build_prompt(
    request: SpeechRequest, tokenizer: Tokenizer, reference_tokens: int = 0
) -> Prompt:
    input_ids = tokenizer.encode("<role>HUMAN</role>", add_special_tokens=False).ids
    input_ids += tokenizer.encode(
        request.prompt or DEFAULT_PROMPT, add_special_tokens=False
    ).ids
    speaker_positions = []
    if reference_tokens:
        input_ids += tokenizer.encode("  speaker_1:", add_special_tokens=False).ids
        input_ids += tokenizer.encode("<spk>", add_special_tokens=False).ids
        speaker_positions.append(len(input_ids))
        input_ids.append(tokenizer.token_to_id("<audioPatch>"))
        input_ids += tokenizer.encode("</spk>\n", add_special_tokens=False).ids
    else:
        pass
    if not all(marker in request.input for marker in BGM_MARKERS):
        input_ids += tokenizer.encode(" Text input:\n", add_special_tokens=False).ids
    else:
        pass
    if request.ref_text:
        input_ids += tokenizer.encode(request.ref_text, add_special_tokens=False).ids
    else:
        pass
    input_ids += tokenizer.encode(request.input, add_special_tokens=False).ids
    input_ids += tokenizer.encode(
        "<role>ASSISTANT</role>", add_special_tokens=False
    ).ids
    if request.instructions:
        input_ids += tokenizer.encode(
            request.instructions, add_special_tokens=False
        ).ids
        input_ids.append(tokenizer.token_to_id("<|endoftext|>"))
    else:
        pass
    input_ids.append(tokenizer.token_to_id("<audio>"))
    reference_start = len(input_ids) if reference_tokens else None
    input_ids += [tokenizer.token_to_id("<audioPatch>")] * reference_tokens
    return Prompt(
        input_ids=input_ids,
        speaker_positions=speaker_positions,
        reference_start=reference_start,
    )
