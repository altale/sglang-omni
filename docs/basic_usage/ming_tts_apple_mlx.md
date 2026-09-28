# Ming-TTS on Apple Silicon: native MLX implementation

This implementation targets `inclusionAI/Ming-omni-tts-16.8B-A3B`, not the
separate dense 0.5B model. Initial MLX Q4 HTTP audio generation has been manually
validated on an M4 Pro with 48 GiB memory. A separate Torch/MPS compatibility
path has passed short text-only real-weight HTTP tests; broader numerical and
audio-quality qualification and performance measurements remain pending.

## Backend and configuration

Use the existing Apple environment and select `SGLANG_USE_MLX=1`. The model's
default pipeline needs no Apple-specific YAML. Set `MING_MODEL_DIR` to the local
official checkpoint directory, then launch:

```bash
HF_HUB_OFFLINE=1 SGLANG_USE_MLX=1 \
.venv-apple/bin/sgl-omni serve \
  --model-path "$MING_MODEL_DIR" \
  --preprocessing.factory.context_length 2048 \
  --reference_encode.factory.context_length 2048 \
  --tts_engine.factory.context_length 2048 \
  --tts_engine.engine.quantization mlx_q4 \
  --model-name ming-omni-tts --host 127.0.0.1 --port 8000
```

All four stages share the default `pipeline` process. No per-stage memory
fractions are specified; this is not a memory usage limit. Previous real-weight
tests used a three-process configuration; these simplified single-process launch
commands still need E2E revalidation. The CUDA example YAML is not an Apple preset.

The four stages and speech API payloads are unchanged. Torch CPU tensors remain
at stage boundaries; model computation uses MLX, not Torch/MPS. CampPlus still
uses the existing CPU ONNX speaker encoder.

The Apple defaults allow one active generation request, TP=1 and one decoder
stream slot; the command sets a 2048-token context. They disable radix/prefix
reuse, chunked prefill, CUDA graphs and overlap/lookahead execution.
`torch_native` is only the CPU scheduler
bookkeeping backend, not the model's attention backend.

Ming uses a model-specific scheduler runner because the generic SGLang MLX
attention wrapper expects split Q/K/V projections and token-logit decoding.
The adapter preserves fused QKV, three-axis RoPE and continuous latent feedback,
uses SGLang's no-weight Torch stub and `ContiguousAttentionKVCache`, and releases
request-local state through normal completion, abort and failure callbacks.
It does not claim shared/paged MLX prefix-cache support.

## Weights and precision

Load official unquantized safetensors with the composite `config.json`, tokenizer
files and `campplus.onnx`. Loading is strict for each component. The AR stage
excludes AudioVAE, the unused LM head and known runtime rotary buffers; audio
stages load only the encoder or decoder they own. No checkpoint code is executed.

The command selects `mlx_q4`: on-load, group-size-64 quantization
of backbone linear/expert layers only. `mlx_q8` is also accepted. Remove this
setting to retain checkpoint precision. Embeddings, routers, CFM/DiT, Aggregator,
stop/speaker heads and AudioVAE are not quantized. Prequantized community
checkpoints are explicitly rejected: their metadata/layout and acoustic
quantization have not been qualified.

On-load quantization still starts from the original weights. Short single-request
Q4 generation succeeded on a 48 GiB Mac, but peak load memory and sustained-load
behavior have not been measured. This configuration is not a guarantee that
memory, quality or throughput targets are met.

CFM retains FP32 solver state; backbone RoPE retains FP32 phase/rotation followed
by casting back to the input dtype. BF16 CUDA equivalence remains to be measured.
Matching random seeds across frameworks does not imply matching noise arrays;
numerical tests supply identical noise explicitly.

## Audio and lifecycle

The AudioVAE port reuses MLX-LM Qwen2 layers with explicit causal/sliding-window
masks, implements patch aggregation and posterior sampling for reference audio,
and preserves linear interpolation lookahead and ISTFT overlap/flush for decode.
It reuses the existing streaming vocoder scheduler, terminal latent patch and
CPU waveform payload format. Stream state is per decoder slot and cleared on
terminal/error/abort; waveform output is 44.1 kHz.

Reference audio loading uses TorchAudio/TorchCodec and requires compatible FFmpeg
shared libraries. The manually tested TorchCodec 0.15 environment used Homebrew
`ffmpeg@8`, with `DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@8)/lib"` set in the
server launch environment; FFmpeg 9 alone did not satisfy its library requirements.

## Validation

### Torch/MPS compatibility

Select `SGLANG_USE_MLX=0` to use Torch on MPS. This path supports one active
request, TP=1, non-streaming text synthesis and reference-voice conditioning.
Streaming requests are rejected
in preprocessing. Use the official unquantized checkpoint; `mlx_q4`/`mlx_q8`
are MLX-only and are not accepted by the Torch/MPS builder.

Torch/MPS reuses the CUDA path's SGLang BailingMoE backbone, model runner and
paged KV pools, with `torch_native` attention and native Torch MoE/RoPE operators.
FlowLoss, Aggregator, reference encoder, weight coverage checks and acoustic
recurrence are shared as well; there is no separate Torch backbone or
request-local Transformers cache. AudioVAE neural computation runs on
MPS, with complex spectrum construction and ISTFT explicitly on CPU. CampPlus
also remains on CPU. This path does not enable a global unsupported-operator
fallback.

The defaults use BF16, without quantization, graphs, compile, prefix reuse,
chunked prefill or overlap. Short text-only generation succeeded on an M4 Pro
with 48 GiB memory; peak and sustained memory usage remain unmeasured.

Manual launch (not part of unit tests):

```bash
HF_HUB_OFFLINE=1 SGLANG_USE_MLX=0 \
.venv-apple/bin/sgl-omni serve \
  --model-path "$MING_MODEL_DIR" \
  --preprocessing.factory.context_length 2048 \
  --reference_encode.factory.context_length 2048 \
  --tts_engine.factory.context_length 2048 \
  --model-name ming-omni-tts --host 127.0.0.1 --port 8000
```

Use non-streaming `/v1/audio/speech` requests. Reference audio needs the same
TorchCodec/FFmpeg setup described above. A concurrency-one text-only benchmark
completed five requests without failures (mean latency 3.62 s, mean RTF 0.8218).
Reference-conditioned MPS generation remains unvalidated. The following unit
tests use synthetic models, not real checkpoints:

```bash
HF_HUB_OFFLINE=1 SGLANG_USE_MLX=0 \
.venv-apple/bin/python -m pytest -q \
  tests/unit_test/ming_tts/test_torch_mps_backbone.py \
  tests/unit_test/ming_tts/test_apple_runtime.py \
  tests/unit_test/ming_tts/test_sglang_model.py \
  tests/unit_test/ming_tts/test_model_runner.py \
  tests/unit_test/ming_omni/test_audio_vae_attention.py
```

### Native MLX

On a Metal-capable terminal, using the project's `.venv-apple`:

```bash
.venv-apple/bin/python -m pytest -q \
  tests/unit_test/ming_tts/test_mlx_loading.py \
  tests/unit_test/ming_tts/test_apple_runtime.py \
  tests/unit_test/ming_tts/test_mlx_model_metal.py \
  tests/unit_test/ming_tts/test_mlx_audio_metal.py
```

Metal tests use tiny synthetic weights and CPU Torch references, not real-model
generation or HTTP. They skip when Metal is unavailable. Scheduler/import tests
also require the installed SGLang Apple runtime dependencies; a collection error
is not a passing test. Broader checkpoint coverage, memory, semantic/quality and
performance acceptance remain separate follow-up validation tasks.

Hardware validation reported by the developer (2026-09-22/23): the new runtime,
model and AudioVAE unit suites passed 85 tests; the existing Ming interface
regression suite passed 132 tests. These results precede the subsequent
cross-thread first-inference regression test and are not a final-head test count.
On M4 Pro / 48 GiB, official A3B weights with on-load `mlx_q4` completed text-only
non-streaming WAV, streaming PCM, and non-streaming reference-voice HTTP requests.
Saved audio was reported to sound normal, with similar reference/generated voice
identity. Real-time streaming continuity, reference streaming, cancellation
recovery, sustained memory, and BF16/CUDA numerical parity remain unqualified.
