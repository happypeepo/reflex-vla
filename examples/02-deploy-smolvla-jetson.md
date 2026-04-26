# 02 — Deploy SmolVLA on a Jetson Orin Nano

**What you'll see:** pull SmolVLA from HuggingFace, export it to ONNX for the Orin Nano, start the inference server, hit `/act`.

**Requires:** Jetson Orin Nano (8 GB) running JetPack 6.x with `nvidia-container-runtime`. About 2 GB of free disk for weights + ONNX. Network for the initial pull.

## Install on the Jetson

```bash
pip install 'reflex-vla[serve,gpu,monolithic]'
```

Why those extras:
- `serve` — FastAPI + uvicorn for the HTTP inference server
- `gpu` — `onnxruntime-gpu` (links to CUDA on the Jetson via the nvidia container runtime)
- `monolithic` — `lerobot` + `transformers==5.3.0` + `onnx-diagnostic`, the cos=+1.0 verified export path

This pulls ~2 GB of dependencies. Takes 5-10 minutes on the Jetson.

## One command — deploy

```bash
reflex go --model smolvla-base
```

What this does, step by step:

```
device:    orin_nano (via tegrastats, GPU=Jetson Orin Nano)
model:     smolvla-base (lerobot/smolvla_base, 900MB, action_dim=7)
  strategy: exact-id
pulling:   lerobot/smolvla_base → ~/.cache/reflex/models/smolvla-base/
           ↓ 900 MB from HuggingFace (~30 sec)
exporting: ~/.cache/reflex/models/smolvla-base → ~/.cache/reflex/exports/smolvla-base
           (target=orin-nano, monolithic, 5-15 min depending on hardware)
           ↓ Loading PyTorch model
           ↓ Tracing torch.export (the heaviest step on Orin Nano)
           ↓ Writing ONNX (~1.6 GB on disk)
           ↓ Validating cos=+1.0 vs PyTorch reference
export complete in 612.4s  ONNX=model.onnx (1623 MB)

Starting serve on http://0.0.0.0:8000
  Loading ONNX into onnxruntime-gpu (CUDAExecutionProvider)...
  TRT engine build (first time)...   ~60-90 sec
  Warmup inference...                ~5 sec
  ✓ Server ready
```

## Hit /act

From another terminal (or a connected workstation):

```bash
curl -X POST http://<jetson-ip>:8000/act \
  -H 'content-type: application/json' \
  -d '{
    "instruction": "pick up the red cup",
    "image": "<base64-png-or-jpeg>",
    "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  }'
```

Response:

```json
{
  "actions": [[...], [...], ...],
  "latency_ms": 47.3,
  "inference_mode": "onnx_trt_fp16",
  "guard_clamped": false
}
```

## What just happened

`reflex go` chained:

1. **Hardware probe** — `tegrastats` confirms Orin Nano (8 GB)
2. **Model resolution** — picked `smolvla-base` from the curated registry; warned if your `--device-class` doesn't match the model's `supported_devices`
3. **Pull** — `huggingface_hub.snapshot_download`; cached in `~/.cache/reflex/models/`
4. **Export** — `reflex.exporters.monolithic.export_monolithic` traces PyTorch → ONNX with `num_steps=10` baked in, validates parity at cos=+1.0; output cached in `~/.cache/reflex/exports/`
5. **Serve** — `reflex.runtime.server.create_app` mounts the ONNX into onnxruntime-gpu, builds a TRT FP16 engine on first run (cached for next time), exposes `/act` and `/health`

Re-running `reflex go --model smolvla-base` skips pull (cache hit) and skips export (`VERIFICATION.md` marker hit), goes straight to serve in ~2 sec.

## Or use the chat

If you'd rather have the chat agent run all this for you:

```bash
reflex chat
you › deploy smolvla to my orin nano
```

Watch it call `list_targets`, `pull_model`, `export_model`, `serve_model` in sequence.

## Troubleshooting

- **"Missing dependencies for monolithic export"** — install `[monolithic]`: `pip install 'reflex-vla[monolithic]'`
- **"CUDA unavailable"** — confirm `nvidia-container-runtime` is set up on the Jetson; `reflex doctor` will tell you which check failed
- **TRT engine build fails** — try `--no-trt` to fall back to plain CUDAExecutionProvider; usually means `trtexec` isn't on PATH
- **Disk full** — SmolVLA needs ~2 GB free for weights + ONNX. `reflex inspect targets` shows memory budgets per hardware tier.
