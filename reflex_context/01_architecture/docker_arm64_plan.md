# Docker arm64/Jetson plan (2026-04-20)

## Why

Our `ghcr.io/rylinjames/reflex-vla:latest` image is x86_64-only. The README's "docker run and go" quickstart lands on a dead-end for the primary customer (Jetson Orin Nano / Orin / Thor). Adding an arm64 variant closes the first-hour onboarding gap.

With pi0 FP16 (6.26 GB) now fitting Orin Nano 8GB, the story `pip install` or `docker pull → reflex serve → POST /act` is achievable — but only if the docker image runs on the Jetson's aarch64 CPU/GPU.

## Approach

**Phase 1 (DONE this session)**: arm64 image with reflex-vla's Python runtime + ONNX deps, no CUDA/TRT bundled. Jetson users `docker run --runtime=nvidia` and the host's JetPack CUDA libs get injected into the container via the NVIDIA container runtime. Image size: ~1.2 GB.

**Phase 2 (future, tracked separately)**: `-jetpack` image variant that bundles NVIDIA's L4T base + TRT. Targets specific JetPack versions. Will land via a `Dockerfile.jetpack` variant + a new workflow pinned to `nvcr.io/nvidia/l4t-pytorch:r36.x-pth2.x-py3`.

## Files

- **`Dockerfile.arm64`** — Python 3.12 slim base (arm64-capable). Installs `reflex-vla[serve,monolithic]` — no `[gpu]` extra because onnxruntime-gpu's CUDA 12.x pin clashes with JetPack's CUDA.
- **`.github/workflows/docker-publish-arm64.yml`** — QEMU + buildx cross-build on the x86 GitHub runner; pushes to `ghcr.io/rylinjames/reflex-vla:<version>-arm64`.
- **`tests/test_docker_arm64.py`** — 12 sanity tests that validate the Dockerfile has the expected ENTRYPOINT, doesn't install `[gpu]`, and the workflow has QEMU + buildx + Dockerfile.arm64 targeting linux/arm64.

## Tag scheme

- `ghcr.io/rylinjames/reflex-vla:<version>-arm64` — arm64/aarch64 variant
- `ghcr.io/rylinjames/reflex-vla:<version>` — existing amd64 image (unchanged)
- Future: `ghcr.io/rylinjames/reflex-vla:<version>` as a multi-arch manifest that auto-selects amd64 or arm64 based on `docker pull`'s host arch. Deferred until both build paths are stable on every release.

## Why not bundle CUDA in the arm64 image

1. **Wheel incompatibility**: `onnxruntime-gpu` pip wheels are pinned to NVIDIA's desktop CUDA 12.x ABIs. JetPack's CUDA (Jetson-specific) has different SONAMEs and driver interface. Installing `onnxruntime-gpu` via pip on Jetson produces a container that loads but crashes on first ORT session.
2. **Image size**: NVIDIA's `nvcr.io/nvidia/l4t-pytorch` base is 4-6 GB before adding reflex-vla. QEMU cross-building that via GitHub Actions takes 30-60 min per release. Not worth it for a general base when the intended pattern is `--runtime=nvidia` mounting host CUDA.
3. **Version pinning lock-in**: bundling JetPack 6.x CUDA locks the image to Jetson devices on that specific L4T version. Users on older/newer JetPack would need to rebuild.

The `:<version>-jetpack` variant (Phase 2) will solve this for customers who explicitly want a bundled-CUDA image.

## Verification path

1. ✅ Goal check passes: `test -f .github/workflows/docker-publish-arm64.yml && grep -q 'arm64' ...`
2. ✅ Unit tests: `pytest tests/test_docker_arm64.py` (12 green).
3. (pending) First tag push or manual workflow dispatch → verify ghcr.io receives the arm64 image.
4. (blocked on hardware) Pull + run on real Jetson Orin Nano → verify reflex serve starts, /health returns 200, /act round-trips an action chunk.

## Rollout

1. Merge this doc + files into `main`.
2. On next release (v0.2.2+), the workflow auto-triggers on `v*` tag push.
3. Update README quickstart to document both `ghcr.io/...-arm64` for Jetson and the unsuffixed tag for amd64.
4. Add a `docker pull + docker run` example specifically for Jetson with `--runtime=nvidia`.

## Related goals

- `docker-image-distribution` (weight 7) — the x86 side of this, already shipping via `docker-publish.yml`.
- `orin-nano-fp16-fit` (weight 8) — closed this session; unlocks pi0 at 6.26 GB fitting Orin Nano 8GB. This docker goal is the shipping-path counterpart.
- `jetson-benchmark-ci` (weight 9) — needs real Jetson hardware; will run `reflex bench` inside the arm64 image once hardware is available.
