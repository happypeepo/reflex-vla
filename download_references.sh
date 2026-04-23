#!/usr/bin/env bash
# Clone reference repos for serve design work.
# See reflex_context/01_decisions/2026-04-22-reference-repos-folder.md.
#
# Usage:  bash download_references.sh
#
# Idempotent — skips repos that already exist. To refresh, delete the
# subdir and re-run.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p reference
cd reference

clone_if_missing() {
  local name="$1"; local url="$2"
  if [[ -d "$name" ]]; then
    echo "  ✓ $name already cloned (skip)"
  else
    echo "  → cloning $name from $url"
    # GIT_LFS_SKIP_SMUDGE=1 avoids failing when git-lfs isn't installed
    # (some repos like Triton have LFS-tracked test fixtures we don't need
    # for source reference).
    # Don't fail the whole script if one clone breaks — just warn and continue.
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$url" "$name" || \
      echo "    ⚠ $name clone failed — continuing"
  fi
}

echo "[reference] cloning competitor OSS repos (shallow, --depth 1)…"

# Inference servers
clone_if_missing triton                  https://github.com/triton-inference-server/server.git
clone_if_missing vllm                    https://github.com/vllm-project/vllm.git
clone_if_missing tgi                     https://github.com/huggingface/text-generation-inference.git
clone_if_missing ray                     https://github.com/ray-project/ray.git
clone_if_missing trtllm                  https://github.com/NVIDIA/TensorRT-LLM.git

# VLA reference impls
clone_if_missing lerobot                 https://github.com/huggingface/lerobot.git
clone_if_missing openpi                  https://github.com/Physical-Intelligence/openpi.git

# Robot embodiment configs (URDF / MJCF) — for per-embodiment-configs (B.1)
clone_if_missing mujoco_menagerie        https://github.com/google-deepmind/mujoco_menagerie.git

# Record/replay + OTel GenAI tracing backend — for record-replay (B.2)
clone_if_missing phoenix                 https://github.com/Arize-ai/phoenix.git

# Robot policy server prior art (gRPC + 1kHz control loop) — serve patterns
clone_if_missing fairo                   https://github.com/facebookresearch/fairo.git

# ACT — the chunking primitive A2C2 corrects against (B.4 + B.5)
clone_if_missing act                     https://github.com/tonyzhaozh/act.git

echo ""
echo "[reference] done. Total size:"
du -sh . 2>/dev/null || true

if [[ ! -f NOTES.md ]]; then
  cat > NOTES.md <<'EOF'
# Reference Notes

Lookup-first log. When you grep for a pattern in `reference/` and find
something useful, jot a one-liner here so future-you doesn't have to
re-grep. Format:

- `<repo>/<path>:<line>` — what it does, when we'd copy it

## Examples

- `vllm/vllm/core/block_manager_v2.py` — paged-attention KV cache. Read
  before designing episode-aware prefix cache.
- `vllm/vllm/entrypoints/openai/api_server.py` — lifespan + warmup + SIGTERM
  drain pattern. Reference for prewarm-crash-recovery (Phase 0.5).
- `lerobot/lerobot/policies/pi05/processing_pi05.py` — pi0.5
  preprocessor. State-in-lang behavior is in PI05PrepareTokenizerStep.
- `lerobot/lerobot/common/robot_devices/control_utils.py` — action/observation
  schemas. Reference for dataset-validator (Phase 0.5).
- `openpi/src/openpi/policies/pi0_pytorch.py` — canonical pi0 forward.
- `openpi/src/openpi/policies/` — RTC overlap logic. Reference for B.3.
- `triton/src/core/dynamic_batch_scheduler.cc` — Triton's batching
  scheduler. Reference for our continuous-batching work.
- `mujoco_menagerie/franka_emika_panda/`, `.../so_arm100/`, `.../universal_robots_ur5e/` —
  canonical URDF/MJCF for embodiments we target. Reference for B.1.
- `phoenix/src/phoenix/trace/` — OTel span ingestion + replay primitives.
  Reference for B.2 record-replay.
- `fairo/polymetis/` — the only battle-tested robot policy server worth
  dissecting. Reference for serve lifecycle + 1kHz control loop patterns.
- `act/detr/models/detr_vae.py` — the chunk-predicting policy architecture
  A2C2 corrects against. Reference for B.4 + B.5.

(Add yours below)

EOF
fi

echo ""
echo "[reference] NOTES.md ready at reference/NOTES.md"
echo "[reference] grep workflow:  grep -r '<concept>' reference/ | head"
