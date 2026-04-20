# Fine-tuning Pipeline — Synthesis & Build Plan

**Date**: 2026-04-20
**Status**: Research complete, ready to build
**Parent goal**: `fine-tuning-pipeline` (GOALS.yaml, weight 5)

This doc is the one-page product decision that synthesizes four research reports. Read this first; the reports below provide the evidence trail if you want to pressure-test any claim.

---

## Research reports (read in order if going deep)

| Doc | What it answers |
|---|---|
| [finetune_competitive_research.md](finetune_competitive_research.md) | What's in the market, top 10 customer pain points, 3 differentiation candidates, contrarian takes |
| [finetune_sota_research.md](finetune_sota_research.md) | Technical SOTA, per-model LoRA configs, backend choices, 3-year evolution hypothesis |
| [finetune_roadmap.md](finetune_roadmap.md) | v0.3 / v0.5 / v1.0 horizons, kill-gates, customer forks, monetization cadence |
| [finetune_architecture.md](finetune_architecture.md) | Final picked design (in-process trainer registry), file-level module plan, 11 open product calls |

---

## One-paragraph summary

Reflex is adding `reflex finetune`, a thin orchestrator that wraps `lerobot-train` (and eventually `openpi` JAX) with reflex-specific value: preflight validation, parity-gated training, calibration-first eval, and auto-chain to `reflex export`. The design is an in-process trainer registry with plugin hooks (action head / optimizer / auxiliary loss / norm-stats) — reflex never rewrites gradient math. v0.3 ships a SmolVLA-only LoRA wrapper (2 weeks) that closes the goal check. v0.5 adds pi0, pre-flight validator, parity-gate, and calibration-first eval — the "table stakes" shape that makes reflex distinct from lerobot-train (3 months). v1.0 conditionally commits to the pluggable action-head / multi-backend / orthogonalized platform story with explicit kill-gates. Monetization: free fine-tune forever; Pro gates `--parity-gate` and `--eval calibration`; never sell GPU-minutes.

---

## The decision tree

### What reflex finetune IS

- A **thin orchestrator** over lerobot/openpi/gr00t native trainers
- A **validation layer**: preflight schema check, parity-gate at checkpoint saves, calibration eval on held-out data
- An **auto-export chain**: training finishes → `reflex export` runs → customer gets deployable ONNX + `VERIFICATION.md`
- A **pluggable registry** of action heads, optimizers, auxiliary losses — designed to absorb VLAs that don't exist yet

### What reflex finetune is NOT

- Not a custom training loop (lerobot owns gradient math)
- Not a GPU rental service (customer goes direct to Modal / RunPod)
- Not hyperparameter search (out of scope until v1.0+)
- Not a replacement for sim-to-real pipelines (reflex stays deploy-focused)

---

## Why each design call was made

### 1. Thin orchestrator over native trainer

**Decision**: Wrap lerobot-train in-process; never fork its gradient path.
**Evidence**: Competitive research Section D (contrarian take #4 — "NVIDIA ships one-click GR00T in a future Isaac release"). If we own gradient math, every upstream improvement is our maintenance burden. Thin orchestrator lets us ride upstream.
**Trade-off accepted**: We inherit lerobot's bugs. Mitigation: parity-gate + preflight catch 8 of top-10 customer pains regardless of upstream correctness.

### 2. r=32 LoRA default (not lerobot's r=16)

**Decision**: Default LoRA rank for VLAs is 32. Bump to 64 for GR00T's DiT action head.
**Evidence**: SOTA research Section A + LoRA-SP arxiv 2603.07404. VLA intrinsic rank is 4-16× higher than LLM. Using r=16 on VLAs is the #6 most-frequent customer pain (silent norm-stats mismatch → 96% → 1% task success).
**Trade-off accepted**: 2-4× memory vs lerobot default. Documented in the preflight validator's memory estimate.

### 3. Python profiles, not YAML configs

**Decision**: Training recipes are Python modules (`profiles/smolvla_libero.py`) with conditional logic.
**Evidence**: SOTA research non-obvious finding #2 — pi0.5's `use_quantile_norm=True` silently kills small-dataset runs. YAML can't encode `if len(dataset) < 500: use mean_std`. Python can.
**Trade-off accepted**: Less declarative than YAML. But the conditionals are the whole point.

### 4. Calibration pulled into v0.5 (not v1)

**Decision**: `--eval calibration` ships in v0.5 even though competitive research sequenced it as v2.
**Evidence**: `src/reflex/eval/calibration.py` already exists (closed this session as the `calibration-metrics` goal). Marginal cost of integration ≈ 0. Roadmap agent independently agreed.
**Trade-off accepted**: One more thing in the v0.5 shape. But it's a differentiator vs lerobot-train (which has no calibration) and we already own the code.

### 5. LeRobotDataset v3 as primary format

**Decision**: v0.3+ supports LeRobotDataset v3 only. RLDS / Open X-Embodiment via adapter in v0.5+.
**Evidence**: SOTA research Section A3 — 54.6% adoption in months, ecosystem consolidating. Format war effectively over.
**Trade-off accepted**: We're betting the ecosystem stays on v3. If v4 displaces v3, we absorb via adapter (not a rewrite).

### 6. Parity-gate pulled into v0.5 (from competitive research's v1)

**Decision**: Every checkpoint save auto-runs cos=+1.0 check; training continues only if passed.
**Evidence**: Parity toolchain already shipped (cos=+1.0000000 on 4 VLAs this week). Roadmap agent + architecture agent both independently pulled it forward. It's our moat; not using it in training would be malpractice.
**Trade-off accepted**: Slows training by N seconds per checkpoint. Documented; can be disabled via `--no-parity-gate`.

### 7. pi0.5 / GR00T deferred to v0.6+

**Decision**: v0.5 supports SmolVLA LoRA + pi0 LoRA only. pi0.5 and GR00T full fine-tune land in v0.6+.
**Evidence**: lerobot 0.5.1 can't load GR00T N1.6 (blocker we hit in GR00T Step 3). pi0.5 LoRA has no community-verified config.
**Trade-off accepted**: 2 of 4 supported VLAs don't have fine-tune in v0.5. Documented with `NotImplementedError` stubs + error message pointing customers at native lerobot-train with reflex-export-post-hoc.

### 8. Free fine-tune, Pro-gated validators

**Decision**: `reflex finetune` free and unlimited. `--parity-gate` and `--eval calibration` are Pro-tier features.
**Evidence**: Competitive research Section D contrarian take #5 — "price modestly on finetune, monetize serve/guard/turbo." Customers shopping for fine-tune compare us to lerobot-train (free) + Modal (per-GPU-hour). We can't charge for the orchestration; we can charge for the validation moat that lerobot doesn't have.
**Trade-off accepted**: Gives up the "charge per checkpoint" revenue model. Keeps the adoption flywheel.

---

## What "done" looks like — per horizon

### v0.3 (2 weeks from start, 1-2 engineers)

**Goal check passes**:
```bash
test -f src/reflex/finetune.py
.venv/bin/python -c "from reflex.finetune import run_finetune"
```

**Customer can do**:
```bash
reflex finetune \
    --base lerobot/smolvla_base \
    --dataset lerobot/libero \
    --output ./my_export \
    --steps 2000
# Training runs, checkpoints save, reflex export auto-runs at end,
# ./my_export/ has model.onnx + VERIFICATION.md ready to serve.
```

**End-to-end integration test exists** — 200-step SmolVLA fine-tune on a 50-sample subset, verifies (a) checkpoint saves, (b) export runs, (c) exported ONNX has cos=+1.0 parity to the fine-tuned PyTorch.

**One measured number** lands in `reflex_context/measured_numbers.md` — "SmolVLA fine-tune end-to-end: PyTorch training, auto-export, cos=+1.0 parity preserved through fine-tune+export chain."

### v0.5 (3 months from start)

Adds on top of v0.3:
- pi0 supported
- Pre-flight validator catches 4 of top-10 pain points before training kicks off
- Parity-gate runs at every checkpoint save
- `--eval calibration` produces ECE/Brier/NLL on held-out data
- Pro tier gating in place
- Documented in `docs/getting_started.md` fine-tune section

### v1.0 (6-12 months, CONDITIONAL)

Three kill-gates (if any fire, stop / re-scope):
1. End of v0.5: if orthogonalized matrix cell-count <10 (across base × embodiment × edge-HW) → scrap the matrix framing
2. Mid-v1.0: if zero external PRs to the plugin registry → drop "platform" positioning
3. Anytime: if NVIDIA ships one-click GR00T fine-tune → drop humanoid column, refocus on Franka/SO-100/ALOHA

Full scope: pluggable action heads, multi-backend (lerobot + openpi-JAX), cross-matrix dashboards, Enterprise SLA tier.

---

## Forward-compatibility commitments

The architecture doc has 10 specific evolutions we absorb without rewrites. Most important:

| Future VLA evolution | How we absorb |
|---|---|
| VLM2VLA arrives (action-as-language) | Pluggable action-head registry; new head is a plugin, not a rewrite |
| Muon / Sophia become standard | Optimizer registry; new optimizers are plugins |
| Training-time quantization becomes standard | QAT hook in the export chain (already have fp16_convert.py) |
| New dataset format displaces LeRobotDataset v3 | Dataset adapter layer; v4 becomes a new adapter, not a rewrite |
| 100B+ VLA lands | FSDP / DeepSpeed hook on the backend registry |
| RLHF-style post-training for VLAs | Auxiliary loss registry absorbs it |

---

## Open product calls (defaults picked, revisit when evidence lands)

From the architecture doc, 11 concrete A/B questions with picked defaults:

1. **Preflight on warnings**: dry-run (fail-safe) vs continue. **Default**: dry-run. Customers can `--force` to bypass.
2. **Parity-gate failure**: stop vs warn. **Default**: stop (can be overridden with `--force-ship`).
3. **Calibration-gate failure**: warn vs stop. **Default**: warn + `--force-ship` flag.
4. **LoRA rank**: 16 vs 32 vs 64. **Default**: 32 (GR00T-DiT: 64 when v0.6+ lands).
5. **Default backend for pi0**: lerobot vs openpi-JAX. **Default**: lerobot in v0.5; openpi optional via `--backend openpi`.
6. **Report location**: checkpoint-dir vs separate /reports. **Default**: separate `<output>/reports/` dir.
7. **Dataset streaming**: preload vs stream. **Default**: stream for dataset >2GB.
8. **FSDP vs DDP**: auto-detect vs explicit flag. **Default**: DDP up to 2 GPUs, FSDP at 4+, flag overrides.
9. **Checkpoint interval**: steps vs epochs vs time. **Default**: steps, every N where N = max(100, steps/20).
10. **VERIFICATION.md includes training log pointer**: yes/no. **Default**: yes, with last-20-lines of logs inline.
11. **Custom profiles**: user-supplied Python vs preset-only. **Default**: user-supplied allowed; preset-only if `--safe-mode`.

---

## Risk register (from architecture doc, ranked)

1. **lerobot API breakage** (High likelihood / High severity). Mitigation: pin lerobot to a known-good version (0.5.1). Escape hatch: subprocess wrapper (rejected Proposal 1) available if lerobot goes private-API.
2. **Customer dataset schema variance** (High / Medium). Mitigation: preflight validator + clear error messages. Supports top-3 schemas out of the box, adapter layer for others.
3. **FP16 conversion fails post-fine-tune** (Medium / Medium). Mitigation: auto-retry with fallback conversion config (Cast-insertion pass already handles most cases).
4. **GR00T compat gap** (High / Low — known since Step 3). Mitigation: documented NotImplementedError + pointer to lerobot + post-hoc reflex-export workflow.
5. **Upstream VLA adds new action head we can't serve** (Medium / Low — VLM2VLA is 2027+). Mitigation: pluggable action-head registry designed for this.

---

## What ships WHEN (the only calendar that matters)

- **Week 1-2** from start: v0.3 MVP. `reflex finetune` works for SmolVLA end-to-end. Goal check passes. One measured_numbers row.
- **Week 3-12**: v0.5 — pre-flight, parity-gate, calibration, pi0 support, Pro tier gating.
- **Month 6-12**: v1.0 (conditional). Pluggable / matrix / SLA.

---

## Ready to build v0.3

Next step: implement `src/reflex/finetune.py` with `run_finetune()` + `reflex finetune` CLI + `scripts/modal_reflex_finetune.py` + 1 end-to-end integration test.

File inventory target (see architecture doc Section D for detail):
```
src/reflex/
├── finetune.py              — Python API (run_finetune, TrainerRegistry, profile loader)
├── cli.py                   — add @app.command finetune
├── finetune_profiles/       — default recipes per model
│   ├── __init__.py
│   └── smolvla_libero.py    — first concrete profile
└── finetune_validators/     — preflight (v0.5, stub in v0.3)
    └── __init__.py

scripts/
└── modal_reflex_finetune.py — Modal scaffold

tests/
└── test_finetune.py         — unit tests + one integration test
```

v0.3 shipping bar:
- `reflex finetune --base lerobot/smolvla_base --dataset <any-lerobot-v3-dataset> --output <dir>` runs end-to-end
- ONNX in `<dir>/model.onnx` loads and serves via `reflex serve <dir>`
- Test fixture: smolvla on a 50-sample LIBERO subset, 200 steps, cos=+1.0 preserved through fine-tune+export

Everything else (parity-gate, calibration, preflight, pi0, Pro tier) is v0.5+.
