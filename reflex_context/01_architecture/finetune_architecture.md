# `reflex finetune` — Architecture

**Date:** 2026-04-19
**Author:** architecture agent
**Status:** Opinionated. One recommended architecture. Senior engineer should be able to implement from this without follow-up questions.
**Missing input:** `finetune_roadmap.md` not present at design time. v0.5 target shape inferred from the research docs + GOALS.yaml `fine-tuning-pipeline` (weight 5). If roadmap lands with a divergent v0.5 definition, Section D is where the deltas belong.

---

## Section A — Design axioms

These are the non-negotiable design constraints. Everything below must satisfy all of them.

**A1. Thin orchestrator over lerobot / openpi / gr00t. Never rewrite gradient math.**
- Source: competitive research section D red-flag #4 ("One pi0.6 or xVLA release and your training wrapper is stale"); SOTA research section A4 ("lerobot-train PEFT support landed in v0.4.4; training is fine, deployment is broken — exactly reflex's wedge").
- Consequence: we do NOT implement Adam step code, flow-matching loss, FSDP config translation, or gradient checkpointing ourselves. We delegate to `lerobot-train` for PyTorch backbones and `openpi` for JAX. The reflex finetune module is a pre-flight + execution-monitor + post-export shell.
- What this forces us NOT to do: no `reflex.train.loop`, no reimplementation of PeftConfig, no custom optimizer wrappers outside registry plumbing, no bespoke rank-adaptive LoRA math.

**A2. Validation is the moat. Every run auto-produces a parity-gate + calibration report.**
- Source: competitive research pain #1 ("Loss converged, wandb looked fine, evaluation was 0%") — the single most-complained-about failure mode; SOTA research section A6 + Zollo 2025; competitive research C1 + C2 differentiators.
- Consequence: `reflex finetune` without `--no-verify` flag always runs the same ONNX parity harness + calibration pass + VERIFICATION.md render that `reflex export` + `reflex validate` produce today. Training is "done" only when these artifacts exist on disk.
- What this forces us NOT to do: no "quick train" shortcut that skips validation; no separate `reflex verify-finetune` command that customers can forget to run.

**A3. Pluggable action-head layer. We do NOT hardcode flow-matching.**
- Source: SOTA research section C1 ("pluggable action heads"); finding 3 (OpenVLA-OFT L1 regression may deprecate flow-matching in 2 years); finding 6 (VLM2VLA action-as-language).
- Consequence: the `ActionHeadStrategy` abstraction is first-class at the config layer. Loss function, normalization mode, sampling_steps, whether the head is head-of-VLM-tokens vs. separate-expert vs. DiT — all resolved by strategy. New head types slot in without touching train loop code.
- What this forces us NOT to do: no `if model_type == "smolvla": flow_matching_loss(...)` branches in the training code path. No assumption that every VLA has a denoising loop at all.

**A4. Pre-flight schema validation runs BEFORE any GPU time.**
- Source: competitive research pains #1, #2, #6 ("shape mismatch 6 vs 7", "LoRA tanked 96%→1% from norm-stats mismatch"), pains #5, #9 ("OOM on dual 32GB", "How much data is enough").
- Consequence: `reflex finetune --dry-run` is first-class and runs: checkpoint loading, dataset schema validation vs checkpoint action/state dims, memory budget computation, norm-stats provenance check (are we reusing base checkpoint stats or recomputing? log both), data-size floor check (warn if below per-base-model threshold from finding 9). Full command runs dry-run implicitly unless `--skip-preflight`.
- What this forces us NOT to do: no "just start training and fail at step 42" behavior; we surface every knowable problem before costing the customer a minute of H100.

**A5. Dataset format is LeRobotDataset v3 — native. Everything else is an adapter.**
- Source: SOTA research section A3 (54.6% HF adoption, format war is over); competitive research pain #4 (v2→v3 broke pipelines).
- Consequence: ingest layer is `reflex.data.ingest` and always emits the canonical `TrajectorySample(obs, state, action, lang, meta)` stream. RLDS, Open X-Embodiment, future formats register as adapters. Training code never sees parquet; it sees `TrajectorySample`.
- What this forces us NOT to do: no parquet I/O in the training module; no direct coupling between HF datasets library and action-head code.

**A6. Export-on-success, not export-as-separate-step.**
- Source: competitive research C1 ("From demonstrations to deployed ONNX in one command"); GOALS.yaml `fine-tuning-pipeline` description ("wraps lerobot's existing training loops + auto-exports the fine-tuned checkpoint").
- Consequence: successful finetune auto-invokes `reflex.exporters.monolithic.export_monolithic(...)` (already the verified production path) on the final checkpoint. A single parity-gate run follows. No `reflex export` command required between finetune and serve.
- What this forces us NOT to do: no customer-visible "here's a checkpoint, now go figure out export yourself" state. A finetune that doesn't reach a cos-verified ONNX is marked FAILED, not SUCCEEDED-but-unexported.

**A7. Training backend identity is a single enum, not a tree of branches.**
- Source: SOTA research A4 decision ("Wrap lerobot-train + openpi JAX. bf16 default."); maintenance burden risk.
- Consequence: `TrainingBackend` enum with exactly three initial values — `LEROBOT`, `OPENPI_JAX`, `HF_TRANSFORMERS_TRAINER` (last one only for future OpenVLA-style custom-trainer models). Auto-selected by base-model type; overrideable via `--backend`. Every backend implements the same `fit()` contract so the calling code doesn't care.
- What this forces us NOT to do: no per-backend CLI flags leaking into the top-level. No `if backend == "lerobot": subprocess.run(...) elif backend == "openpi": jax.train_loop(...)` scattered across files — that logic belongs behind the `Backend` interface.

---

## Section B — Three architecture proposals

### Proposal 1 — Config-as-code wrapper (subprocess lerobot-train)

**Architectural summary:**
- `reflex finetune` is primarily a draccus-config generator + subprocess launcher.
- Reflex writes a `lerobot_config.draccus` tailored to the chosen base model, wedges a pre-flight check ahead of it, subprocess-invokes `lerobot-train` (or `openpi train.py`), watches stdout for metric events, post-processes the final checkpoint into `reflex export`.
- Heavy reliance on upstream's CLI; minimal Python in `src/reflex/finetune/`.
- Training metrics come through parsed stdout (brittle) or wandb API (cleaner).

**Module tree:**
```
src/reflex/finetune/
  __init__.py
  cli.py                  # typer entrypoint
  orchestrator.py         # the subprocess shell
  preflight.py            # schema + memory + norm-stats checks (pure-Python, cheap)
  backends/
    lerobot_config_gen.py # emits draccus YAML for lerobot-train
    openpi_config_gen.py  # emits openpi config.py snippet
  monitor.py              # stdout tail / wandb poll for live metrics
  postprocess.py          # checkpoint -> export_monolithic -> VERIFICATION.md
  profiles/
    smolvla_lora.yaml     # reproducibility presets (pain #3)
    pi0_lora.yaml
    pi05_lora.yaml
    gr00t_full.yaml
```

**Data flow:** customer runs `reflex finetune --base smolvla --dataset <hf_id> --output ./out`. `cli.py` → parse → `preflight.run()` → `lerobot_config_gen.render(base, dataset, out)` writes `./out/lerobot_config.draccus` → `orchestrator.run_subprocess(["lerobot-train", "--config-path", ...])` → `monitor.tail()` publishes step/loss/lr to console + training_log.jsonl → on success, `postprocess.auto_export(checkpoint)` → `reflex.exporters.monolithic.export_monolithic()` → `reflex.verification_report.write_verification_report()` → exit.

**Pluggability:** new VLA = add a profile yaml + a config-generator function. Training code is entirely lerobot/openpi's problem.

**Interop:**
- `monolithic.py` — called directly from `postprocess.auto_export()`.
- `verification_report.py` — called after export as today.
- `calibration.py` — called in a mandatory calibration phase after export. **Requires extending calibration.py with a mid-training hook (small lift).**
- `fp16_convert.py` — reused as-is when target requires FP16.

**Effort:** 2 engineer-weeks to v0.5. Preflight is new code but small; most effort is draccus config templates + monitor parsing.

**Risk profile:**
1. **lerobot-train stdout format changes** (severity: medium, likelihood: HIGH). Every lerobot minor release can change log lines. Monitor parser breaks silently. Mitigation: prefer wandb API over stdout; pin a tested lerobot version per reflex release.
2. **Subprocess isolation loses Python-level plugin hooks** (severity: HIGH, likelihood: certain). When calibration wants to peek at model state mid-training, or when we later add an auxiliary loss registry (SOTA C5), subprocess boundary blocks this. Requires refactor to Proposal 2 shape.
3. **Draccus churn** (severity: low, likelihood: medium). If HF swaps config framework or changes key names, profile YAMLs silently produce wrong configs.

**Death scenarios:**
1. An auxiliary-loss / world-model co-training pattern (SOTA C5 + arxiv 2512.18007) becomes standard. Subprocess wrapper can't inject extra loss terms without upstream support — we would fork lerobot-train.
2. Customers ask for mid-training sim-eval (competitive research pain #8) — subprocess boundary prevents clean in-process rollout; we'd spawn a separate eval subprocess and coordinate via filesystem.
3. Runtime QAT (SOTA C4) arrives, requiring fake-quant ops injected during training. Subprocess-level we can't. Requires in-process plugin layer, meaning Proposal 2.

---

### Proposal 2 — In-process trainer registry with plugin hooks (RECOMMENDED)

**Architectural summary:**
- Reflex imports lerobot + openpi as libraries, not subprocesses.
- `Trainer` abstraction is a thin facade over upstream's core training loop, exposing named extension points: `before_step`, `after_step`, `on_checkpoint`, `loss_modifier`, `action_head_factory`.
- Base action-head, optimizer, auxiliary-loss, norm-stats strategies live in per-concern registries (`action_heads/`, `auxiliary_losses/`, `optimizers/`, `norm_stats/`).
- Training still uses lerobot's optimizer step, FSDP, and action-head forward — we just sit above them.
- Mid-training callbacks can compute ECE, run LIBERO smoketests, write intermediate VERIFICATION.md drafts.

**Module tree:**
```
src/reflex/finetune/
  __init__.py
  cli.py                         # typer entrypoint
  run.py                         # public API: run_finetune(FinetuneConfig) -> FinetuneResult
  config.py                      # FinetuneConfig dataclass (draccus-compatible)
  profiles/                      # reproducibility presets (pain #3)
    smolvla_lora.py              # Python not YAML so profiles can have logic
    pi0_lora.py
    pi05_lora.py
    gr00t_full.py
    __init__.py                  # registry of named profiles
  preflight/
    __init__.py
    schema.py                    # dataset-vs-checkpoint schema validation (pain #1/#2)
    memory.py                    # pre-flight memory budget (pain #5)
    norm_stats.py                # detect + reuse base-checkpoint norm-stats (pain #6)
    dataset_size.py              # learning-curve extrapolation (pain #9)
  backends/
    __init__.py                  # TrainingBackend enum + resolve_backend()
    base.py                      # Backend protocol: fit(trainer_ctx) -> CheckpointResult
    lerobot_backend.py           # wraps lerobot.scripts.train.train(cfg)
    openpi_backend.py            # wraps openpi JAX training loop
    hf_backend.py                # wraps transformers Trainer (future: OpenVLA, VLM2VLA)
  action_heads/
    __init__.py                  # ActionHeadStrategy registry
    flow_matching.py             # wraps flow-matching loss from lerobot
    l1_regression.py             # OFT-style regression head
    fast_tokens.py               # pi0-FAST / tokenized action head
    ddpm_dit.py                  # GR00T-style DDPM DiT head
    # Future: action_language.py (VLM2VLA)
  optimizers/
    __init__.py                  # registry — default AdamW, accepts muon / sophia
    adamw.py
    # Future: muon.py, sophia.py
  auxiliary_losses/
    __init__.py                  # registry — e.g., rehearsal-shard replay, world-model
    rehearsal.py                 # SOTA finding 7 — 6.2% mix
    # Future: motion_image.py (arxiv 2512.18007)
  norm_stats/
    __init__.py
    reuse_base.py                # the pain #6 fix: reuse base-checkpoint stats for pretrained dims
    compute_deltas.py            # only compute new stats for new dims
  hooks/
    __init__.py                  # HookRegistry: before_step, after_step, on_checkpoint, ...
    calibration_hook.py          # invokes reflex.eval.calibration every N steps
    libero_smoketest_hook.py     # run LIBERO on checkpoint save
    export_on_checkpoint_hook.py # auto-run reflex export + parity gate
    wandb_hook.py
    val_loss_hook.py             # pain #8 — openpi missing val loss
  monitor.py                     # training-log.jsonl writer (uniform across backends)
  postprocess.py                 # final-checkpoint -> monolithic export -> calibration -> VERIFICATION.md
  templates/
    modal_finetune.py            # scaffold script for Modal execution (SOTA A7)
    runpod_finetune.sh           # scaffold for RunPod
```

**Data flow:**
1. `reflex finetune --base smolvla --dataset lerobot/libero-10 --output ./out` → `cli.py`.
2. `cli.py` → `FinetuneConfig.from_cli(args)` → `run_finetune(cfg)`.
3. `run.py` → `preflight.run(cfg)` → `schema.py` validates dataset vs checkpoint; `memory.py` computes budget; `norm_stats.py` resolves reuse-vs-recompute; `dataset_size.py` reports floor.
4. Preflight passes (or fails loudly with concrete fix) → `backends.resolve_backend(cfg.base)` returns `LerobotBackend()`.
5. Backend receives a `TrainerContext(config, hooks, action_head_strategy, optimizer_spec, aux_losses, norm_stats)` and calls lerobot's internal training loop with hooks wired in.
6. Per-step hooks fire (wandb, val loss, optional calibration-every-N, optional LIBERO-every-N).
7. On final checkpoint: `postprocess.finalize()` → `export_monolithic()` → `validate_roundtrip()` (cos-parity) → `calibration.run_on_holdout()` → `verification_report.write(parity=..., calibration=...)`.
8. Exit code 0 only if parity passes AND calibration ECE < cfg.calibration_gate (default 0.2).

**Pluggability:** new VLA architecture = register action-head strategy + optional new backend + new profile. New dataset format = new `data.ingest` adapter. New optimizer = register in `optimizers/`. New auxiliary loss = register in `auxiliary_losses/`. None of these touch training-loop code, CLI code, or postprocess code.

**Interop:**
- `monolithic.py` — `postprocess.finalize()` calls `export_monolithic()` directly. Same signature, same verified artifacts.
- `verification_report.py` — extended with a `calibration=...` kwarg (small PR) to emit a combined Parity+Calibration section. Already accepts `parity=...` today.
- `calibration.py` — `hooks/calibration_hook.py` calls `compute_ece`/`compute_brier`/`compute_nll` as pure functions. `postprocess.finalize()` runs a full held-out calibration pass. The `CALIBRATION.md` referenced in calibration.py's docstring becomes a section of VERIFICATION.md instead (one artifact, not two).
- `fp16_convert.py` — invoked from `postprocess.finalize()` when target hw requires FP16.

**Effort:** 3 engineer-weeks to v0.5. Week 1: preflight + SmolVLA LoRA lerobot backend + end-to-end happy path. Week 2: pi0 + pi0.5 lerobot support, openpi JAX backend (stub), hooks, calibration integration. Week 3: GR00T full-finetune backend, Modal templates, reproducibility profiles, test suite, docs.

**Risk profile:**
1. **lerobot internal API drift** (severity: medium, likelihood: HIGH). Importing `lerobot.scripts.train.train` couples us to private-ish API. Mitigation: pin lerobot version in `[finetune]` extra, CI matrix tests against pinned + latest, break loudly not silently.
2. **Hook ordering / side-effect bugs** (severity: medium, likelihood: medium). Callbacks that compute ECE on partially-trained LoRA weights can produce misleading numbers that customers over-read. Mitigation: label mid-training metrics "in-flight" vs final; gate the calibration gate on the *final* checkpoint only.
3. **openpi JAX + lerobot PyTorch dual-backend maintenance tax** (severity: medium, likelihood: certain). Two code paths means two sets of bugs. Mitigation: v0.5 ships lerobot backend complete; openpi backend is a v0.6 item with a clear `NotImplementedError` stub. Customers who *need* native openpi training in v0.5 fall back to subprocess (Proposal 1 escape hatch).

**Death scenarios:**
1. A VLA arrives whose "training" isn't gradient descent on a loss at all — e.g., pure RL post-training with environment simulator, or test-time-training only. Our hook registry assumes a loss-step shape. We'd need a phase abstraction above Trainer. Mitigable: SOTA C5 already anticipates `--phase imitation|rl|distill`. Not fatal.
2. A 100B+ VLA lands that needs 3D parallelism (TP + PP + DP) and can't run through lerobot's single-node FSDP. We'd need to bring in DeepSpeed or FSDP-2 / device-mesh. This is a real risk but the fix is adding a backend, not replacing the architecture.
3. Upstream lerobot fundamentally restructures its trainer (e.g., removes in-memory Python API in favor of config-only subprocess). We'd fall back to Proposal 1 mode. Not fatal — our preflight / postprocess / calibration layer survives untouched.

---

### Proposal 3 — State-machine orchestrator with step-level backend calls

**Architectural summary:**
- Reflex owns the training loop explicitly as a state machine: `(LOAD_CHECKPOINT, LOAD_DATA, SETUP_OPTIMIZER, TRAIN_STEP, EVAL_STEP, CHECKPOINT, FINALIZE)`.
- lerobot / openpi are called at step granularity — we get a batch, call backend's `forward_and_loss(batch)`, get a loss tensor, call backend's `backward_and_step(loss)`, etc.
- Maximum control, maximum plugin surface, maximum maintenance burden.
- Customer-visible state at every step; rich observability by construction.

**Module tree:** similar to Proposal 2 but with an additional `loop/` package owning the state machine explicitly (`loop/state_machine.py`, `loop/train_step.py`, `loop/eval_step.py`, `loop/checkpoint_step.py`) and a much richer `backends/` surface area (per-backend `forward`, `backward`, `step`, `zero_grad`, `state_dict`, `load_state_dict`).

**Data flow:** CLI → preflight → state machine init → `while not done: state = state.next(backend)`. Each state transition is a discrete, loggable, testable step.

**Pluggability:** every abstraction is a plugin. Ultimate flexibility.

**Interop:** same as Proposal 2 — postprocess still calls `export_monolithic`, etc.

**Effort:** 6+ engineer-weeks to v0.5. Each backend needs per-step translation (lerobot's FSDP-wrapped forward is not a pure function — we would need to simulate FSDP ourselves or wrap it carefully). Much higher test-matrix cost.

**Risk profile:**
1. **We write bugs in the gradient-step path** (severity: HIGH, likelihood: HIGH). Violates axiom A1. This is the most damning issue.
2. **Per-backend step-level adapters rot on every upstream release** (severity: HIGH, likelihood: certain). Much higher surface area than Proposal 2.
3. **Over-engineering for customer need** (severity: medium, likelihood: certain). Customers want "I fine-tuned SmolVLA and it deployed," not "I have 47 hook points in my training loop."

**Death scenarios:**
1. FSDP / FSDP-2 / DTensor API churn makes our per-step backend adapter impossible to maintain — we'd silently lose distributed training correctness.
2. JAX/PyTorch interop at step level forces us to build an intermediate IR just for reflex. This is not what reflex is.
3. Any complex new training regime (masked reconstruction + flow-matching + RL all in one training run) requires state-machine states we didn't anticipate. Proposal 2 absorbs this with new hooks; Proposal 3 needs new states.

**Verdict:** Proposal 3 violates axiom A1. Rejected for v0.5. It is, however, a useful reference for what *not* to do when we're tempted to "take more control."

---

## Section C — Recommended architecture

**Pick: Proposal 2 (in-process trainer registry with plugin hooks).**

### Why it's best for the 3-year horizon

- **Axiom A1 satisfied** — we import lerobot's training code, we don't rewrite it. When pi1 ships and lerobot v1 supports it, we get pi1 support by bumping the pin + adding a profile + adding an action-head strategy if the head is novel. When xVLA / GR00T N2 ship, same path.
- **Axiom A3 natively absorbed** — the `action_heads/` registry is the exact abstraction SOTA research section C1 describes. When VLM2VLA becomes standard (finding 6), we add `action_language.py` and register it; nothing else moves. When OpenVLA-OFT's L1 regression becomes dominant (finding 3), we add `l1_regression.py` and it composes with any backend that supports a regression loss path.
- **Axiom A2 (validation moat) structurally cemented** — the `hooks/` layer + `postprocess.finalize()` is not a layer a customer can forget to use. No `reflex verify-finetune` command exists to be skipped.
- **Pre-flight is sharable real estate** — Proposal 1's preflight.py is the same preflight.py. The win of Proposal 2 is that the preflight isn't the only Python we write; we get to run hooks during training that subprocess can't.
- **Orthogonalization (competitive research C3)** — `backends × profiles × action_heads × optimizers × auxiliary_losses × norm_stats` is a literal cross-product. Third parties (or future reflex releases) add a row/column via PR without touching existing code.
- **Escape hatch to Proposal 1** — if lerobot becomes un-importable (deliberately private), the `LerobotBackend` drops to a subprocess invocation. Hooks degrade to stdout+wandb parsing; no other layer changes. Survival path.

### Compromises we accept

- **Dual-backend maintenance cost (openpi JAX + lerobot PyTorch).** Acceptable because SOTA research A1 says pi0/pi0.5 have openpi-native LoRA recipes that outperform community-reproduced ones. We want pi customers well-served.
- **Tight-ish coupling to lerobot internals.** Mitigable via version pin + CI. Not worse than the existing monolithic.py's pin on `transformers==5.3.0`.
- **Not solving cross-embodiment at v0.5.** SOTA research A1 calls out r=128 / LoRA-SP for cross-embodiment; SOTA D#4 honestly flags we don't know actual success rate gap. We ship `--mode lora-cross-embodiment` as a stub that logs "not yet tested" and defer validation. Customer-visible message: "cross-embodiment mode is experimental."

### Compromises we later undo

- **Calibration-on-flow-matching is "inventing methodology" per SOTA D#3.** For v0.5 we publish ECE numbers on SmolVLA/pi0/pi0.5 with a clear "bin-discretization caveat" in VERIFICATION.md. When the community methodology stabilizes (or when we publish our own validation), we remove the caveat. Undo timing: when either a paper validates or we ship our own empirical validation against real-task success.
- **openpi backend is `NotImplementedError` in v0.5.** Undo when pi0 customers explicitly request native JAX training (vs. lerobot's pi0 port). We have signal already — the openpi upstream val-loss gap (competitive pain #8) tells us openpi users will come asking. Undo timing: v0.6.
- **GR00T LoRA is absent at v0.5.** SOTA B#5 flags that NVIDIA ships no official LoRA for N1.6. We ship only `--mode full` for GR00T. When a community LoRA recipe emerges, or when we do the 1-2 weeks of recipe-hunting (SOTA D#1), we add a `gr00t_lora` profile. Undo timing: v0.7.
- **`--estimate` (learning-curve sweep) is a separate command.** Per competitive pain #9. v0.5 ships `reflex finetune --dry-run` with dataset-size warnings from `preflight/dataset_size.py`; the full learning-curve sweep command ships later. Undo timing: v0.7.

---

## Section D — Concrete module/file plan for the v0.5 shape

Target: a senior engineer implementing from this list, in the order listed, reaches a shippable v0.5 in 3 engineer-weeks.

**Note on v0.5 scope:** absent `finetune_roadmap.md`, v0.5 is defined as: SmolVLA LoRA + pi0 LoRA end-to-end with parity gate + calibration gate + VERIFICATION.md. GR00T full + pi0.5 LoRA land in v0.6. openpi JAX backend lands in v0.6.

### CLI subcommand structure

New subcommand on `src/reflex/cli.py` (one new `@app.command()`):

```
reflex finetune <base> --dataset <hf_id> --output <dir>
  [--mode lora|lora-cross-embodiment|full]       # default: lora
  [--profile smolvla-libero|pi0-aloha|...]       # named reproducibility preset
  [--backend lerobot|openpi|hf]                  # auto-selected from base
  [--num-steps 20000]                            # training steps
  [--batch-size 8]
  [--learning-rate 1e-4]
  [--rehearsal-fraction 0.0]
  [--freeze-vision-encoder/--no-freeze-vision-encoder]
  [--action-norm-mode auto|quantile|mean-std]
  [--precision bf16|fp32]
  [--gpus N]
  [--target orin-nano|orin|thor|desktop]         # export-side target
  [--calibration-gate 0.2]                       # ECE threshold
  [--parity-gate on|off]                         # default: on
  [--dry-run]                                    # preflight only, no GPU
  [--skip-preflight]                             # danger flag for CI
  [--no-export]                                  # finetune only, skip export
  [--no-verify]                                  # training only, skip parity+calibration
  [--modal-gpu a100-80gb]                        # emit modal template, don't run
  [--resume <checkpoint_dir>]
```

Plus:
```
reflex finetune --estimate <base> --dataset <hf_id>  # v0.7, stubbed
```

### Python API (public)

```
src/reflex/finetune/__init__.py:
    run_finetune(cfg: FinetuneConfig) -> FinetuneResult
    FinetuneConfig (dataclass)
    FinetuneResult (dataclass with: export_dir, parity_summary, calibration_summary, training_log_path)
```

### New Python modules / files

All paths relative to `/Users/romirjain/Desktop/building projects/reflex-vla/`.

| File | Purpose |
|---|---|
| `src/reflex/finetune/__init__.py` | Public API re-exports: `run_finetune`, `FinetuneConfig`, `FinetuneResult`. |
| `src/reflex/finetune/cli.py` | Typer command body; parses args, builds `FinetuneConfig`, calls `run_finetune`. Registered on `src/reflex/cli.py` via `from reflex.finetune.cli import finetune_command`. |
| `src/reflex/finetune/config.py` | `FinetuneConfig` dataclass + `from_cli()` + draccus compatibility + JSON schema serialization (forward-compat per SOTA D#9). |
| `src/reflex/finetune/run.py` | `run_finetune()` orchestration: preflight → backend.fit → postprocess.finalize. Thin. |
| `src/reflex/finetune/preflight/__init__.py` | `run_preflight(cfg) -> PreflightReport` — runs all checks, emits structured report. |
| `src/reflex/finetune/preflight/schema.py` | Validates dataset action/state dims vs base checkpoint; detects 6-vs-7 DoF (pain #2); validates image keys. |
| `src/reflex/finetune/preflight/memory.py` | Pre-flight VRAM budget per mode (full / lora / lora-cross-embodiment) × nGPU × precision. Emits "fits / OOM risk / refuse + rent $X". |
| `src/reflex/finetune/preflight/norm_stats.py` | Detects base-checkpoint norm-stats, resolves reuse vs recompute, logs both in a `norm_stats_provenance.json` (pain #6 answer). |
| `src/reflex/finetune/preflight/dataset_size.py` | Warn if episode count below per-base-model floor (SOTA finding 9 — pi0.5 needs ~1000; SmolVLA fine on 50-200). |
| `src/reflex/finetune/backends/__init__.py` | `TrainingBackend` enum + `resolve_backend(base: str) -> Backend`. |
| `src/reflex/finetune/backends/base.py` | `Backend` Protocol: `.fit(trainer_ctx: TrainerContext) -> CheckpointResult`. |
| `src/reflex/finetune/backends/lerobot_backend.py` | Wraps `lerobot.scripts.train.train` as a callable. Wires hooks via lerobot's callback API (or monkey-patches the train loop if needed — prefer not). |
| `src/reflex/finetune/backends/openpi_backend.py` | v0.5: `raise NotImplementedError("openpi JAX backend lands in v0.6; fall back to --backend lerobot")`. |
| `src/reflex/finetune/backends/hf_backend.py` | v0.5: skeleton + stub. Intended for future OpenVLA / VLM2VLA. |
| `src/reflex/finetune/action_heads/__init__.py` | `ActionHeadStrategy` registry + default resolution by base model. |
| `src/reflex/finetune/action_heads/flow_matching.py` | Strategy for SmolVLA, pi0, pi0.5 — delegates to lerobot's existing flow-matching loss. |
| `src/reflex/finetune/action_heads/ddpm_dit.py` | Strategy for GR00T — delegates to lerobot's GR00T DDPM loss. |
| `src/reflex/finetune/action_heads/l1_regression.py` | Stub for OFT-style (SOTA finding 3). v0.5: raises NotImplementedError. |
| `src/reflex/finetune/action_heads/fast_tokens.py` | Stub for pi0-FAST / tokenized heads. v0.5: raises NotImplementedError. |
| `src/reflex/finetune/optimizers/__init__.py` | Registry. v0.5 ships AdamW. Stubs for muon/sophia with NotImplementedError. |
| `src/reflex/finetune/auxiliary_losses/__init__.py` | Registry. |
| `src/reflex/finetune/auxiliary_losses/rehearsal.py` | Implements rehearsal-shard mix at configurable fraction (SOTA finding 7). Requires base checkpoint to ship a rehearsal shard descriptor. |
| `src/reflex/finetune/norm_stats/__init__.py` | Public API: `resolve_norm_stats(base_ckpt, dataset) -> NormStatsResult`. |
| `src/reflex/finetune/norm_stats/reuse_base.py` | Reuses base-checkpoint stats for shared dims (pain #6 fix). |
| `src/reflex/finetune/norm_stats/compute_deltas.py` | Computes new stats only for dimensions absent from base. |
| `src/reflex/finetune/hooks/__init__.py` | `HookRegistry` + `TrainerContext.hooks`. |
| `src/reflex/finetune/hooks/val_loss_hook.py` | Validation loss every N steps (pain #8). |
| `src/reflex/finetune/hooks/calibration_hook.py` | Calls `reflex.eval.calibration.compute_ece` on held-out shard every K steps. Labels "in-flight" not "final". |
| `src/reflex/finetune/hooks/export_on_checkpoint_hook.py` | Optional: auto-export every checkpoint to a side dir for drift visibility. Default off. |
| `src/reflex/finetune/hooks/wandb_hook.py` | W&B integration. |
| `src/reflex/finetune/hooks/libero_smoketest_hook.py` | v0.6 item (gated on `--benchmark`). |
| `src/reflex/finetune/monitor.py` | Uniform `training_log.jsonl` writer — every backend emits the same shape. |
| `src/reflex/finetune/postprocess.py` | `finalize(checkpoint_dir, cfg) -> FinetuneResult` — runs `export_monolithic` → `validate_roundtrip` → `calibration.run_on_holdout()` → `write_verification_report(parity=..., calibration=...)`. |
| `src/reflex/finetune/profiles/__init__.py` | Profile registry + lookup by name. |
| `src/reflex/finetune/profiles/smolvla_libero.py` | Reproducibility preset: SmolVLA LoRA r=32 on libero-10, pinned LR/batch/steps/seed (pain #3). |
| `src/reflex/finetune/profiles/pi0_aloha.py` | Pi0 LoRA matching openpi's `pi0_aloha_pen_uncap_lora` (SOTA A1). |
| `src/reflex/finetune/profiles/pi05_libero.py` | Pi0.5 LoRA with `use_quantile_norm=False` default (SOTA finding 2 — #1 silent killer). |
| `src/reflex/finetune/profiles/gr00t_full.py` | GR00T N1.6 full fine-tune on a generic SO-100 dataset shape. |
| `src/reflex/finetune/templates/modal_finetune.py` | Modal scaffold — `modal run modal_finetune.py --config ./out/finetune_config.json`. Accepts `--modal-gpu a100-80gb`. |
| `src/reflex/finetune/templates/runpod_finetune.sh` | RunPod equivalent. |
| `src/reflex/verification_report.py` | **Edit existing**: add `calibration=...` kwarg to `write_verification_report`, emit Calibration section. Keep backward compat with `calibration=None`. |
| `src/reflex/eval/calibration.py` | **Edit existing**: add `run_on_holdout(export_dir, dataset, num_samples) -> CalibrationResult` entry point. Leaves existing pure-numpy metric functions untouched. |
| `src/reflex/cli.py` | **Edit existing**: register `finetune` subcommand via one import + decoration. |

### Test files

| File | Purpose |
|---|---|
| `tests/test_finetune_preflight.py` | Unit: schema mismatch detection; memory budget arithmetic; norm-stats reuse logic; dataset-size floor warnings. No GPU. |
| `tests/test_finetune_config.py` | Unit: config dataclass; CLI parsing; profile lookup; forward-compat with unknown fields (SOTA D#9). |
| `tests/test_finetune_profiles.py` | Unit: every shipped profile loads, has required fields. |
| `tests/test_finetune_action_heads.py` | Unit: registry lookup + default resolution per base model. |
| `tests/test_finetune_postprocess.py` | Integration with mocked backend: finalize() calls export_monolithic, writes verification report with calibration+parity sections. |
| `tests/test_finetune_hook_registry.py` | Unit: hooks fire in order; val_loss hook produces structured entries; calibration_hook rejects final-checkpoint usage by label. |
| `tests/test_finetune_cli_smoke.py` | Integration: `reflex finetune --dry-run lerobot/smolvla_base --dataset lerobot/libero_10` runs preflight, exits 0, writes a dry-run report. |
| `tests/test_finetune_end_to_end_smolvla.py` | Integration with tiny dataset: run SmolVLA LoRA for 10 steps, assert parity gate passes on final checkpoint, assert VERIFICATION.md contains Calibration section. Runs on Modal in CI (GPU-expensive, daily). |
| `tests/test_finetune_end_to_end_pi0.py` | Same for pi0. Daily CI. |

### Docs

| File | Purpose |
|---|---|
| `docs/finetune.md` | How-to guide for the three supported flows (SmolVLA LoRA, pi0 LoRA, GR00T full). |
| `docs/finetune_recipes.md` | Each reproducibility profile: base model, dataset, expected success rate range, time/cost. |
| `docs/finetune_troubleshooting.md` | Mapping from the 10 pain points in competitive research to reflex-side error messages + fix actions. |
| `docs/architecture/finetune.md` | Architectural overview for contributors — points at this doc. |
| `README.md` | **Edit existing**: add "3-command finetune" quickstart alongside the "3-command deploy" quickstart. |
| `CHANGELOG.md` | **Edit existing**: v0.5 entry. |

### Hooks into existing code

- `src/reflex/cli.py` gains a `finetune` command and imports `reflex.finetune.cli.finetune_command`. No other edits required.
- `src/reflex/verification_report.py` gains `calibration=...` kwarg. Backward-compatible default `None`.
- `src/reflex/eval/calibration.py` gains a `run_on_holdout()` function. No edits to the existing metric functions.
- `src/reflex/exporters/monolithic.py` is called unchanged from `postprocess.finalize()`. No edits.
- `src/reflex/exporters/fp16_convert.py` is called unchanged when target requires FP16. No edits.
- `src/reflex/validate_roundtrip.py` is called unchanged (the parity gate). No edits.
- `GOALS.yaml` — `fine-tuning-pipeline` goal check expression already tests `src/reflex/finetune.py` existence; the new location is `src/reflex/finetune/__init__.py`. Small edit to the check glob: `test -f src/reflex/finetune/__init__.py`.

---

## Section E — Open design questions (defaults picked; needs product call)

Each is phrased as a concrete A/B/C with the architecture's current default.

**Q1. Default SmolVLA backend: lerobot (A) or HF Transformers Trainer (B)?**
- **Default: A (lerobot).** SmolVLA is a lerobot-native policy; lerobot's PeftConfig path is the most-tested. HF Trainer requires heavy subclassing for flow-matching loss per SOTA A4. Product call: if we later want one-command "`reflex finetune any-HF-model`" parity, we need B.

**Q2. Reports directory: inside the export dir (A) or a parallel `reports/` tree (B)?**
- **Default: A (inside `--output`).** One dir = one artifact = one thing customers tar up for compliance. Product call: some customers want separate immutable reports dir for audit — they would prefer B. Can revisit without architectural change.

**Q3. Calibration gate failure behavior: refuse export (A) or warn and proceed with `--force-export-anyway` (B)?**
- **Default: B (warn + flag).** Matches existing `reflex export` behavior. Refusing export on calibration failure is a louder stance. Product call: if we want to make the "calibration is the deploy gate" story maximally strong (C2 differentiator), switch to A.

**Q4. Pre-flight `--dry-run` implicit or explicit?**
- **Default: implicit.** `reflex finetune ...` runs preflight, prompts y/N on any WARN, proceeds on PASS. `--skip-preflight` is a deliberate footgun. Product call: if we think customers will hate the prompt, make `--dry-run` explicit and default to "just train." Recommend keeping implicit — saves the $100 runs that would otherwise fail at hour 3.

**Q5. Profile language: Python (A) or YAML (B)?**
- **Default: A (Python).** Profiles can have conditional logic (e.g., "if dataset < 500 episodes, set quantile_norm=False"). YAML can't. Product call: YAML is customer-editable; Python is contributor-editable. If we want first-class customer profile customization, ship profile=YAML + a "compile-to-python" pass. Recommend starting with Python and shipping a YAML loader later when customer signal emerges.

**Q6. Resume semantics: bit-exact (A) or best-effort (B)?**
- **Default: B (best-effort).** `--resume` restores optimizer + scheduler + step count + RNG seed but does not guarantee bit-exact replay (FSDP + dataloader order makes this expensive). Product call: bit-exact matters for published-paper reproducibility. If we ship reproducibility profiles (pain #3), customers will expect bit-exact. Escalate to A when a customer complaint cites it.

**Q7. `reflex export` and `reflex finetune --no-export` — is export still its own command or only the finetune-finalize step?**
- **Default: both.** `reflex export` stays as the direct-checkpoint path (unchanged). `reflex finetune` defaults to auto-export and accepts `--no-export` to skip. Product call: unified one-command flow matters for the pitch, but experienced users want `export` standalone. Current default keeps both.

**Q8. Primary backend for pi0 / pi0.5: lerobot (A) or openpi native JAX (B)?**
- **Default: A in v0.5, B optional in v0.6.** lerobot has a pi0 port; openpi has the original JAX. lerobot is in-process-friendly; openpi requires JAX. Product call: openpi users *want* openpi. If the v0.5 customer signal is "it's close enough to what openpi would give me," we can defer B to v0.7. If customers complain about val_loss gap, prioritize.

**Q9. LoRA rank default: r=16 (A, lerobot's default) or r=32 (B, SOTA A1 recommendation)?**
- **Default: B (r=32).** SOTA A1 explicitly says VLAs need 4-16x higher rank than LLMs. r=16 is the LLM-culture default. r=32 is the VLA-reality default. Product call: we diverge from lerobot here — flag this in docs so customers who cross-reference lerobot config don't get confused.

**Q10. Auto-invoke `--dry-run` only mode when GPU count is 0?**
- **Default: yes.** `reflex finetune ... --gpus 0` becomes `--dry-run`. Avoids customer confusion when they run on CPU by accident. Product call: could instead refuse. Current pick is softer — customers can run dry-run on their laptop.

**Q11. Is there a `--parity-gate off` option or is parity mandatory?**
- **Default: off is allowed but loud.** `--parity-gate off` flag exists; if set, VERIFICATION.md's verdict field reads `UNVERIFIED` and the exit code is non-zero. Product call: mandatory parity maximizes the C1 differentiator; we currently compromise for flexibility. Recommend keeping the flag — research/debug workflows need it.

---

## Section F — Forward-compatibility checklist

For each anticipated 3-year evolution, the specific Proposal 2 module that absorbs it.

**F1. A new action-head type (e.g. VLM2VLA action-as-language lands widely in 2027).**
→ Add `src/reflex/finetune/action_heads/action_language.py` + register. Nothing else moves. `TrainerContext.action_head_strategy` dispatches via the registry. If the head is unchunked (one action-token per forward), loss_fn changes but `Backend.fit()` still sees a loss tensor.

**F2. A new optimizer (Muon, Sophia, Shampoo) becomes standard.**
→ Add `src/reflex/finetune/optimizers/muon.py` + register. `Backend.fit()` receives the optimizer spec via `TrainerContext.optimizer_spec`. If the optimizer is incompatible with lerobot's internal loop, we add a `custom_optimizer_handler` hook; no core rewrite.

**F3. A new dataset format displaces LeRobotDataset v3 (e.g. NVIDIA InternData-A1 hybrid, SOTA C2).**
→ Add a `src/reflex/data/ingest/internal_data_a1.py` adapter emitting the canonical `TrajectorySample` shape. Training code sees no difference. Preflight's schema.py gains a new format-detect branch.

**F4. Training-time quantization (QAT int8/int4) becomes standard (SOTA C4).**
→ Add `src/reflex/finetune/hooks/qat_hook.py` that wires `torchao.quantization.qat` into the training loop via lerobot's existing model-wrapping hooks. `postprocess.finalize()` passes the fake-quant graph to `monolithic.py`; `monolithic.py` needs a `--qat-aware` flag (small PR — the fake-quant ops are ONNX-exportable already). `fp16_convert.py` skips FP16 conversion when fake-quant is present.

**F5. Multi-task / multi-embodiment curricula become common.**
→ Add `src/reflex/finetune/curriculum/` package with scheduler strategies. `FinetuneConfig.datasets` becomes a `list[DatasetRef]` with per-dataset weights. The data ingest layer concatenates/interleaves based on scheduler. No training-loop change. Honest caveat: SOTA D#7 flags interference dynamics are under-specified in the literature; we'll likely ship a "supported when you know what you're doing" caveat.

**F6. 100B+ VLAs appear and need 3D parallelism (TP+PP+DP).**
→ Add `src/reflex/finetune/backends/nemo_backend.py` or `deepspeed_backend.py`. Existing preflight / postprocess / calibration layer unaffected. Risk flagged in Proposal 2 risk #3 — this backend is non-trivial but strictly additive.

**F7. A VLA with RL-style post-training lands (per SOTA C5 — SimpleVLA-RL, VLA-RFT, πRL).**
→ Add `src/reflex/finetune/phases/` package. `FinetuneConfig.phase = imitation | rl | distill`. Each phase has its own backend implementation under `backends/rl_backend.py`. Hooks registry expands with `on_rollout`, `on_reward`. The imitation-phase code path is untouched.

**F8. New VLA architecture drops the flow-matching denoise loop entirely (e.g., single-step regression from day one).**
→ This is exactly the OpenVLA-OFT scenario (SOTA finding 3). Handled by F1 — add the new action head, register, done. `postprocess.finalize()` calls export_monolithic which already supports non-flow-matching (GR00T uses DDPM today; OpenVLA uses tokens). The assumption "VLA = flow-matching denoise" is not embedded in any load-bearing module.

**F9. A new safety eval (path-deviation attention heads per SOTA B#4, arxiv 2603.13782) becomes standard.**
→ Add `src/reflex/finetune/hooks/path_deviation_hook.py` (runs on checkpoint save). `verification_report.py` gains a `safety=...` kwarg (same shape as the existing `parity=` and upcoming `calibration=`). Third required section; no other changes.

**F10. HuggingFace deprecates lerobot in favor of a successor library (not forecast, but low-likelihood-high-impact).**
→ `backends/lerobot_backend.py` is swapped for `backends/<successor>_backend.py`. Every other module survives because they depend on `Backend` protocol, not on `lerobot`.

---

## Flags, caveats, divergences from the research

- **No conflict detected between the two research docs** at the architectural level. Competitive and SOTA agree that (a) thin-orchestrator-over-upstream is the right posture, (b) LeRobotDataset v3 is the primary format, (c) calibration is a differentiator, (d) lerobot+openpi are the primary backends. They reinforce each other.
- **Divergence from SOTA research A1's r=32 default only for lerobot compatibility:** SOTA says r=32 as reflex's compromise; lerobot's default PeftConfig ships r=16. We pick r=32 and document the divergence (Q9 above) rather than tracking upstream. Reason: VLA reality over LLM convention.
- **Divergence from competitive research's "C1 first, C2/C3 later" ordering:** our v0.5 bundles C1 (parity-gated) AND C2 (calibration-first) from the start because the marginal cost of adding calibration once ECE/Brier/NLL already ship in `src/reflex/eval/calibration.py` is near-zero. Competitive research's "v2–v3" framing assumes we'd build calibration from scratch; we don't.
- **Missing roadmap doc:** `finetune_roadmap.md` was not present at design time. Section D's "v0.5 shape" is inferred. If roadmap lands with different scope (e.g., "v0.5 = SmolVLA only" or "v0.5 = include GR00T LoRA"), adjust the per-file ship lists. The architecture itself is unaffected — the same module tree serves any v0.5 scope definition.
- **No invention of a reflex-native dataset format.** SOTA A3 explicitly recommends against it. We did not.
- **No writing of gradient-step code.** Axiom A1 enforced. We did not.
