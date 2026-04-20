# `reflex finetune` — Rollout Roadmap

Date: 2026-04-19
Author: product strategy
Status: commitment document, not a survey
Inputs: `finetune_competitive_research.md`, `finetune_sota_research.md`, `GOALS.yaml`

The research is done. `reflex finetune` ships. This doc is sequencing: what lands in v0.3, v0.5, v1.0, what we say no to, and where customer signal overrides the plan.

Core framing: reflex already has cos=+1.000000 ONNX parity on 4 models, FP16 Orin Nano fit, ROS2 bridge, eval-time calibration, safety wedges. **Finetune is not a new product — it is the front of the existing funnel.** First-finetune feeds into recurring-deploy, where ARR lives. Sequence accordingly.

---

## Section A — Three horizons

### v0.3 — MVP ship (next 2 weeks)

**Stable CLI surface**
- `reflex finetune --base lerobot/smolvla_base --dataset <hf_id> --output ./dir` runs end-to-end on one GPU, produces a LoRA checkpoint + auto-triggers `reflex export` to monolithic ONNX.
- `src/reflex/finetune.py` with `run_finetune()` importable (closes `fine-tuning-pipeline`, weight 5).
- Thin wrapper over `lerobot-train`: we own CLI, dataset resolution, export handoff. We do **not** own gradient math.
- SmolVLA only. LoRA r=32 (SOTA A1). bf16. Single GPU. No FSDP.
- VERIFICATION.md gains a "finetune provenance" block: base hash, dataset ID, LoRA rank, seed, steps, train-loss thumbnail.

**Customer win that wouldn't exist otherwise**
"My 80-demo SO-100 dataset → deployed monolithic ONNX on Orin Nano in one command, with a VERIFICATION.md receipt." Today a hobbyist runs `lerobot-train` → hits a shape or norm bug → finds no ONNX exporter → gives up. v0.3 closes that loop for SmolVLA-on-SO-100, the highest-volume persona in the LeRobot Discord.

**Hypothesis**
Hobbyists and indie integrators will pay to skip the `lerobot-train` → ONNX gap even if training itself is a wrapper. If fewer than ~20 users complete end-to-end runs within 4 weeks, hypothesis is wrong and v0.5 pauses.

**Working invocation**
```
reflex finetune --base lerobot/smolvla_base \
  --dataset lerobot/so100_pick_place_80eps \
  --output ./so100_smolvla --mode lora --steps 20000
```
Produces `checkpoint.safetensors`, auto-exported `monolithic.onnx`, `VERIFICATION.md`, `reflex_config.json`. Then `reflex serve ./so100_smolvla` works.

---

### v0.5 — Table stakes (3 months out)

**Stable CLI surface**
- `reflex finetune --base {smolvla, pi0}` — pi0 joins. openpi JAX backend behind `--backend openpi` (SOTA A4); default picks backend by model family.
- **`--preflight`**: schema + norm-stats + memory-budget validator runs *before* GPU spin-up. Closes pain points #1, #2, #5, #6. Highest-leverage feature in v0.5. Refuses the run with a specific fix rather than wasting 4 hours on mis-matched action dims.
- **`--parity-gate on`**: checkpoint-save hook runs single-step ONNX parity (cos ≥ 0.999) at each save; fails loudly on drift. **This is C1, promoted from v1 to v0.5** — we already own the parity toolchain (12 bugs documented, 4 models at cos=+1.0), wiring into training is a 2-3 week extension. Makes us the only tool that can't ship a trained-but-unservable model.
- **`--eval calibration`**: ECE + Brier + action-MSE on a held-out 10% slice every N steps; early-stop on sim-success plateau, not train loss. **This is C2.** Closes pain point #8. Lifts directly from the shipped `calibration-metrics` goal into training-time.
- **`--estimate`**: learning-curve sweep on a 20% subset, extrapolates predicted success before a 24-hour H100 run. Closes pain point #9.
- LeRobotDataset v2→v3 auto-upgrade on the fly (pain point #4).
- `--preset libero-smolvla-paper` pins batch/LR/steps/seed and writes a repro manifest.

**Customer value shift**
v0.3: "training succeeded, here's your ONNX." v0.5: "we refused your bad dataset, saved you $60 of GPU, here's the fix." When training does run, parity-gate + calibration eval prevent the silent-killer bugs (#1, #6). Wrapper → opinionated pre/post-flight validator.

**Hypothesis**
Pre-flight + parity-gate turn one-off runs into recurring relationships because trust-per-dollar-spent rises. Observable: do v0.5 users re-run finetune within 30 days more than v0.3 users? If yes, retention wedge. If no, we're polishing a one-off.

---

### v1.0 — Long-term product shape (6-12 months)

**Stable CLI surface**
- **Pluggable action heads** (SOTA C1): internal registry for `flow_matching | ddpm | l1_regression | fast_tokens | action_language`. Lets pi1 / xVLA / GR00T N2 slot in without re-architecting. Keeps reflex out of the 2027 museum.
- **Multi-backend solid**: `--backend lerobot | openpi | gr00t-scripts`. GR00T reaches parity with SmolVLA/pi0. NVIDIA ships no official LoRA for GR00T N1.6 (SOTA B5); we fill that gap.
- **The orthogonalized matrix** (C3): `--base × --embodiment × --target-hw` with a published compat matrix. Supported bases: smolvla, pi0, pi0.5, gr00t-n1.6, openvla, xvla. Embodiments: so-arm-100, franka, ur5, bimanual, humanoid-g1, custom. Hardware: orin-nano, orin, thor, rtx-4090, h100. See Section B for defense.
- `reflex adapt --from 7dof --to 6dof` schema adapter (pain #2).
- `--distill snapflow`: one-step flow-matching distill. SOTA A5 reports 9.6× latency win on pi0.5/LIBERO in ~12 hours single-GPU. Aligned with existing `distill-dmpo` goal.
- `--qat int8`: wires `torchao.quantization.qat`. SOTA C4 says table-stakes by 2027.
- Extension API: third parties PR a new matrix row (embodiment) or column (base model); matrix compounds.

**Customer value**
v0.5 is "I trust reflex for SmolVLA or pi0 on SO-100." v1.0 is "I trust reflex for *any* supported VLA × robot × Jetson, with consistent receipts across the matrix." Consistency is the platform claim.

---

## Section B — The orthogonalized platform hypothesis (C3)

**Pick: (c) conditional commit, with specific reassessment checkpoints.**

**Pressure test.** What does a customer running `reflex finetune --base pi0 --embodiment ur5 --target-hw orin-nano` see that they can't get from `lerobot-train` + their own ONNX export + their own Jetson fit check?

Defensible answer:
1. **Pre-tested cell**: someone already shipped `(pi0, ur5, orin-nano)` through the pipeline and published the VERIFICATION.md. Customer runs a known recipe, not a research experiment. `lerobot-train` can't give this because lerobot doesn't own deploy.
2. **Integration-tax resolution**: LoRA rank right for pi0 (not generic r=16), norm mode right for dataset size, memory fits Orin Nano, opset accepted by TRT on Thor. One 3-D lookup, not three separate ones.
3. **Compatibility negative**: `(gr00t-n1.6, humanoid-g1, orin-nano)` returns "insufficient VRAM, nearest tested cell is (gr00t-n1.6, humanoid-g1, orin)." Saves a wasted run.

**Table stakes vs wedge**: points 2-3 become table stakes the moment *one* competitor commits to the matrix. The wedge is point 1 — the network effect of community-published cells. Closest analog: Docker Hub images or HuggingFace models. The aggregation is the product.

**Reassessment checkpoints:**
- **End of v0.5**: if total cell count <10, matrix is vaporware. Double down on C1 as the technical moat.
- **Mid v1.0**: if the extension API has zero external PRs after 6 months, the community thesis is wrong. Fall back to us owning ~36 cells; drop the platform framing.
- **If NVIDIA ships one-click GR00T-finetune-and-deploy by mid-2026**: the humanoid column loses its wedge. Retreat to SO-100 + Franka + UR5 as the defensible set.

**Commitment**: C3 is v1.0's north star, *conditional* on v0.5 C1+C2 earning their place. If v0.5 validators fail on adoption, C3 becomes 36 configurations of a tool nobody trusts. C1 is foundation; C3 compounds on top.

---

## Section C — Explicit non-goals per horizon

### v0.3
1. **No pi0/pi0.5/GR00T finetune.** SmolVLA only (450M, proven export path, single A100 iteration). Adding pi0 doubles QA surface without closing the goal check faster.
2. **No multi-GPU / FSDP.** Single GPU only. FSDP on VLAs needs 1-2 days per base model to debug (SOTA D8). Defer to v0.5.
3. **No pre-flight validator.** Painful, ship anyway with "data must match base schema" caveat. Building it pre-MVP delays launch 3+ weeks for a feature only useful once users exist.
4. **No custom action-head abstraction.** Consume lerobot's config verbatim. No `--action-head` flag.
5. **No pricing gate.** Free, unlimited. Zero friction at adoption gate.

### v0.5
1. **No GR00T finetune.** NVIDIA ships no official LoRA (SOTA B5); finding the right DiT rank is 1-2 weeks of blind trial (SOTA D1). Defer to v1.0 once funnel signals GR00T customers.
2. **No xVLA / OpenVLA / VLM2VLA.** Ship the two flow-matching families we already have parity on. Tokenized-action families need unvalidated discrete-action calibration methodology (SOTA D3, D6). Don't ship research as product.
3. **No RL post-training.** Reserve the CLI slot `--phase imitation|rl|distill`; don't implement `rl`. Full-RL is a 2027 buy-in.
4. **No multi-task / multi-embodiment in a single run.** One task, one robot, one base per call. Multi-task literature is under-specified (SOTA D7). Customers chain finetunes if they need it.
5. **No managed cloud.** Emit `modal`/`runpod` templates (SOTA A7); customer pays compute directly. We don't resell GPUs this horizon.

### v1.0
1. **No sim-to-real cotraining.** Pain #10 is real but unsolved algorithmically (RLinf-Co). We're not an RL lab. Revisit only as $50k+ professional services.
2. **No world-model co-training auxiliary heads.** No customer pull.
3. **No new-optimizer support** (Muon, Sophia). Robotics is 18 months behind LLM optim trends (SOTA C3). Reserve registry slot, don't implement.
4. **No reflex-native dataset format.** LeRobotDataset v3 is at 54.6% adoption (SOTA A3). We ship adapters, never a format.
5. **No fleet fine-tuning orchestration.** Enterprise-tier story. Quote professional services if asked.

---

## Section D — Customer fork-in-the-road moments

### Fork #1: After v0.3 (weeks 2-6 post-launch)

Three signals determine v0.5's lean:
- **C1 (parity-gate)**: ≥30% of v0.3 users file "ONNX export failed after training" or "PyTorch works, ONNX doesn't" in Discord/GitHub within 4 weeks → silent checkpoint drift is the binding pain.
- **C2 (calibration)**: ≥20% ask "how do I know if my model is good?" or "what's a good loss value?" within 4 weeks → users can't read loss curves.
- **C3 (matrix)**: ≥5 users request bases we don't support (pi0/pi0.5/GR00T), OR ≥5 request non-SO-100 embodiments → single-cell MVP hits a surface-area wall, accelerate C3 into v0.5.

**Default if no signal dominates**: ship C1 in v0.5. Defensible moat, parity toolchain already owned. **Needs a product call** if signals tie: founder picks the one closest to the nearest enterprise sales conversation.

### Fork #2: Mid v0.5 (month 2 of 3)

- **Finetune runs <50/month across all users**: wrapper is not being used. Cut v0.5 scope, ship parity-gate only, redirect to distillation + Turbo (serve-tier ARR, not finetune).
- **Parity-gate false-positive rate >10%**: our gate fails good models too often. Ship opt-in only; revisit after v0.5.
- **A v0.3 user publishes a successful on-robot deployment**: opposite signal — **accelerate**. Package their workflow as a `--preset`, front-page the case study. Case studies beat features.

### Fork #3: End of v0.5 (gate to v1.0)

Commitment to C3:
- **Cell count ≥10** (us + external): ship v1.0 matrix as planned.
- **Cell count 5-9**: scope v1.0 to SmolVLA + pi0 × 3 embodiments × 3 hardware (18 cells min); defer GR00T to v1.1.
- **Cell count ≤4**: **abandon C3.** Pivot v1.0 to distillation + QAT + fleet-serve. The matrix thesis is dead; reflex becomes a tight 2×2×2 tool with best-in-class parity + calibration + latency. Smaller product, real one.

**Enterprise override**: one signed contract >$50k ARR naming the matrix as a buying reason beats 10 community cells. Ship full v1.0 scope regardless of cell count in that case.

---

## Section E — Monetization cadence

Research's central insight: **finetune is one-off; deploy is recurring. Price modestly on finetune; monetize serve/guard/turbo.**

### v0.3: Free, unlimited
- No pricing gate. No Stripe. No API key for `reflex finetune`.
- Adoption-first. Finetune is lead-gen into the serve funnel. Friction at v0.3 kills the funnel.
- Pro tier (existing `stripe-license-gating`, weight 6) continues to gate `reflex serve --fleet`, `turbo`, `distill`. Finetune is the free on-ramp.

### v0.5: Free base finetune; Pro gates `--parity-gate` and `--eval calibration`
- Validators are the value-add. Base finetune stays free; the receipts-and-guardrails go behind the existing Pro tier.
- Matches "serve is where ARR lives." Hobbyist gets a working finetune; integrator with a safety case pays for VERIFICATION.md that includes parity-gate passes and calibration curves.
- **Market-aligned**: Pro stays at existing tier pricing; we're bundling finetune-validators into the deploy subscription, not creating a new SKU.

### v1.0: Free finetune + Pro (validators + GR00T + distill) + Enterprise (matrix + SLA)
- **Free**: `reflex finetune` on any supported base × LoRA × community-verified cells. No run limits.
- **Pro**: parity-gate, calibration eval, `--distill snapflow`, QAT, GR00T LoRA, multi-GPU FSDP, `serve --fleet`, Turbo.
- **Enterprise**: access to not-yet-community-validated cells, professional services for new embodiment onboarding, customized VERIFICATION.md for customer safety-case format, SLA on parity-gate pass rate, priority support.
- **Benchmark** against Cohere/Together enterprise fine-tune pricing for the Enterprise tier. Don't invent numbers.

**Never monetize** raw GPU-minutes. We're not Modal or RunPod. If we ever invoice per H100-hour, we've lost the thread — the wedge is the semantic layer, not the compute bill.

---

## Sequencing one-liner

**v0.3 closes the goal check and proves the wrapper wedge.** **v0.5 turns the wrapper into a validator and tests the trust hypothesis.** **v1.0 commits to the matrix only if the validators earned the right to build on top of them.** No horizon assumes more than 2 engineers. Each horizon has a specific kill-gate. This is sequencing, not aspiration.
