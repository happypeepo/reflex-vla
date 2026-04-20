# `reflex finetune` — Competitive + Customer-Pain Research

Date: 2026-04-19
Author: research agent
Purpose: inform `reflex finetune --base <model> --dataset <hf_id> --output <dir>` scoping for a 2–3 year differentiation window.

## Section A — Competitive matrix

| Tool | Target customer | Model support | Training UX | Pricing | What's NOT bundled | Notable gap |
|---|---|---|---|---|---|---|
| **HF AutoTrain** | No-code ML devs | LLM/NER/image/tabular. **No VLA.** | Web UI + CLI, task picker | Per-minute HF Spaces hardware; free local; free tier | Robot I/O, action norm, ONNX, sim eval | Does not know what a policy is |
| **Together AI** | LLM fine-tune productizers | LLMs only | JSONL upload → API, SFT+DPO+LoRA+Full | $0.48–$3.20/M tokens; H100 $3.49/hr, H200 $4.19/hr, B200 $7.49/hr | Robot data, ONNX, eval, RL | Token billing assumes text |
| **OpenPipe** | SaaS teams replacing GPT-4 | LLMs; wraps OpenAI SDK on prod traffic | Zero-click fine-tune from logged traffic | Not publicly disclosed | Robotics anything | "Record prod → fine-tune" pattern is the right analog; nothing robotics-aware |
| **Modal Labs** | Python ML devs | General; axolotl LLM template, no VLA template | You write Python, Modal runs serverless | T4 $0.59/hr → B100 $6.25/hr; prod ≈3.75× base w/ multipliers | Every robotics-specific piece | Infra primitive not a product |
| **RunPod** | Cost-sensitive GPU renters | General templates (vLLM, SDXL). No VLA | Pick template, SSH in | Community: 3090 $0.19, 4090 $0.34, A100 $0.89/hr; A100×24h ≈$21 | Robotics, eval, export | Raw compute only |
| **lerobot train** | HF robotics research community | pi0, pi0.5, SmolVLA, ACT, Diffusion, GR00T N1.5 | `lerobot-train --dataset.repo_id=... --policy.path=...` | Open source (Apache 2.0) | ONNX/TRT, guard, serve, sim eval | **The incumbent.** Research-grade UX: shape mismatches, undocumented hyperparams, dataset-format churn. Users ready to pay to skip pain |
| **openpi train** | Pi-model specialists | pi0, pi0-FAST, pi0.5 only | JAX: `compute_norm_stats.py` → `train.py`; custom LeRobot converter | Open source | Everything else | Needs A100 80GB full / 22.5GB LoRA. [Val loss missing until user-requested](https://github.com/Physical-Intelligence/openpi/issues/544) |
| **NVIDIA GR00T** | Industrial / humanoid pilots on NVIDIA | GR00T N1.5, N1.6 | `scripts/gr00t_finetune.py`; Jupyter walkthrough | Free; needs ~25GB VRAM | Non-Jetson ONNX/TRT, Edge TPU. [DGX Spark aarch64 broken](https://github.com/NVIDIA/Isaac-GR00T/issues/474) | Lock-in: assumes Isaac Sim + Jetson |
| **Isaac Lab / Sim** | Sim-first RL researchers | Custom RL, not VLA fine-tune | GPU-accelerated sim ~90k FPS | Free + GPU cost | Teleop ingest, VLA fine-tune, real deploy | Different paradigm (RL from scratch in sim) |
| **Skild AI** | Humanoid OEMs, enterprise | Proprietary "Skild Brain" | Managed AI-factory, not self-serve; 20k+ hr sim + video pretrain | Not disclosed; $1.4B raise, $14B val | Self-serve, API, cost transparency | Different market tier |
| **robosuite + SB3** | RL academics | None directly VLA; PPO/SAC/TD3 on custom envs | Python scripts + handwritten envs/rewards | Free | Data pipeline, VLA, real transfer | Sample-inefficient, sim-to-real gap |
| **Cohere fine-tune** | Enterprise LLM devs | Command R family | Web + API, SFT + RLHF | $3/M train + $0.30/M in + $1.20/M out | Robotics anything | UX benchmark for enterprise-ready fine-tune |
| **Mistral fine-tune** | Text devs wanting cheap small fine-tunes | Mistral Small/Medium (Medium 3 is API-only, no tune) | API + docs | Small 3.1 $0.10/$0.30 per M; no rate-limit tiers published | Robotics | "Customize small to beat large on your task" — the right positioning to borrow |

### Category read
- Horizontal LLM fine-tune (Together/OpenPipe/Cohere/Mistral): strong DX, zero robotics semantics.
- Horizontal GPU rental (Modal/RunPod): cheap, infra-only.
- Robotics-native (lerobot/openpi/GR00T): painful UX, free, the **real competitive set**.
- Sim-RL (Isaac Lab/robosuite): adjacent paradigm, not a direct substitute.
- Enterprise (Skild): managed, not self-serve.

## Section B — Top 10 customer pain points (ranked)

### 1. "Loss converged, wandb looked fine, evaluation was 0%"
- **Pain**: fine-tune succeeds by every training signal yet the robot doesn't move — silent action/state/norm misalignment between user data and pretrained checkpoint.
- **Frequency**: the single most common complaint. [lerobot #2259](https://github.com/huggingface/lerobot/issues/2259), [#1791](https://github.com/huggingface/lerobot/issues/1791), [#1370](https://github.com/huggingface/lerobot/issues/1370), [openpi #711](https://github.com/Physical-Intelligence/openpi/issues/711).
- **Persona**: academic + indie hobbyist.
- **Quote**: "despite the training running without issues (loss converges and wandb plots look fine), the evaluation yields 0% success. I suspect I'm misunderstanding how to correctly define the state and action spaces for SmolVLA" ([#2259](https://github.com/huggingface/lerobot/issues/2259))
- **Workaround**: forum plea; some never recover the run.
- **Reflex answer**: **pre-flight schema validator** before touching a GPU — dims, ranges, image-key names vs the base checkpoint's distribution, with explicit "your data will fail because X" errors. Post-train: auto ECE/Brier + tiny LIBERO smoketest, not just loss.

### 2. "Shape mismatch 6 vs 7"
- **Pain**: action-dim mismatch (6-DoF user data vs 7-DoF checkpoint) surfaces only inside the normalizer at eval, often after hours of training.
- **Persona**: indie + integrator.
- **Quote**: "RuntimeError: The size of tensor a (6) must match the size of tensor b (7)" ([#2418](https://github.com/huggingface/lerobot/issues/2418))
- **Workaround**: manual `rename_map`, patching the unnormalizer.
- **Reflex answer**: schema validator (#1) + `reflex adapt --from 7dof --to 6dof` that drops/pads with a logged warning in VERIFICATION.md.

### 3. "Can't reproduce the paper"
- **Pain**: published VLA results unreachable because effective batch size, LR schedule, commit hash aren't documented ([lerobot #3287](https://github.com/huggingface/lerobot/issues/3287)); recurring in arXiv limitations sections.
- **Persona**: academic.
- **Quote**: "I suspect the discrepancy might be primarily due to the effective batch size" ([#3287](https://github.com/huggingface/lerobot/issues/3287)).
- **Workaround**: file GitHub issues and guess.
- **Reflex answer**: **reproducibility presets** — `--preset libero-smolvla-paper` pins batch/LR/steps/seed; writes a repro manifest into VERIFICATION.md. No competitor provides this.

### 4. "Dataset format broke between versions"
- **Pain**: LeRobot v2→v3 broke pipelines; no direct v2.0→v3.0 converter ([lerobot #1998, #1999, #2158](https://github.com/huggingface/lerobot/issues/1998)).
- **Persona**: anyone with prior data.
- **Quote**: "Error After Converting Dataset from v2.1 to v3 Using lerobot.record" ([#1999](https://github.com/huggingface/lerobot/issues/1999)).
- **Workaround**: multi-step manual conversion.
- **Reflex answer**: auto-detect + auto-upgrade dataset version on the fly; version-pin the loader with a tested compat matrix.

### 5. "CUDA OOM on my dual 32GB setup"
- **Pain**: pi0.5 full fine-tune needs >70GB single-GPU; LoRA >22.5GB. 3090/4080/5080 users hit OOM mid-training ([#2608, #2371, #2209](https://github.com/huggingface/lerobot/issues/2608)).
- **Persona**: indie + small academic lab.
- **Quote**: "memory increase continuously during training Groot" ([#2371](https://github.com/huggingface/lerobot/issues/2371)).
- **Workaround**: grad-checkpointing, tiny batch, LoRA.
- **Reflex answer**: **pre-flight memory budget** (params + activations + optimizer at configured precision); auto-enable LoRA/grad-ckpt/bf16 or refuse with a "need more VRAM, rent cost is $X" estimate.

### 6. "LoRA tanked from 96% to 1%"
- **Pain**: LoRA pi0.5 LIBERO 96%→1% from silent norm-stats mismatch between pretrain and recomputed stats ([openpi #711](https://github.com/Physical-Intelligence/openpi/issues/711)).
- **Persona**: academic + advanced indie.
- **Quote**: "after the finetuning process, the success rate is very low (1% on spatial task suite)" ([#711](https://github.com/Physical-Intelligence/openpi/issues/711)).
- **Workaround**: hand-merge stats.
- **Reflex answer**: **norm-stats as first-class artifact**. Reuse base-checkpoint stats for pretrained dims; only compute deltas for new dims. Log both in VERIFICATION.md.

### 7. "DGX Spark aarch64 Blackwell won't install anything"
- **Pain**: CUDA 13.0 aarch64 has no prebuilt wheels for flash-attn, pytorch3d, decord; transformers enforces decord even unused ([Isaac-GR00T #474](https://github.com/NVIDIA/Isaac-GR00T/issues/474)).
- **Persona**: integrator + enterprise pilot.
- **Quote**: "I couldn't find much documentation for this specific setup" ([#474](https://github.com/NVIDIA/Isaac-GR00T/issues/474)).
- **Workaround**: build from source, mock modules.
- **Reflex answer**: ship Docker arm64 training image with flash-attn/pytorch3d prebuilt for Jetson + Grace-Blackwell — extending the arm64 muscle we already have for inference.

### 8. "No validation loss, no idea when to stop"
- **Pain**: openpi didn't log val loss until a user filed an issue — users train blind, discover overfit weeks later at real-robot eval ([openpi #544](https://github.com/Physical-Intelligence/openpi/issues/544)).
- **Persona**: all.
- **Quote**: "I find that having the validation loss logged during training is helpful to e.g. know when to stop fine-tuning" ([#544](https://github.com/Physical-Intelligence/openpi/issues/544)).
- **Workaround**: patch train.py.
- **Reflex answer**: log val loss + calibration (ECE/Brier) + sim success every N steps; early-stop on sim-success plateau, not train loss.

### 9. "How much data is enough?"
- **Pain**: users guess at demo counts. Helix used 500hr teleop; LingBot-VLA 20,000hr; blogs say "80 demos" or "50 trajectories". No principled guide.
- **Persona**: indie + integrator (first VLA contact).
- **Workaround**: collect 100 and hope.
- **Reflex answer**: `reflex finetune --estimate` runs a learning-curve sweep on a subset and extrapolates predicted success at full-data — user knows *before* a 24hr H100 run whether they have enough data.

### 10. "Sim-to-real drops everything"
- **Pain**: sim-fine-tuned VLAs transfer poorly to real; DR requires manual tuning that scales poorly to long-horizon tasks (RLinf-Co, multiple 2026 papers).
- **Persona**: academic + startup.
- **Quote** ([arxiv 2602.12628](https://arxiv.org/html/2602.12628)): "real-world deployment typically depends on zero-shot sim-to-real transfer with domain randomization, frequently leading to significant performance drops on real robots."
- **Workaround**: hand-tune DR + iterate.
- **Reflex answer (v2)**: `reflex finetune --cotrain sim:isaac+real:dataset` mixing ratios, calibration-ECE reported per-source. Deferred past v1.

Note: I searched Reddit r/robotics and r/MachineLearning via site-search and turned up no open discussion threads indexed for "VLA fine-tune", "pi0 fine-tune", or "lerobot training" in the past six months — a **finding in itself**: these users are on GitHub issues and Discord, not public Reddit. Our distribution plan should reflect that (target LeRobot Discord, HF forums, GitHub issue replies rather than Reddit threads). HN search via Algolia returned similar emptiness on "VLA training" as an org-building thread — VLA discussion on HN is model-release-driven, not tooling-driven.

## Section C — Three long-term differentiation candidates

### C1. "From demonstrations to deployed ONNX in one command, with cos=+1.000000 parity guaranteed"
- **What reflex does**: a vertically integrated `reflex finetune → reflex export → reflex serve` with the parity assertion (cos=+1.0 between PyTorch reference and ONNX/TRT output) **as a gate in the finetune loop**, not an afterthought. If a fine-tuned checkpoint fails ONNX parity, finetune fails loudly and tells you which layer drifted.
- **Defensible long-term because**: every competitor treats train and deploy as separate worlds with their own bug classes. We already own the parity toolchain (12 bugs documented, cos=+1.000000 verified on 4 models). Feeding that back into training is a moat a month's work can't clone.
- **Need to build**: training wrapper around openpi/lerobot trainers that emits checkpoints in our validated format; a `reflex finetune --parity-gate on` mode that runs single-step parity checks at each save.

### C2. "Calibration-first evaluation — because sim success is not real success"
- **What reflex does**: `reflex finetune` treats ECE + Brier + NLL on a held-out slice as **the primary stopping signal** — not train loss, not sim success. Zollo 2025 (in our vault) is the only known monotonic link to real-world task success. No competitor does this.
- **Defensible long-term because**: every new VLA architecture (pi0 → pi0.5 → pi1 → xVLA → …) reshuffles what "good loss curve" means. Calibration metrics are architecture-agnostic and survive the 3-year horizon when DiT/AdaRMSNorm/flow-matching all cycle out.
- **Need to build**: port the goal already in GOALS.yaml (`calibration-metrics`, weight 8, v0.3 deferred) forward from inference-time to training-time. This is a 1–2 week feature once the base finetune command exists.

### C3. "Any base model × any embodiment × any edge hardware — orthogonalized"
- **What reflex does**: the finetune command accepts `--base {pi0,pi0.5,smolvla,gr00t-n1.6,openvla,xvla,pi1-when-it-ships,any-new-vla}` × `--embodiment {so-arm-100, franka, ur5, custom-6dof, bimanual, humanoid-g1}` × `--target-hw {jetson-orin-nano, jetson-thor, rtx-4090, h100}` as three independent axes — and resolves all integration tax (schema adapters, memory budget, distillation strategy, export opset) automatically.
- **Defensible long-term because**: VLA architectures are moving fast; embodiments are proliferating (humanoids + arms + legged + drones); edge hardware is diversifying (Orin Nano, Thor, Groq, AMD MI, edge-TPU). The **orthogonalization** itself — not any individual combo — is the product. Competitors each pick one axis (lerobot: many models one backend; GR00T: one model one hardware; openpi: one family one use). No one is building the full cross-product matrix.
- **Need to build**: a compatibility matrix with tested (model × embodiment × hardware) cells + an extension API so third parties can add a row (new robot) or column (new model) in a PR. This is the **platform** play and matches reflex's existing 7-wedge + multi-model parity muscle.

## Section D — Red flags / contrarian takes

1. **"Training is commoditized — you're charging for glue."** Partly true for LLM fine-tune (Together/OpenPipe own it), false for VLA because the pain is *semantic* (action spaces, norm stats, calibration) not infrastructural. Risk: if NVIDIA ships one-click GR00T fine-tune in a future Isaac release, our GR00T differentiation shrinks.

2. **"Finetune is one-off; deploy is recurring. You'll dilute your ARR wedge."** Mostly right. First-finetune is rare per customer; *recurring* deploys carry the volume. Finetune should be lead-gen into the deploy funnel, not a P&L line item. Price modestly for runs; make money on serve/guard/turbo.

3. **"GitHub complainers are academics who won't pay."** Partial truth: lerobot issue authors skew academic. But GR00T #474 (DGX Spark setup) and sim-to-real literature are integrator signals. Tier accordingly — cheap credits for academics, support contract for integrators.

4. **"One pi0.6 or xVLA release and your training wrapper is stale."** Strongest point. Mitigation: keep `reflex finetune` a *thin orchestrator* over lerobot/openpi/gr00t's native trainers. We own schema validation, pre-flight, parity gate, calibration eval, export — *not* gradient math. When pi1 lands, we add an exporter; the rest keeps working.

5. **"Why pay reflex when Modal + a shell script is cheaper?"** Depends on persona. Academics: yes, run locally. Hobbyists: will pay once to skip pain. Integrators: care about audit trail + SLA (our VERIFICATION.md is exactly that), not $50 on the GPU bill. Charge per verified-and-deployed finetune, not GPU-minute.

6. **"Calibration is cute research; customers ask 'does it work on my robot'."** Fair — C2 must ship alongside a sim-eval hook, not instead of it. Our AllenAI vla-eval wrap covers LIBERO/SimplerEnv. Combine: calibration = cheap continuous signal during training; sim-eval = discrete at checkpoint save; real-eval = customer's, but we emit the VERIFICATION.md they attach to their safety case.

### Net take
Build C1 (parity-gated finetune) as v1 — one-month reach from our existing parity toolchain, hardest wedge for competitors to clone in the same timeline. C2 and C3 are v2–v3. Do not over-invest in raw training infra: wrap upstream trainers, own the semantic + validation layer.

---

Sources heavily used:
- [lerobot GitHub issues #1370, #1698, #1791, #1998, #1999, #2158, #2209, #2242, #2259, #2371, #2418, #2446, #2608, #3287](https://github.com/huggingface/lerobot/issues)
- [openpi GitHub issues #544, #662, #667, #711, #763, #842](https://github.com/Physical-Intelligence/openpi/issues)
- [NVIDIA Isaac-GR00T #474](https://github.com/NVIDIA/Isaac-GR00T/issues/474)
- [Together AI pricing](https://www.together.ai/pricing), [Modal pricing](https://modal.com/pricing), [RunPod pricing](https://www.runpod.io/)
- [RLinf-Co arxiv 2602.12628](https://arxiv.org/html/2602.12628), [OpenVLA-OFT arxiv 2502.19645](https://arxiv.org/abs/2502.19645)
- [openpi README](https://github.com/Physical-Intelligence/openpi), [Isaac-GR00T finetune docs](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/2_finetuning.ipynb)
- [Skild AI funding coverage](https://techcrunch.com/2026/01/14/robotic-software-maker-skild-ai-hits-14b-valuation/)
- Internal vault: [competitors/lerobot.md](../../../reflex_context/02_research/competitors/lerobot.md), [competitors/physical_intelligence.md](../../../reflex_context/02_research/competitors/physical_intelligence.md), [GOALS.yaml calibration-metrics, distill-dmpo](../../GOALS.yaml)
