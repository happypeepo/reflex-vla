# VLA Fine-Tune SOTA Research — `reflex finetune` Architecture Input

**Scope.** Evidence base for `reflex finetune --base <model> --dataset <hf_id> --output <dir>` → auto-`reflex export`. Designed to stay relevant across pi0 → pi0.5 → pi1, SmolVLA v2, GR00T N1.6 → N1.7 → N2, and new families (OpenVLA-2, xVLA, Gemini Robotics). Citations throughout; unverified claims marked `UNCITED`.

---

## Section A — Answer matrix (opinionated, evidence-grounded)

### A1. Fine-tuning approach matrix (LoRA / full / hybrid)

| Model | Params | Backbone | Action head | Recommended default in reflex |
|---|---|---|---|---|
| SmolVLA | 450M | SmolVLM2-500M (SigLIP + SmolLM2) | Flow-matching expert ~100M | **LoRA r=32, train_expert_only=True** — lerobot default freezes VLM, trains 100M expert fully. Matches HF guidance. |
| pi0 | 3.3B | PaliGemma-3B | Flow-matching action expert 300M | **LoRA `gemma_2b_lora` on PaliGemma + `gemma_300m_lora` on expert**, per openpi's own example (`pi0_aloha_pen_uncap_lora`). EMA off during LoRA. |
| pi0.5 | 3.3B | PaliGemma-3B + FAST action tokenization | Flow + discrete FAST tokens | **LoRA with `pi05=True` flag**, but verify `use_quantile_norm=False` on small datasets ([openpi#763](https://github.com/Physical-Intelligence/openpi/issues/763)). Language grounding loss is the main risk ([#768](https://github.com/Physical-Intelligence/openpi/issues/768)). |
| GR00T N1.6 | 3B | NVIDIA Cosmos-Reason-2B (Qwen3-VL) VLM + 32-layer DiT | DiT flow action head | **Full fine-tune default** — NVIDIA's [official guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md) exposes no LoRA path; 40 GB+ VRAM recommended. LoRA on the DiT requires custom implementation. |

**Why the asymmetry.** LoRA-SP ([arxiv 2603.07404](https://arxiv.org/abs/2603.07404)) shows VLAs have an intrinsic rank ~128 on cross-embodiment transfer, vs r=4–8 for LLMs. For same-embodiment post-training, r=16–32 is empirically enough (LeRobot default `r=16` in [`PeftConfig`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/configs/default.py)). For cross-embodiment/out-of-distribution, standard LoRA fails and rank-adaptive methods (LoRA-SP, α-LoRA [arxiv 2510.21345](https://arxiv.org/abs/2510.21345)) become necessary.

**reflex decision.** Ship 3 profiles: `--mode lora` (default r=32, α=64, targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` on VLM + full training on action expert); `--mode lora-cross-embodiment` (r=128, LoRA-SP router); `--mode full` (models <500M or when `--rehearsal` is set).

### A2. Catastrophic forgetting + distribution shift

**Evidence base.**
- OpenPI explicitly co-trains on general VLM data during fine-tune to preserve semantic reasoning ([pi0.5 paper arxiv 2504.16054](https://arxiv.org/abs/2504.16054)).
- pi0.5 user reports in [openpi#768](https://github.com/Physical-Intelligence/openpi/issues/768): 250 video / 90-min fine-tune → language ignored, trajectories replayed. The dataset-size floor is real.
- VLM2VLA ([arxiv 2509.22195](https://arxiv.org/abs/2509.22195)): action-as-language token representation + LoRA completely averts the forgetting that custom action heads cause.
- Simpler fix ([arxiv 2512.13706](https://arxiv.org/abs/2512.13706)): even 6.2% of original pre-train data mixed in during fine-tune eliminates forgetting in math→NLI; same pattern observed in robotics.

**reflex decision.** Three forgetting-mitigation knobs:
1. `--rehearsal-fraction 0.05` — if base model ships a rehearsal shard, mix 5%/step.
2. `--freeze-vision-encoder` on by default (matches SmolVLA + OFT: 76.5%→97.1% *without* unfreezing vision [arxiv 2502.19645](https://arxiv.org/abs/2502.19645)).
3. `--action-norm-mode quantile|mean-std` — default mean-std below 500 episodes, quantile above ([openpi#763](https://github.com/Physical-Intelligence/openpi/issues/763)).

Skip EWC / L2-SP as first-class features — Bayesian PEFT ([arxiv 2402.12220](https://arxiv.org/abs/2402.12220)) exists but the operational payoff on VLAs is unproven. Rehearsal + frozen vision covers 80% of the forgetting surface.

### A3. Dataset format landscape

| Format | Adoption signal (2025–26) | Reflex role |
|---|---|---|
| **LeRobotDataset v3.0** | 1,633 datasets / ~54.6% of HF robotics as of Oct 2025 ([kamenski.me](https://www.kamenski.me/articles/lerobot-datasets-oct-2025)). Parquet + MP4/AV1. 5–10× storage vs v2.1. Streaming-native. [HF blog](https://huggingface.co/blog/lerobot-datasets-v3). | **Primary input** — reflex ingests this natively, no conversion. |
| **RLDS** (TFDS-based) | Foundational for Open X-Embodiment + RT-1-X/RT-2-X/Octo. [OXE paper arxiv 2310.08864](https://arxiv.org/abs/2310.08864). Still alive but stagnant — the momentum has shifted to LeRobot format. | **Convert-from** — reflex ships `reflex convert rlds→lerobot` wrapper around [any4lerobot](https://github.com/Tavish9/any4lerobot). |
| **Open X-Embodiment** | 1M+ trajectories, 22 embodiments. Single largest pretraining corpus. | Supported as a data source via RLDS conversion. |
| **Custom / NVIDIA InternData-A1 / simulator** | Vendor-locked — NVIDIA Cosmos sim data, PhysX output. | Out of scope for reflex; customer pipelines their own. |

**reflex decision.** LeRobotDataset v3 as native; ship an RLDS converter as v1 of `reflex convert`. Do not invent a reflex-native format.

Pain points to design around:
- v3.0's parquet-per-chunk packs multiple episodes per file — simple filtering ("episodes 1–5 only") needs the relational metadata in `meta/episodes/`. Don't assume one-file-per-episode.
- `delta_timestamps` (temporal windowing) is where most fine-tune misconfig happens ([lerobot#2061](https://github.com/huggingface/lerobot/issues/2061) shows async is already brittle). Default to the policy's `n_obs_steps` and reject overrides with sanity bounds.

### A4. Training stack maturity

**lerobot-train** (HuggingFace's [`src/lerobot/scripts/train.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/train.py)):
- PEFT support landed in v0.4.4 via PR #1411. `PeftConfig(r=16, method_type="LORA", target_modules=None, full_training_modules=None)` as the dataclass.
- Gaps a 6-month robotics startup will hit: async inference issues ([lerobot#2356, #3204, #2980](https://github.com/huggingface/lerobot/issues/2356)), no ONNX export (#3146, #1899, #1923 — all open), torchvision>0.21 blocks JetPack torch<=2.5 (#819 closed as "not planned"). Training is fine, deployment is broken — exactly reflex's wedge. Fine-tune with lerobot → export with reflex is the right split.

**HuggingFace Trainer** — rarely used for VLAs. Incompatible with flow-matching loss schedules without heavy subclassing. Not a realistic target.

**PyTorch Lightning** — UNCITED, but essentially absent in robotics repos (survey of openpi, lerobot, Isaac-GR00T, OpenVLA, X-VLA shows zero Lightning). Don't build on Lightning.

**FSDP vs DeepSpeed.** OpenVLA uses FSDP-native ([github.com/openvla/openvla](https://github.com/openvla/openvla)); OpenPI uses JAX/Flax (not PyTorch at all); lerobot-train defaults to plain `Accelerator` / single-GPU. For the 2–4 A100 customer, FSDP FULL_SHARD on 7B models is the 2026 consensus ([vrlatech blog](https://vrlatech.com/deepspeed-vs-pytorch-fsdp-which-distributed-training-framework-in-2026/), [HF deepspeed-to-fsdp blog](https://huggingface.co/blog/deepspeed-to-fsdp-and-back)). DeepSpeed adds ops complexity without clear wins under 70B.

**Mixed precision.** bf16 is the 2025–26 standard across openpi, Isaac-GR00T, X-VLA ("set `policy.dtype=bfloat16` to avoid OOM" — [lerobot X-VLA docs](https://huggingface.co/docs/lerobot/en/xvla)). fp16 causes flow-matching loss instability in our [ONNX export gotchas memory](project_reflex_vla_onnx_export_gotchas.md); training should stay bf16 even though deployment flips to fp16.

**reflex decision.** Wrap lerobot-train (`--backend lerobot`) + openpi JAX (`--backend openpi`). bf16 default. FSDP auto-enabled when nGPU≥2 and model≥2B. No Lightning, no DeepSpeed.

### A5. Distillation & efficiency as fine-tune phases

**SnapFlow ([arxiv 2604.05656](https://arxiv.org/abs/2604.05656), April 2026).** Self-distills flow-matching VLAs (pi0, pi0.5, SmolVLA) to 1 NFE. On pi0.5/LIBERO: 98.75% vs 97.75% teacher, 9.6× denoising speedup, 274ms → 83ms latency. ~12h on a single GPU. No teacher, no architecture change.

**OpenVLA-OFT ([arxiv 2502.19645](https://arxiv.org/abs/2502.19645)).** L1 regression + parallel decoding + action chunking → 26× faster action generation, 76.5% → 97.1% success on LIBERO. Proof that "better fine-tune recipe" beats "bigger model" in 2025.

**VITA-VLA ([arxiv 2510.09607](https://arxiv.org/abs/2510.09607)).** Action-expert distillation into VLMs — 97.3% LIBERO (vs the teacher's ~86%). Useful when the customer has a teacher policy and wants a smaller student.

**Quantization-aware fine-tune.** [arxiv 2412.01034](https://arxiv.org/abs/2412.01034) "Quantization-Aware Imitation-Learning for Resource-Efficient Robotic Control" — 4-bit weight-quant on Jetson AGX Orin with **2.5× energy savings, accuracy preserved**. PyTorch now has native QAT for LLMs ([PyTorch blog 2024](https://pytorch.org/blog/quantization-aware-training/)). This will become table-stakes for edge VLAs in 2027.

**reflex decision.** Ship `--distill snapflow|oft|none` (default `none`; promote `snapflow` for flow-matching bases in a follow-up). QAT is post-process today — but architect so `--qat` slots in: don't assume `reflex export` is pure bf16→int8 PTQ; leave a hook for QAT fake-quant nodes.

### A6. Safety & eval integration

**Validation during training.**
- LIBERO mid-training rollouts: [Isaac-GR00T finetune guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md) notes "validation during fine-tuning is disabled by default" — a footgun.
- VLA-RFT ([openreview Jaut99EHeu](https://openreview.net/forum?id=Jaut99EHeu)) retains a small-weight flow-MSE as auxiliary supervision during RL fine-tune — stability trick worth lifting for imitation fine-tune too.
- Action-MSE on held-out episodes is the cheapest and most informative proxy (UNCITED as community consensus; seen in every openpi script).

**Calibration.**
- [Zollo & Zemel arxiv 2507.17383](https://arxiv.org/abs/2507.17383) "Confidence Calibration in Vision-Language-Action Models" — first dedicated study. Two techniques that actually work: prompt ensembles (zero-training, ensemble over 5 rephrased instructions) and action-wise Platt scaling. Reduces ECE across 3 LIBERO suites × 3 VLA variants.
- Path-deviation detection via attention heads alone ([arxiv 2603.13782](https://arxiv.org/abs/2603.13782)): 3 specific heads → 44.6% deviation detection, 11.7% FPR. Training-free. Reflex should compute this per-checkpoint as a post-training report card.

**Training-time safety (weight clipping, action-magnitude constraints).** UNCITED — not a published pattern in VLAs. Action-magnitude constraints at deploy time are industry standard (Tesla FSD, Physical Intelligence safety layer per blog posts — UNCITED), but baking them into the loss during fine-tune is not a thing today.

**reflex decision.** `reflex finetune` emits a **training report** at the end:
1. Offline action-MSE vs held-out 10% split.
2. ECE via prompt-ensemble on a frozen 50-episode eval set.
3. LIBERO rollout success (if LIBERO env available; opt-in to avoid Modal/Isaac dependency).
4. Calibration curve PNG.
This feeds directly into the `reflex export` step — models with action-MSE > threshold or ECE > 0.2 emit a loud warning, and the CLI requires `--force-export-anyway` to proceed.

### A7. Multi-GPU / distributed training for small orgs

**Reality check on 2–4 A100s.**
- OpenVLA: ~15h on 1×A100 with LoRA; 8×A100 for 1–2 days for OFT full fine-tune.
- SmolVLA: 450M — single A100 is fine. 20k steps in ~a day.
- pi0 / pi0.5 / GR00T N1.6 @ 3B: 2–4×A100 FSDP is the sweet spot. Full fine-tune ~3 days, LoRA ~12h.
- Modal/RunPod A100 spot pricing (April 2026 UNCITED — verify): ~$1.30/hr. Full pi0 fine-tune ~$100 on spot, LoRA ~$20.

**reflex decision.** Default training target: 0 GPUs = refuse; 1 GPU = force `--mode lora`; 2–4 GPUs = FSDP FULL_SHARD + bf16; ≥8 GPUs = + activation checkpointing. Emit `modal` and `runpod` templates (`reflex finetune --template modal`) for customers without cluster ops.

---

## Section B — Non-obvious findings

1. **LoRA rank for VLAs is 4–16× higher than for LLMs.** LoRA-SP ([arxiv 2603.07404](https://arxiv.org/abs/2603.07404)) shows r≈128 near-full-rank is needed for cross-embodiment transfer; r=4–8 LLM practice *actively fails* on robotics. Reflex's default r=32 is a compromise; cross-embodiment mode must go higher.

2. **pi0.5 `use_quantile_norm` is the #1 silent fine-tune killer.** Default-on in pi0.5, but on small datasets (< 500 episodes) it tanks performance — [openpi#763](https://github.com/Physical-Intelligence/openpi/issues/763). reflex must default to mean-std and escalate to quantile only above a dataset-size threshold.

3. **OpenVLA-OFT's L1 regression matches diffusion-based fine-tuning for success rate.** Against community expectation that "flow-matching / diffusion is essential for continuous actions," a plain L1 regression + parallel decoding + chunking matches or exceeds it ([arxiv 2502.19645](https://arxiv.org/abs/2502.19645)), at 26× throughput. This may deprecate flow-matching as a default action head within 2 years.

4. **Three attention heads can detect path deviation with 44.6% recall, training-free ([arxiv 2603.13782](https://arxiv.org/abs/2603.13782)).** For reflex's safety story, this means runtime hallucination detection is *achievable without extra models* — embedded into `reflex serve` as a "guard head" in 2026–27.

5. **NVIDIA Isaac-GR00T ships no official LoRA.** Despite GR00T N1.6 being 3B-class, the [official fine-tune doc](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md) assumes full fine-tune on 40GB+ GPUs. This is a real gap — SO-100 hobbyists cannot fine-tune GR00T without LoRA. reflex can serve this need.

6. **VLM2VLA's action-as-language approach may eliminate catastrophic forgetting entirely** ([arxiv 2509.22195](https://arxiv.org/abs/2509.22195)). By encoding actions as text tokens, the action task distribution becomes compatible with the VLM's pretrain corpus and LoRA alone preserves 100% of semantic reasoning. If this direction wins, the "action-head" abstraction becomes vestigial by 2028.

7. **Catastrophic forgetting requires only 6.2% rehearsal fraction to fully resolve** ([arxiv 2512.13706](https://arxiv.org/abs/2512.13706)). The industry-standard 10% mix is 60% over-provisioned. reflex can default to 5% rehearsal and still match best-in-class.

8. **SmolVLA's action expert is 75% the width of the VLM hidden size, NOT 100%** ([HF blog](https://huggingface.co/blog/smolvla)). Subtle — any LoRA targeting "all linear" layers will have mismatched rank capacity per module. Target-module config must be explicit.

9. **pi0.5's language grounding degradation on small datasets is reproducible and unresolved upstream** ([openpi#768](https://github.com/Physical-Intelligence/openpi/issues/768)). 250 videos / 90 min of data → model replays trajectories, ignores language. The fix is dataset size (UNCITED threshold ~1000 episodes for pi0.5), co-training (openpi's official approach), or VLM2VLA-style action-as-language.

10. **LeRobotDataset v3.0 is at >54.6% adoption of HF robotics datasets within months of release** ([kamenski.me Oct 2025](https://www.kamenski.me/articles/lerobot-datasets-oct-2025)). RLDS adoption growth has stalled. The format war is effectively over for the open ecosystem — reflex betting on LeRobotDataset is non-controversial.

---

## Section C — 3-year evolution hypothesis

### What a production VLA fine-tune pipeline NEEDS in 2027–2029

#### C1. Pluggable action heads (tokenized vs flow-matching vs DDPM vs regression)

The action-head landscape is in flux: pi0/pi0.5 flow-matching, GR00T DiT-diffusion, OpenVLA-OFT L1 regression, pi0-FAST + VLM2VLA token-based. Each has different fine-tune losses, numerical scales, and eval signatures.

**Architecture pattern.** reflex's finetune config should express action head as a pluggable strategy:
```python
ActionHeadStrategy(
    type="flow_matching" | "ddpm" | "l1_regression" | "fast_tokens" | "action_language",
    loss_fn="flow_matching_loss" | "mse" | "l1" | "cross_entropy",
    normalization="quantile" | "mean_std" | "none",
    sampling_steps=10 | 1,  # supports SnapFlow-style distillation
)
```
Abstracting this now means pi1, xVLA, GR00T N2 slot in without re-architecting.

#### C2. Evolving dataset formats

LeRobotDataset v3 is dominant *today*. By 2028:
- NVIDIA InternData-A1 + Cosmos synthetic data will force a hybrid real+sim format ([Codatta blog Dec 2025](https://blog.codatta.io/ai/2025/12/19/open-robotic-data-at-scale-ecosystem-formation-and-implications.html)).
- Cross-embodiment datasets will add explicit morphology metadata (joint limits, kinematics).
- Video-first datasets (1000+hr human video → action) per [WIYH arxiv 2512.24310](https://arxiv.org/abs/2512.24310).

**Architecture pattern.** Ingest layer (`reflex.data.ingest`) separate from training layer (`reflex.train.loop`). Ingest emits a canonical `TrajectorySample(obs, state, action, lang, meta)` stream. Any new format becomes one adapter. Avoid deep coupling between parquet readers and policy heads.

#### C3. New optimizer families

Muon ([arxiv 2502.16982](https://arxiv.org/abs/2502.16982) — UNCITED, verify) and Sophia are already in LLM pretrain. Robotics is 18 months behind LLM optim trends. By 2027, AdamW-only will be a handicap.

**Architecture pattern.** Optimizer registry (`reflex.optim.registry`) parallel to `torch.optim`. Default AdamW. Accept `--optimizer muon|sophia|adamw`. Hyperparams flow through config.

#### C4. Training-time quantization becoming standard

PyTorch has shipped native LLM QAT ([PyTorch blog 2024](https://pytorch.org/blog/quantization-aware-training/)). For edge-first robotics (Jetson Orin Nano is reflex's stated FP16 verified target), the natural next step is int8/int4 QAT. Arxiv [2412.01034](https://arxiv.org/abs/2412.01034) shows 2.5× energy savings with accuracy preserved.

**Architecture pattern.** `reflex finetune --qat int8` flag wires PyTorch's `torchao.quantization.qat` into the training loop. `reflex export` must accept the QAT-emitted fake-quant ops (today it's bf16 → fp16 PTQ, which will break on QAT inputs). Preserve an "export-graph shape" contract so the export step is agnostic to whether fake-quant ops are present.

#### C5. Other patterns to architect for

- **RL fine-tune phase.** SimpleVLA-RL ([ICLR 2026](https://github.com/PRIME-RL/SimpleVLA-RL)), VLA-RFT, AcceRL ([arxiv 2603.18464](https://arxiv.org/abs/2603.18464)), πRL ([arxiv 2510.25889](https://arxiv.org/abs/2510.25889)) — RL post-training after imitation is becoming standard. reflex should reserve a `--phase imitation|rl|distill` slot in the CLI. Phase pipelines chain together.
- **World-model co-training.** [arxiv 2512.18007](https://arxiv.org/abs/2512.18007) "Joint Learning with Motion Image Diffusion" — dual-head (action + motion image) during training boosts LIBERO to 97.5%. Expect these auxiliary heads to multiply. reflex's training loop should accept auxiliary loss registrants.
- **Safety eval as first-class output.** Calibration, path-deviation attention heads, action-magnitude distributions — all become part of the "is this model deployable" gate. reflex's export pipeline already verifies `cos=+1.0` numerical parity; extend this to a multi-metric ship gate.

---

## Section D — Honest unknowns

1. **What rank actually works for GR00T N1.6 DiT action head?** No paper, no community config. Full fine-tune is NVIDIA's only published path. Reflex may be the first to publish a LoRA recipe for the 32-layer DiT. Budget 1–2 weeks of trial to find it.

2. **Does quantile-norm vs mean-std actually matter at the pi0 level, or is it just pi0.5?** [openpi#763](https://github.com/Physical-Intelligence/openpi/issues/763) is specifically pi0.5. Pi0 default is mean-std; small-dataset behavior there is unvalidated.

3. **Calibration (ECE) on flow-matching VLAs.** [Zollo & Zemel arxiv 2507.17383](https://arxiv.org/abs/2507.17383) studied pi0-FAST/OpenVLA (discrete-action). Flow-matching models don't have a natural per-token confidence — calibration methodology is not yet defined for them. reflex shipping an ECE number on pi0/SmolVLA is *inventing methodology*. Plan for internal validation before publishing as a customer metric.

4. **Cross-embodiment transfer limits.** LoRA-SP shows cross-embodiment needs r≈128. But what's the actual success rate gap between "native" vs "cross-embodiment" after fine-tune? Paper reports ≤31.6% improvement over standard LoRA — not an absolute number. Customer demo on an unseen embodiment may disappoint even at r=128.

5. **LeRobotDataset v3 long-horizon streaming stability.** Streaming mode is new. For a multi-day fine-tune that re-streams a 500GB dataset, failure modes (HF hub outages, partial-chunk corruption) are not documented. Expect to ship `--streaming-retry-policy` logic reactively.

6. **SnapFlow on pi0-FAST (discrete actions).** Paper validates on flow-matching only. Applying it to tokenized-action VLAs (pi0-FAST, VLM2VLA-style) is an open question. If customers want "one-step deployment" on a pi0-FAST checkpoint, reflex has no distillation path today.

7. **Multi-task / multi-embodiment fine-tune dynamics.** Every paper above is single-task-ish. A customer with 5 tasks × 3 robots fine-tuning simultaneously is under-specified in the literature. Interference between tasks is empirically known to be ugly (UNCITED, but seen in every multi-task paper ablation). reflex may need a scheduling layer that the research doesn't provide.

8. **Do FSDP config defaults transfer from LLM to VLA?** FSDP FULL_SHARD works great for decoder-only LLMs. VLAs have a vision encoder + LM + action head with different parallelism-friendliness. Activation-recompute boundaries for the action expert specifically are not documented. Plan for 1–2 days of tuning on each new 3B+ base model.

9. **What breaks first when pi1 or GR00T N2 ships?** Almost certainly the action-head abstraction (new head type), tokenization (FAST → something), and optimizer choice. Reflex's finetune config must be JSON-schema-validated with forward-compat fallbacks; silent downgrade on unknown fields is worse than loud error.

10. **Customer expectations on "dataset size floor."** 50–200 demonstrations is the public claim for LoRA fine-tune ([DigitalOcean tutorial](https://www.digitalocean.com/community/tutorials/vision-language-action-finetuning-robotics), [LeRobot v0.5.0 notes](https://awesomeagents.ai/news/lerobot-v050-humanoid-open-source/)). Real pi0.5 failure at 250 episodes / 90 min ([openpi#768](https://github.com/Physical-Intelligence/openpi/issues/768)) contradicts this. Publish floor guidance *per base model*, not as a blanket number.

---

## Key citations (by research area)

**LoRA configs & failure modes:**
- LoRA-SP: [arxiv 2603.07404](https://arxiv.org/abs/2603.07404)
- α-LoRA: [arxiv 2510.21345](https://arxiv.org/abs/2510.21345)
- Towards Accessible Physical AI: [arxiv 2512.11921](https://arxiv.org/abs/2512.11921)
- lerobot PeftConfig: [src/lerobot/configs/default.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/configs/default.py)
- openpi LoRA example: `pi0_aloha_pen_uncap_lora` in [openpi/src/openpi/training/config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)

**Catastrophic forgetting:**
- VLM2VLA: [arxiv 2509.22195](https://arxiv.org/abs/2509.22195)
- Mixed training NLI: [arxiv 2512.13706](https://arxiv.org/abs/2512.13706)
- pi0.5 co-training: [arxiv 2504.16054](https://arxiv.org/abs/2504.16054)
- Bayesian PEFT: [arxiv 2402.12220](https://arxiv.org/abs/2402.12220)

**Architecture details:**
- SmolVLA: [arxiv 2506.01844](https://arxiv.org/abs/2506.01844), [HF blog](https://huggingface.co/blog/smolvla)
- pi0 / pi0-FAST: [arxiv 2410.24164](https://arxiv.org/abs/2410.24164), [HF blog](https://huggingface.co/blog/pi0)
- pi0.5: [arxiv 2504.16054](https://arxiv.org/abs/2504.16054)
- GR00T N1: [arxiv 2503.14734](https://arxiv.org/abs/2503.14734)
- OpenVLA-OFT: [arxiv 2502.19645](https://arxiv.org/abs/2502.19645)

**Dataset formats:**
- LeRobotDataset v3: [HF blog](https://huggingface.co/blog/lerobot-datasets-v3), [HF docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- Open X-Embodiment: [arxiv 2310.08864](https://arxiv.org/abs/2310.08864)
- Codatta ecosystem analysis: [blog Dec 2025](https://blog.codatta.io/ai/2025/12/19/open-robotic-data-at-scale-ecosystem-formation-and-implications.html)

**Distillation & efficiency:**
- SnapFlow: [arxiv 2604.05656](https://arxiv.org/abs/2604.05656)
- VITA-VLA distillation: [arxiv 2510.09607](https://arxiv.org/abs/2510.09607)
- QAT imitation learning: [arxiv 2412.01034](https://arxiv.org/abs/2412.01034)
- PyTorch QAT blog: [pytorch.org 2024](https://pytorch.org/blog/quantization-aware-training/)

**Safety & eval:**
- VLA calibration: [arxiv 2507.17383](https://arxiv.org/abs/2507.17383)
- Path deviation heads: [arxiv 2603.13782](https://arxiv.org/abs/2603.13782)
- VLA-RFT stability: [openreview Jaut99EHeu](https://openreview.net/forum?id=Jaut99EHeu)
- T-MEE action error reshape: [arxiv 2602.04228](https://arxiv.org/abs/2602.04228)

**RL & advanced fine-tune:**
- SimpleVLA-RL: [github PRIME-RL](https://github.com/PRIME-RL/SimpleVLA-RL)
- πRL: [arxiv 2510.25889](https://arxiv.org/abs/2510.25889)
- AcceRL: [arxiv 2603.18464](https://arxiv.org/abs/2603.18464)
- InstructVLA: [github InternRobotics](https://github.com/InternRobotics/InstructVLA)

**Survey / landscape:**
- VLM-based VLA survey: [arxiv 2508.13073](https://arxiv.org/abs/2508.13073), pure VLA survey: [arxiv 2509.19012](https://arxiv.org/abs/2509.19012), ICLR 2026 VLA state: [Reuss blog](https://mbreuss.github.io/blog_post_iclr_26_vla.html)
