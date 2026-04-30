# 04 — Record `/act` traces, browse them, replay against another model

**What you'll see:** flip on JSONL trace recording on a running `reflex serve`, list captured traces, replay one against a different export to compare action chunks.

**Requires:** A running `reflex serve` (see [02-deploy-smolvla-jetson.md](02-deploy-smolvla-jetson.md)) and the base `reflex-vla` install.

## Why record traces

- **Debugging:** when a robot does something weird, you want to replay the exact `/act` request that produced it
- **Regression testing:** capture traces on a known-good model, replay them against a new export, diff the actions
- **Compliance:** the EU AI Act-style audit trail Reflex's safety wedge writes (per-action SHA-256 hash chain in `--safety-config` mode)

## Record

Start the server with `--record`:

```bash
reflex serve ./reflex_export --record /tmp/traces
```

The server writes JSONL files like:

```
/tmp/traces/20260427-091823-3a4f5b6c-7d8e9f0a.jsonl.gz
```

Filename: `<YYYYMMDD-HHMMSS>-<model_hash>-<session_id>.jsonl[.gz]`. One file per server session. Compressed by default; pass `--record-no-gzip` for plain `.jsonl`.

Each line is one `/act` request + response, schema documented in [docs/record_replay.md §D.1](https://github.com/FastCrest/reflex-vla/blob/main/docs/record_replay.md). Sensitive image bytes are SHA-256-hashed (not stored raw) by default — pass `--record-images full` to capture raw images (large, only for debug).

Now hit `/act` from your robot or test harness as usual. Every call gets logged.

## List traces

```bash
reflex inspect traces
```

```
Recorded traces (3 of 3 total)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Modified           ┃ File                          ┃ Task                     ┃ Records ┃ Size   ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ 2026-04-27 14:20  │ 20260427-142053-…-…jsonl.gz   │ pick up the red cup      │ 87      │ 412 KB │
│ 2026-04-27 13:45  │ 20260427-134505-…-…jsonl.gz   │ stack the blocks         │ 134     │ 651 KB │
│ 2026-04-27 09:18  │ 20260427-091823-…-…jsonl.gz   │ open the drawer          │ 52      │ 198 KB │
└────────────────────┴───────────────────────────────┴──────────────────────────┴─────────┴────────┘
```

Filters:

```bash
reflex inspect traces --since 24h
reflex inspect traces --task "pick up the red cup"
reflex inspect traces --since 7d --limit 20
reflex inspect traces --dir /custom/trace/path
```

## Replay against a different export

Suppose you have two exports on disk:
- `./reflex_export_v1/` — pi0.5 teacher (10-step)
- `./reflex_export_v2/` — SnapFlow 1-step student

Replay the captured trace against the student:

```bash
reflex replay /tmp/traces/20260427-142053-….jsonl.gz \
    --model ./reflex_export_v2 \
    --diff actions
```

Output (abbreviated):

```
Replay summary
  records:        87
  reproduced:     87
  per-record diffs:
    record_idx  cos_sim   max_abs_diff
    0           0.99987   3.2e-04
    1           0.99991   2.7e-04
    ...
    86          0.99983   3.8e-04
  aggregate:
    cos_sim_mean: 0.99988
    max_abs_diff_p99: 4.1e-04
```

If the student's actions match the teacher's at cos≈1.0 across all records, the distillation is faithful. Drift > a few percent → regression to investigate.

## Or use the chat agent

```bash
reflex chat
you › list my recent traces from today
you › replay the most recent trace against ./reflex_export_v2 and tell me if it matches
```

The agent calls `list_traces` then `replay_trace` for you and summarizes the diff.

## Caveats

- **Image redaction** — by default only SHA-256 hashes are stored, so you can detect *which* image was sent but not reproduce pixel-exact replay. Use `--record-images full` for full-fidelity replay (storage cost: ~100x larger traces).
- **Timing** — traces include latency metadata but replay always runs as fast as the target model can do inference; replay isn't a load test.
- **Schema versioning** — schema v1 documented in `docs/record_replay.md §D.1`. v2 (with executed-action vs predicted-action separation) lands when A2C2 wedge ships per-step.
