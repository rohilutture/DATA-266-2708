# HW4.5 — Training throughput and memory engineering (BERT, Twitter sentiment)

One-GPU run (RTX 4090). Part C is skipped (optional per instructor).

## Setup (on the lab machine)
```bash
python -m venv .venv      && source .venv/bin/activate      && pip install -r requirements-train.txt && deactivate
python -m venv .venv-vllm && source .venv-vllm/bin/activate && pip install -r requirements-vllm.txt  && deactivate
```
vLLM pins its own torch version, hence two virtualenvs.

## Before you run anything
1. Open `config.py`. `SLICE`, `SEED` and the dataset must be the HW4.a values (the instructor will compare). The defaults
   are tweet_eval/sentiment, rows 0..4999, seed 42. Change them now if your HW4.a used something else.
2. Dry run (about 5 minutes): `HW45_QUICK=1 PY_TRAIN=.venv/bin/python PY_VLLM=.venv-vllm/bin/python ./run_all.sh`
   then delete `results/`, `RUN_LOG.txt`, `uuid_capture_*.txt`, `figures/`, `METRICS.md`, `GPU_HOURS.md` before the real run.

## Real run
```bash
PY_TRAIN=.venv/bin/python PY_VLLM=.venv-vllm/bin/python ./run_all.sh
```
Produces `results/<gpu>/*.json`, `figures/*.png`, `METRICS.md`, `GPU_HOURS.md`, `RUN_LOG.txt`, `uuid_capture_<gpu>.txt`.
Individual stages: `hw4a_train.py`, `partAB.py {all|sweep NAME|train NAME|accum}`, `serve_bench.py --engine {hf|vllm}`, `kv_cache.py`.

## After the run (what is still on you)
- `results/<gpu>/HW4A_REPORT.md`: write the five one-line diagnoses.
- `METRICS.md`: fill every **TODO** (apply-first / skip technique, chosen config, 3–4 sentence latency answer, HW4-vs-HW4.5 note).
- Fill `RESERVATION.md` and `AI_USE.md`, add the booking screenshot.
- Tag and push: `git init && git add -A && git commit -m "HW4.5" && git tag hw4-5` (do not commit `results/*/hw4a_model`; it is large).

## Methodology notes (cite in METRICS.md)
- Probes time 12 optimiser steps after 3 warm-ups on random train rows already resident on the GPU, so throughput reflects GPU compute, not data loading. Batches larger than the 3,500-row train split sample with replacement.
- Peak memory = `torch.cuda.max_memory_allocated` (reserved is also stored). Each probe runs in a fresh process.
- Sequences are padded to 128 tokens, so memory is deterministic.
- AMP uses bf16 (no loss scaler needed on a 4090). Checkpointing uses non-reentrant `torch.utils.checkpoint`.
- Single seed per run: accuracy differences under ~1.5 points on a 750-row test set are noise.
