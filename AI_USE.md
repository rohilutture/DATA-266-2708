# AI_USE.md

## Tools used
- Claude (Anthropic) generated the initial code for this repository (data loading, training/probing harness, Part A/B sweeps,
  serving and KV-cache benchmarks, figure and METRICS generators) and the templates for these documents.
- Claude Code (Claude Sonnet 5.5) also did the setup and run in this session: created the virtualenvs, ran the dry run and the
  full run, fixed the problems below, and at my request wrote the free-text parts of METRICS.md, the five misclassification
  diagnoses in HW4A_REPORT.md, the GPU_HOURS.md session table, RESERVATION.md and this file.

## What I did myself
- Asked for the run and chose the settings; dataset, SLICE and SEED in `config.py` were left at the defaults (tweet_eval/sentiment, rows 0..4999, seed 42).
- The free-text analysis, the diagnoses and the latency answer were drafted by Claude from the measured numbers, not written by me; review them before submitting.
- Reservation details were not available to Claude and are marked as not recorded in `RESERVATION.md`.

## Changes made to the scripts / environment
- `capture_env.sh`: use the training venv's python/pip (it used the system python, so the torch version was missing).
- `common.py`: cap CUDA memory at 90% of VRAM (`set_per_process_memory_fraction(0.9)`) so oversized batches raise OOM; under WSL/WDDM they otherwise spilled into system RAM and gave invalid numbers (40 GiB peak on a 24 GiB card).
- Environment only: venvs built with `uv` on Python 3.12 in the Linux filesystem; vLLM run with `VLLM_USE_FLASHINFER_SAMPLER=0` and a `ziglang` C-compiler wrapper (`CC`) because the machine has no gcc/nvcc.

## What AI did NOT do
- All numbers in METRICS.md come from runs on this machine's RTX 4090; none were produced or estimated by the AI.

## Verification
- Ran `HW45_QUICK=1` dry run first (passed after fixes), then the full run; checked there were no tracebacks, no negative memory savings, and that the batch sweeps end in a real OOM.
- Observed the invalid pre-cap sweep (bs=512 at 40 GiB) and reran from scratch after the cap; KV-cache measured sizes match the analytic formula.
