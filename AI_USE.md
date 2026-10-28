# AI_USE.md

## Tools used
- Claude (Anthropic) generated the initial code for this repository (data loading, training/probing harness, Part A/B sweeps,
  serving and KV-cache benchmarks, figure and METRICS generators) and the templates for these documents.

## What I did myself (edit to match reality)
- Chose / confirmed the dataset, SLICE and SEED in `config.py` and checked they match HW4.a.
- Reserved and ran everything on the RTX 4090; captured `nvidia-smi -L` (see `uuid_capture_*.txt`).
- Read through the generated scripts before running them; fixed: <list anything you changed>.
- Wrote the free-text analysis in METRICS.md (the TODO lines), the five misclassification diagnoses, and the
  3–4 sentence latency-vs-throughput answer.

## What AI did NOT do
- All numbers in METRICS.md come from my own runs on my reserved GPU; none were produced or estimated by the AI.

## Verification
- <e.g. ran `HW45_QUICK=1 ./run_all.sh` first; spot-checked peak memory against `nvidia-smi`; reran one probe to check variance>
