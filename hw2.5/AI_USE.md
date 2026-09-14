# AI_USE.md — HW2.5

**Tool:** Anthropic Claude (chat interface) **Date:** 2026-09-14
**Author:** Rohil Utture

I used an AI assistant on this assignment. This file documents exactly what it
did and what I did, so the boundary is clear.

## What the AI assisted with

- **Benchmark code.** The AI wrote the benchmarking suite I ran:
  `benchlib.py` (timing, GPU/UUID provenance, theoretical-peak table, CSV +
  RUN_LOG helpers), `part_b_precision.py`, `part_c_roofline.py`,
  `part_d_attention.py`, `part_e_sustained.py`, the post-hoc `analyze_e.py`,
  and the `verify.py` / `probe.py` check snippets.
- **Environment setup and debugging.** It helped me pick the correct PyTorch
  build for the RTX 5090 (cu128 wheel, native `sm_120`, verified — not PTX
  fallback), set up the venv, split the pip install across the PyTorch and PyPI
  indexes, and diagnose the Windows CUDA **sysmem-fallback** setting that was
  silently masking the Part D OOM boundary (a 33 GB "fit" on a 32 GB card).
- **Result interpretation.** It flagged and explained findings I then verified
  against my data: the boost-clock cause of the >100%-of-peak figures in Part B,
  the `a = 4·B·H` peak-memory model in Part D, the roofline placement in Part C,
  and the power-limited (not thermal) throttling in Part E.
- **Scaffolding and sources.** It drafted the `README`, the `PROGRESS` log, and
  the structure of `METRICS.md`, and located the NVIDIA source documents for
  Part A.

## What I did myself

- Reserved and ran every benchmark on the RTX 5090 workstation. **Every
  measurement, CSV, figure, and log entry is from my own runs** on GPU UUID
  `GPU-41051861-72ca-e198-fd60-2bf89fd2a67b`.
- Made the experimental choices — e.g. moving to `heads=32` so the attention
  OOM boundary fell inside the tested sequence lengths — and set the CUDA
  sysmem-fallback policy in the NVIDIA Control Panel.
- Captured provenance (`nvidia-smi -q`, the `verify.py` native-kernel check) and
  sanity-checked each run before trusting it.
- Reviewed and ran all AI-written code, and wrote the analysis in `METRICS.md`
  in my own words after confirming each interpretation against my results.

## Integrity

- **No measurements were generated or fabricated by the AI.** All numbers come
  from my hardware runs; the AI could not run the GPU and did not produce any
  results.
- All AI-generated code was read, executed, and validated by me before its
  output was used in any deliverable.
