# AI_USE - HW12 (GPTQ quantization & knowledge distillation)

**Tool:** Claude Code (Claude Sonnet 5.5) running locally in the Claude desktop app, driven by Rohil Utture.

## What the AI did
- Set up the environment on the RTX 3060 Laptop (6 GB): CUDA PyTorch, `transformers`, `datasets`, `gptqmodel`; extracted `teacher_model/`.
- Ran the provided notebook end to end and monitored the run; diagnosed and fixed slowdowns (see "Deviations").
- Re-measured latency / peak GPU memory with one model per process (`measure_isolated.py`, 3 runs, median) because the in-notebook peak-memory numbers for the teacher and student were contaminated (both models resident on the GPU at once).
- **Drafted** the interpretation text in the notebook (Table HW11.1 findings, the five-example commonality, the quantization-vs-distillation paragraph), `METRICS.md`, `RUN_LOG.txt` (filtered notebook console output) and this file. The numbers in them come from the executed notebook and `isolated_measurements.jsonl`.

## Deviations from the provided notebook / assignment (please read)
1. **1 epoch per student instead of 2** (3,750 steps, batch 32), for all 7 trained students, because of the submission deadline on a 6 GB laptop GPU. All other settings are identical across runs (SEED = 2708).
2. **Pre-tokenized training set** (`AGNewsDistillDataset` tokenizes once in `__init__` instead of per item). Same tokens and padding; only faster data loading.
3. **Each trained student is moved to CPU after evaluation** (`temp_models[T] = model.cpu()`, same for alpha). Keeping all students on the 6 GB GPU caused VRAM spill into shared memory and a roughly 8x slowdown. The best student is moved back to the GPU before the final evaluation.
4. Added printing of the `gpt2` revision hash (`607a30d783dfa663caf39e06633721c8d4cfcd7e`) as required by the assignment.
5. **Environment workarounds:** `gptqmodel` 7.12.2 asks for torch >= 2.15, which has no Windows CUDA wheel; it was run on torch 2.13.0+cu126 (a version its own safety check lists as supported). Its `pypcre` dependency needs Visual Studio Build Tools, which are not installed, so a small local shim module `pcre.py` mapping onto Python's `re` was placed in the virtualenv. GPTQ therefore ran with the pure-PyTorch `TorchLinear` kernel, which is why GPTQ latency is worse than the base model.
6. Peak-memory and latency values in the final tables are from the isolated re-measurement, not from the notebook's own table cells (those are still in the executed notebook).

## Not verified / limitations
- Single seed; sweep differences are about 0.3 accuracy points or less and are not statistically distinguishable.
- The five error examples are the first five by index, not a random sample.
- `facebook/opt-125m` (the model in the Hugging Face GPTQ docs example) was *not* used; the notebook uses `gpt2`. Whether the instructor specified a different base model was not confirmed.
- The interpretation text is an AI draft and should be reviewed and rewritten in the student's own words before submission.
