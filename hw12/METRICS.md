# METRICS - HW12

SID4 = 2708, SEED = 2708 (SLICE = 708, HP_ID = 2, CLS_A = 8, CLS_B = 9; not used directly in this homework).

**Table HW11.1 — GPTQ comparison on the same LLM/task** (`gpt2` @ revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, Wikitext-2 test perplexity, batch 1 x 128 tokens, RTX 3060 Laptop 6 GB; latency = median of 3 isolated runs, one model per process).

| Configuration | Quality metric (PPL) | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---|---|---|---|
| Base LLM (fp16) | 25.18 | 13.2 / 14.2 | 0.262 | 240.8 |
| GPTQ-quantized LLM (4-bit, g=128, desc_act) | 29.94 | 39.9 / 42.5 | 0.175 | 121.1 |

(The table printed by the code cell above is from the in-notebook run, where other models were still resident on the GPU, and it shows GPTQ p50 = 75 ms. The isolated re-measurement is `measure_isolated.py` / `isolated_measurements.jsonl`; GPTQ latency on this setup is noisy, 39-48 ms p50 across the three isolated runs.)

**Findings.** GPTQ raised Wikitext-2 perplexity from 25.18 to 29.94 (about +19%), halved the on-disk size (240.8 to 121.1 MB) and cut peak GPU memory by about a third (0.262 to 0.175 GB). The surprise is latency: the quantized model was about 3x *slower* (p50 about 13 to 40 ms), not faster. On this Windows setup there were no fused CUDA/Triton kernels, so `gptqmodel` used its pure-PyTorch `TorchLinear` backend, which unpacks the 4-bit weights to fp16 on every forward pass, and `desc_act=True` adds a column permutation on top. The size saving is also below 4x because the embeddings (`wte`, `wpe`) are not quantized.

**Table HW11.2 — the separate distillation comparison on AG News** (test set, 7,600 examples; latency = median of 3 isolated runs, one model per process).

| Configuration | Accuracy | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---|---|---|---|
| Teacher (BERT-base) | 0.9443 | 9.6 / 11.1 | 0.422 | 418.6 |
| Best distilled student (DistilBERT, T=2, alpha=0.5) | 0.9416 | 4.6 / 5.5 | 0.263 | 256.1 |

(In the code cell above, teacher and student both show peak memory 1.080 GB. That is a measurement artifact: both models were resident on the GPU in the same process. The values in this table are from the isolated re-measurement.)

**Sweeps** (every student trained for **1 epoch**, 3,750 steps at batch 32, because of the time limit on a 6 GB laptop GPU; one seed, SEED = 2708). Temperature sweep at alpha = 0.5: T=1: 0.9392, **T=2: 0.9416**, T=5: 0.9409, T=10: 0.9387. Loss-weighting sweep at T=2: alpha=0.25: 0.9411, **alpha=0.5: 0.9416**, alpha=0.75: 0.9412. All of these lie within about 0.3 points (roughly 20 of 7,600 test examples) of each other from a single seed, so the "best" settings are not statistically distinguishable from the others. The distilled student is 0.28 points below the teacher, about 2.1x faster at p50, 39% smaller on disk and uses 38% less peak memory.

Source files: `HW12_GPTQ_Distillation_executed.ipynb`, `measure_isolated.py`, `isolated_measurements.jsonl`, `RUN_LOG.txt`.
