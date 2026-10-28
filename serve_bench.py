"""Serving benchmark: HF pipeline (batch 1) vs vLLM (batch 1/8/32).   Table HW4.1 and Part D.

  python serve_bench.py --engine hf
  python serve_bench.py --engine vllm
Run it in the vLLM virtualenv (vLLM pins its own torch). Results -> results/<gpu>/serve_<engine>.json

Definitions (also written into METRICS.md):
  * every request generates exactly SERVE_MAX_NEW_TOKENS tokens (EOS ignored) so token counts are identical.
  * tokens/sec  = total generated tokens / total wall time over all trials.
  * latency     = wall time until the request's result is available. For vLLM offline batching a batch of B
                  requests is submitted at once and returns together, so per-request latency = batch wall time.
                  p50 / p95 are taken over trials (HF: over individual requests).
"""
import argparse, json, statistics, time

import config as C
from common import gpu_name_and_vram, save

PROMPTS = [C.MY_PROMPT] + [
    "Summarise the plot of Romeo and Juliet in three sentences.",
    "Write a short product description for a reusable water bottle.",
    "What are the main differences between TCP and UDP?",
    "Give me three tips for preparing for a data science interview.",
    "Explain gradient descent to a high-school student.",
    "Write a haiku-style description of a rainy Monday morning (do not quote existing poems).",
    "List four ways to reduce GPU memory use when training a transformer.",
    "Why is the sky blue? Answer briefly.",
] * 5   # 41 prompts: enough variety to rotate through


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(lat_ms, tokens, wall_s):
    return dict(tokens_per_s=tokens / wall_s, p50_ms=pct(lat_ms, .5), p95_ms=pct(lat_ms, .95),
                mean_ms=statistics.mean(lat_ms), n_latency_samples=len(lat_ms))


def run_hf(args):
    import torch
    from transformers import pipeline
    pipe = pipeline("text-generation", model=args.model, torch_dtype=torch.bfloat16, device=0)
    kw = dict(max_new_tokens=C.SERVE_MAX_NEW_TOKENS, min_new_tokens=C.SERVE_MAX_NEW_TOKENS, do_sample=False,
              return_full_text=False)
    for p in PROMPTS[:2]:
        pipe(p, **kw)                                   # warm-up
    lat, t_all = [], 0.0
    for i in range(max(C.SERVE_TRIALS * 2, 20)):
        p = PROMPTS[i % len(PROMPTS)]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        pipe(p, **kw)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        lat.append(dt * 1000); t_all += dt
    return {"hf_pipeline_bs1": dict(batch=1, **summarize(lat, C.SERVE_MAX_NEW_TOKENS * len(lat), t_all))}


def run_vllm(args):
    from vllm import LLM, SamplingParams
    llm = LLM(args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_util, max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=C.SERVE_MAX_NEW_TOKENS, ignore_eos=True)
    llm.generate(PROMPTS[:4], sp, use_tqdm=False)       # warm-up
    out = {}
    for b in C.SERVE_BATCH_SIZES:
        lat, toks, t_all = [], 0, 0.0
        for t in range(C.SERVE_TRIALS):
            batch = [PROMPTS[(t * b + j) % len(PROMPTS)] for j in range(b)]
            t0 = time.perf_counter()
            res = llm.generate(batch, sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            toks += sum(len(r.outputs[0].token_ids) for r in res)
            t_all += dt; lat.append(dt * 1000)
        out[f"vllm_bs{b}"] = dict(batch=b, **summarize(lat, toks, t_all))
    # your own prompt, for the "experiment with your own prompt" requirement
    r = llm.generate([C.MY_PROMPT], SamplingParams(temperature=0.7, max_tokens=96), use_tqdm=False)[0]
    out["own_prompt_demo"] = dict(prompt=C.MY_PROMPT, completion=r.outputs[0].text)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "vllm"], required=True)
    ap.add_argument("--model", default=C.SERVE_MODEL)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    a = ap.parse_args()
    name, vram = gpu_name_and_vram() if a.engine == "hf" else ("see nvidia-smi", 0)
    res = run_hf(a) if a.engine == "hf" else run_vllm(a)
    res["_meta"] = dict(model=a.model, engine=a.engine, max_new_tokens=C.SERVE_MAX_NEW_TOKENS, dtype="bfloat16",
                        trials=C.SERVE_TRIALS)
    save(f"serve_{a.engine}.json", res)
    print(json.dumps(res, indent=2))
