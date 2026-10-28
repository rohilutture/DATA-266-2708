"""Part D: KV-cache memory vs context length at batch 8, and the longest context the card can serve at batch 8.

  python kv_cache.py            # run in the vLLM virtualenv

1. Analytic KV bytes from the model config, and MEASURED KV bytes from a real HF prefill (cache tensor sizes),
   for context lengths 128 .. 32768 at batch 8 (stops at the first OOM).
2. vLLM capacity: start vLLM, read how many KV-cache tokens it could allocate, longest ctx at batch 8 =
   capacity // 8 (capped by the model's own max positions); then VALIDATE by actually serving 8 sequences of that length.
"""
import json, os, subprocess, sys
import config as C
from common import gpu_name_and_vram, save

BATCH = 8
CTXS = [2 ** k for k in range(7, 16)]   # 128 .. 32768


def kv_bytes_analytic(cfg, ctx, batch, dtype_bytes=2):
    heads = cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", heads)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
    return 2 * cfg.num_hidden_layers * kv_heads * head_dim * dtype_bytes * ctx * batch


def cache_bytes(pkv):
    tot = 0
    if hasattr(pkv, "layers"):
        for l in pkv.layers:
            tot += l.keys.numel() * l.keys.element_size() + l.values.numel() * l.values.element_size()
    elif hasattr(pkv, "key_cache"):
        for k, v in zip(pkv.key_cache, pkv.value_cache):
            tot += k.numel() * k.element_size() + v.numel() * v.element_size()
    else:
        for k, v in pkv:
            tot += k.numel() * k.element_size() + v.numel() * v.element_size()
    return tot


def hf_measure():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(C.SERVE_MODEL)
    max_pos = getattr(cfg, "max_position_embeddings", 32768)
    model = AutoModelForCausalLM.from_pretrained(C.SERVE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    weights_gb = torch.cuda.memory_allocated() / 2**30
    body = getattr(model, "model", None)
    rows = []
    for ctx in CTXS:
        row = dict(ctx=ctx, batch=BATCH, analytic_kv_gb=kv_bytes_analytic(cfg, ctx, BATCH) / 2**30)
        if ctx > max_pos:
            row["note"] = "exceeds model max_position_embeddings"; rows.append(row); break
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(100, 5000, (BATCH, ctx), device="cuda")
            with torch.no_grad():
                out = body(input_ids=ids, use_cache=True) if body is not None else model(input_ids=ids, use_cache=True, logits_to_keep=1)
            row["measured_kv_gb"] = cache_bytes(out.past_key_values) / 2**30
            row["peak_total_gb"] = torch.cuda.max_memory_allocated() / 2**30
            del out, ids
        except torch.cuda.OutOfMemoryError:
            row["oom"] = True; rows.append(row); break
        rows.append(row)
        print(row, flush=True)
    return dict(model=C.SERVE_MODEL, weights_gb=weights_gb, max_position_embeddings=max_pos, rows=rows)


def vllm_stage(stage, ctx=None):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # lets us read the allocated KV blocks in-process
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig
    max_pos = getattr(AutoConfig.from_pretrained(C.SERVE_MODEL), "max_position_embeddings", 32768)
    if stage == "capacity":
        llm = LLM(C.SERVE_MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=2048, max_num_seqs=BATCH)
        eng = llm.llm_engine
        cc = getattr(eng, "cache_config", None) or getattr(getattr(eng, "vllm_config", None), "cache_config", None)
        blocks = getattr(cc, "num_gpu_blocks", None)
        bs = getattr(cc, "block_size", 16)
        cap = blocks * bs if blocks else None
        return dict(kv_capacity_tokens=cap, num_gpu_blocks=blocks, block_size=bs,
                    longest_ctx_bs8=min(cap // BATCH, max_pos) if cap else None, model_max_pos=max_pos)
    L = ctx
    llm = LLM(C.SERVE_MODEL, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=L, max_num_seqs=BATCH)
    prompts = [{"prompt_token_ids": [100 + (i * 7 + j) % 4000 for j in range(L - 32)]} for i in range(BATCH)]
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=16, ignore_eos=True), use_tqdm=False)
    return dict(validated_ctx=L, ok=all(len(o.outputs[0].token_ids) == 16 for o in outs))


def run_child(stage, ctx=None):
    p = subprocess.run([sys.executable, __file__, "_vllm", stage, str(ctx or 0)], capture_output=True, text=True)
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("RESULT:"):
            return json.loads(line[7:])
    return dict(error=(p.stderr or p.stdout)[-800:])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_vllm":
        print("RESULT:" + json.dumps(vllm_stage(sys.argv[2], int(sys.argv[3]))))
        sys.exit(0)
    name, vram = gpu_name_and_vram()
    hf = hf_measure()
    cap = run_child("capacity")
    val = run_child("validate", cap.get("longest_ctx_bs8")) if cap.get("longest_ctx_bs8") else dict(
        note="could not read KV capacity from vLLM; use the 'GPU KV cache size: N tokens' log line and divide by 8")
    save("kv_cache.json", dict(gpu=name, vram_gb=vram, hf=hf, vllm_capacity=cap, vllm_validation=val))
    print(json.dumps(dict(capacity=cap, validation=val), indent=2))
