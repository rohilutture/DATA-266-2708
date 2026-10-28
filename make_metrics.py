"""Generate METRICS.md (HW4.5 summary + Tables HW4.1 / HW4.5.1) and GPU_HOURS.md from results/<gpu>/*.json."""
import json
from pathlib import Path
import config as C

ROOT = Path(__file__).resolve().parent
TAGS = sorted(p.name for p in (ROOT / "results").iterdir() if p.is_dir())
SING = ["baseline", "amp", "ckpt", "optim8"]
COMB = ["amp+optim8", "amp+ckpt", "amp+ckpt+optim8"]
ACC_GUARD = 0.01   # a "best" config may not lose more than 1 accuracy point vs baseline


def J(t, n):
    p = ROOT / "results" / t / n
    return json.loads(p.read_text()) if p.exists() else None


def at(sw, bs):
    return next((r for r in sw["rows"] if r["bs"] == bs and not r.get("oom")), None) if sw else None


def best_tp(sw):
    ok = [r for r in sw["rows"] if not r.get("oom")]
    return max(ok, key=lambda r: r["throughput"]) if ok else None


def plateau(sw, tol=0.05):
    ok = [r for r in sw["rows"] if not r.get("oom")]
    for a, b in zip(ok, ok[1:]):
        if b["throughput"] < a["throughput"] * (1 + tol):
            return a["bs"]
    return ok[-1]["bs"] if ok else None


f2 = lambda x, n=2: "n/a" if x is None else f"{x:.{n}f}"


def card_section(t):
    L = [f"## Card: `{t}`", ""]
    a = J(t, "hw4a_baseline.json"); ds = J(t, "data_split.json")
    sw = {n: J(t, f"sweep_{n}.json") for n in SING + COMB}
    fu = {n: J(t, f"full_{n}.json") for n in SING + COMB}
    acc = J(t, "accum.json")
    B = sw["baseline"]
    if ds:
        L += [f"**Dataset:** {ds['dataset']} — rows **{ds['row_range']}** ({ds['n_rows']} rows, SLICE={ds['slice']}, SEED={ds['seed']}); "
              f"split {ds['split_sizes']}. Same rows/split/seed as HW4.a.", ""]
    if a:
        L += [f"### HW4.a baseline (HW4 settings, batch {C.HW4_BATCH}, fp32, AdamW, lr {C.LR}, {C.EPOCHS} epochs)",
              f"- peak GPU memory **{a['peak_mem_gb']:.2f} GiB** allocated ({a['peak_reserved_gb']:.2f} GiB reserved); wall-clock **{a['wall_s']:.1f} s** "
              f"({a['train_s']:.1f} s training only); {a['throughput']:.0f} samples/s; best epoch {a['best_epoch']}; test acc (best-val weights) {a['test_acc_best']:.4f}.", ""]

    # ---------------- Part A
    L += ["### Part A — batch scaling to the memory wall (baseline, fp32)", "",
          "| batch | samples/s | peak mem (GiB) | est. time/epoch (s) |", "|---|---|---|---|"]
    if B:
        for r in B["rows"]:
            L.append(f"| {r['bs']} | OOM | OOM | OOM |" if r.get("oom") else
                     f"| {r['bs']} | {r['throughput']:.0f} | {r['peak_mem_gb']:.2f} | {r['est_epoch_s']:.1f} |")
        L += ["", f"Largest batch that runs: **{B['max_batch']}**" + (" (hit the sweep cap, no OOM reached)" if B["hit_cap"] else "") +
              f". Throughput stops improving (<5% gain per doubling) beyond batch **{plateau(B)}**. Time/epoch = n_train / throughput "
              f"(probe of {C.PROBE_OPT_STEPS} timed optimiser steps after 3 warm-up steps; the full-run epoch time is in the HW4.a report).", ""]
    L += ["Figures: `figures/fig_partA_throughput.png`, `figures/fig_partA_memory.png`.", ""]

    # ---------------- Part B
    L += ["### Table HW4.5.1 — memory techniques, each applied independently", "",
          f"Peak memory and 'tput @32' are at the HW4 batch ({C.HW4_BATCH}); 'max batch' and 'tput @max' come from the OOM sweep. "
          f"Accuracy = test accuracy after the full {C.EPOCHS}-epoch run (test set = {ds['split_sizes']['test'] if ds else '?'} rows, so ±~1.5 pt is noise).", "",
          "| Technique | Max batch | Throughput @max (samp/s) | Throughput @32 (samp/s) | Peak mem @32 (GB) | Eval accuracy |", "|---|---|---|---|---|---|"]
    names = {"baseline": "Baseline (HW4 settings)", "amp": "Mixed precision (bf16)", "ckpt": "Gradient checkpointing", "optim8": "8-bit optimiser (AdamW8bit)"}
    for n in SING:
        s = sw[n]
        if not s: continue
        m, r32, f = best_tp(s), at(s, C.HW4_BATCH), fu[n]
        L.append(f"| {names[n]} | {s['max_batch']}{'+' if s['hit_cap'] else ''} | {f2(m['throughput'], 0)} | {f2(r32 and r32['throughput'], 0)} | "
                 f"{f2(r32 and r32['peak_mem_gb'])} | {f2(f and f['test_acc'], 4)} |")
    if acc and B:
        pa = acc["probe_accum"]; fa = acc["full_accum"]
        L.append(f"| Gradient accumulation ({acc['physical']}×{acc['accum_steps']} = {acc['effective_batch']} effective) | {B['max_batch']} physical / {acc['effective_batch']} effective | "
                 f"{f2(pa.get('throughput'), 0)} | n/a | {f2(pa.get('peak_mem_gb'))} | {f2(fa.get('test_acc'), 4)} |")
    # best combination
    cands = [n for n in SING + COMB if sw.get(n) and fu.get(n)]
    base_acc = fu["baseline"]["test_acc"] if fu.get("baseline") else 0
    ok = [n for n in cands if fu[n]["test_acc"] >= base_acc - ACC_GUARD]
    best = {}
    if ok:
        best["throughput"] = max(ok, key=lambda n: best_tp(sw[n])["throughput"])
        best["memory"] = min((n for n in ok if at(sw[n], C.HW4_BATCH)), key=lambda n: at(sw[n], C.HW4_BATCH)["peak_mem_gb"])
        best["wall"] = min(ok, key=lambda n: fu[n]["wall_s"])
        n = best["throughput"]; s = sw[n]; m = best_tp(s); r32 = at(s, C.HW4_BATCH)
        L.append(f"| **Best combination** (`{n}`) | {s['max_batch']}{'+' if s['hit_cap'] else ''} | {f2(m['throughput'], 0)} | {f2(r32 and r32['throughput'], 0)} | "
                 f"{f2(r32 and r32['peak_mem_gb'])} | {f2(fu[n]['test_acc'], 4)} |")
    L += ["", f"Best-combination rule: among the single techniques and combinations {COMB}, highest throughput at its own max batch whose accuracy is within "
          f"{ACC_GUARD*100:.0f} point of baseline. Combinations tested: " + ", ".join(f"`{n}`" for n in COMB) + ".", ""]
    if acc:
        d, a_ = acc["probe_direct"], acc["probe_accum"]
        L += [f"**Gradient accumulation detail:** effective batch {acc['effective_batch']} from physical batch {acc['physical']}. Wall-clock for {acc['n_opt_steps']} optimiser steps: "
              f"accumulation {f2(a_.get('time_n_opt_steps_s'))} s vs direct batch {acc['effective_batch']}: "
              + ("OOM" if d.get("oom") else f"{d['time_n_opt_steps_s']:.2f} s") +
              f". Accuracy: accumulation {f2(acc['full_accum'].get('test_acc'), 4)} vs direct {f2(acc['full_direct'].get('test_acc'), 4) if not acc['full_direct'].get('oom') else 'OOM'}.", ""]

    # ranking: memory saved per unit of throughput lost
    L += ["#### Ranking: memory saved per % throughput lost", "",
          "| Rank | Technique | Memory saved (GiB) | Throughput change | Score (GiB per % lost) |", "|---|---|---|---|---|"]
    rank = []
    b32 = at(B, C.HW4_BATCH) if B else None
    if b32:
        for n in ("amp", "ckpt", "optim8"):
            r = at(sw[n], C.HW4_BATCH) if sw[n] else None
            if r:
                saved = b32["peak_mem_gb"] - r["peak_mem_gb"]; lost = (b32["throughput"] - r["throughput"]) / b32["throughput"] * 100
                rank.append((n, saved, lost, f"vs baseline @{C.HW4_BATCH}"))
    if acc and B:
        d, a_ = acc["probe_direct"], acc["probe_accum"]
        if not d.get("oom") and not a_.get("oom"):
            rank.append(("accum", d["peak_mem_gb"] - a_["peak_mem_gb"], (d["throughput"] - a_["throughput"]) / d["throughput"] * 100, f"vs direct batch {acc['effective_batch']}"))
    rank.sort(key=lambda x: -x[1] / max(x[2], 0.1))
    for i, (n, sv, lo, note) in enumerate(rank, 1):
        chg = f"{-lo:+.1f}% ({'faster' if lo < 0 else 'slower'})"
        L.append(f"| {i} | {n} | {sv:.2f} | {chg} {note} | {'free (no loss)' if lo <= 0 else f'{sv / lo:.3f}'} |")
    L += ["", "Apply first: **TODO choose from ranking above (usually the top row).** Skip on a 32 GB card: **TODO choose from ranking above** (a technique that costs throughput but whose memory is not needed because the model already fits comfortably).", ""]

    # ---------------- Part C
    L += ["### Part C", "", "Part C skipped (optional)", ""]

    # ---------------- Part D
    hf, vl, kv = J(t, "serve_hf.json"), J(t, "serve_vllm.json"), J(t, "kv_cache.json")
    L += ["### Table HW4.1 / Part D — serving comparison", ""]
    if hf and vl:
        m = hf["_meta"]
        L += [f"Model `{m['model']}`, bf16, {m['max_new_tokens']} new tokens per request (EOS ignored), {m['trials']} trials per vLLM batch size. "
              "Latency = wall time until the request's result is available (vLLM offline batch returns together → batch wall time); p50/p95 over trials (HF: over requests).", "",
              "| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) |", "|---|---|---|---|"]
        rows = [("HF pipeline, batch 1", hf["hf_pipeline_bs1"])] + [(f"vLLM, batch {b}", vl[f"vllm_bs{b}"]) for b in C.SERVE_BATCH_SIZES]
        for n, r in rows:
            L.append(f"| {n} | {r['tokens_per_s']:.0f} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} |")
        base = hf["hf_pipeline_bs1"]; prev = None; cross = None
        for b in C.SERVE_BATCH_SIZES:
            r = vl[f"vllm_bs{b}"]
            if prev and r["p50_ms"] > base["p50_ms"] and r["tokens_per_s"] > prev["tokens_per_s"] and cross is None:
                cross = b
            prev = r
        L += ["", f"Computed fact for the 3–4 sentence answer: per-request p50 latency first exceeds the HF batch-1 baseline ({base['p50_ms']:.0f} ms) "
              f"while total throughput is still improving at vLLM batch **{cross if cross else 'none of the tested sizes'}**. "
              "**TODO write your 3–4 sentence answer (interactive apps care about latency; batch/offline jobs care about throughput).**", ""]
        if "own_prompt_demo" in vl:
            L += [f"Own-prompt demo: \"{vl['own_prompt_demo']['prompt']}\" → {vl['own_prompt_demo']['completion'].strip()[:300]}", ""]
    else:
        L += ["_serving results missing — run serve_bench.py --engine hf / vllm_", ""]
    L += ["HW4 vs HW4.5 comparison: HW4.a serving was regenerated on this card with this same script, so the two runs coincide. "
          "If you also have an earlier HW4.a table, compare it here and attribute any difference to hardware, precision (bf16 vs the earlier dtype) or a measurement flaw (e.g. missing warm-up, latency measured per batch vs per request). **TODO**", ""]
    if kv:
        L += ["#### KV cache vs context length (batch 8)", "", f"Weights: {kv['hf']['weights_gb']:.2f} GiB. Analytic = 2·layers·kv_heads·head_dim·2 bytes·ctx·batch; measured = size of the cache tensors after a real prefill.", "",
              "| context | analytic KV (GiB) | measured KV (GiB) | peak total (GiB) |", "|---|---|---|---|"]
        for r in kv["hf"]["rows"]:
            L.append(f"| {r['ctx']} | {f2(r.get('analytic_kv_gb'), 3)} | {'OOM' if r.get('oom') else f2(r.get('measured_kv_gb'), 3)} | {f2(r.get('peak_total_gb'))} |")
        c, v = kv["vllm_capacity"], kv["vllm_validation"]
        L += ["", f"vLLM KV capacity at gpu_memory_utilization 0.90: {c.get('kv_capacity_tokens')} tokens → **longest context at batch 8 = {c.get('longest_ctx_bs8')} tokens** "
              f"(model max positions {c.get('model_max_pos')}); validation run serving 8 × that length: {v}.", "", f"Figure: `figures/fig_kv_{t}.png`.", ""]

    # ---------------- Part E
    a_tp = fu["baseline"]["throughput"] if fu.get("baseline") else (a["throughput"] if a else None)
    L += ["### Part E — summary", ""]
    if a_tp and best:
        ntr = (ds or {"split_sizes": {"train": 0}})["split_sizes"]["train"]
        bn = best["throughput"]; btp = best_tp(sw[bn])["throughput"]
        L += [f"- **GPU-hours, baseline:** {C.EPOCHS} epochs × {ntr} samples ÷ {a_tp:.0f} samples/s = {C.EPOCHS*ntr/a_tp:.0f} s = **{C.EPOCHS*ntr/a_tp/3600:.4f} GPU-hours**.",
              f"- **GPU-hours, best config (`{bn}`):** {C.EPOCHS} × {ntr} ÷ {btp:.0f} = {C.EPOCHS*ntr/btp:.0f} s = **{C.EPOCHS*ntr/btp/3600:.4f} GPU-hours**.", "",
              "## HW4.5 SUMMARY (best configurations)", "",
              f"- **Best throughput:** `{best['throughput']}` — {btp:.0f} samples/s at batch {best_tp(sw[bn])['bs']}.",
              f"- **Lowest peak memory @ batch {C.HW4_BATCH}:** `{best['memory']}` — {at(sw[best['memory']], C.HW4_BATCH)['peak_mem_gb']:.2f} GiB (baseline {b32['peak_mem_gb']:.2f} GiB).",
              f"- **Lowest wall-clock for the full fine-tune:** `{best['wall']}` — {fu[best['wall']]['wall_s']:.1f} s (baseline {fu['baseline']['wall_s']:.1f} s).",
              "- (HW4.5 summary only — not an HW5 baseline.)", "",
              "**Single configuration I would use for a full fine-tune: TODO pick from the tables, with a one-sentence justification.**", ""]
    return L


def gpu_hours():
    L = ["# GPU-hour record", "", "## Estimated from measured throughput", "", "GPU-hours = epochs × n_train ÷ (samples/s) ÷ 3600.", ""]
    for t in TAGS:
        ds, a, fu = J(t, "data_split.json"), J(t, "hw4a_baseline.json"), J(t, "full_baseline.json")
        tp = (fu or a or {}).get("throughput")
        if not (ds and tp): continue
        ntr = ds["split_sizes"]["train"]
        full = 45615   # full tweet_eval sentiment train split, for scale
        L += [f"### {t}", f"- {C.EPOCHS} × {ntr} ÷ {tp:.0f} = {C.EPOCHS*ntr/tp:.0f} s = **{C.EPOCHS*ntr/tp/3600:.4f} GPU-h** for the HW4.a fine-tune.",
              f"- For scale: the full 45,615-row train split would be {C.EPOCHS*full/tp/3600:.3f} GPU-h at the same throughput.", ""]
    L += ["## Actual lab session usage (fill in)", "", "| date | GPU | start | end | hours | what ran |", "|---|---|---|---|---|---|", "| | | | | | |", "",
          "Tip: `RUN_LOG.txt` has a timestamp for every stage; sum them for the real total."]
    (ROOT / "GPU_HOURS.md").write_text("\n".join(L))


if __name__ == "__main__":
    L = ["# METRICS.md — HW4.5 summary", "",
         "Generated by `make_metrics.py` from `results/`. Lines marked **TODO** need your own judgement/wording.", "",
         "Part C skipped (optional)", "", "Reservation, UUID captures, figures and logs: see `RESERVATION.md`, `uuid_capture_*.txt`, `figures/`, `RUN_LOG.txt`.", ""]
    for t in TAGS:
        L += card_section(t)
    if len(TAGS) == 1:
        L += ["_Only one card was measured (instructor: one GPU is enough). Part A's cross-card max-batch ratio vs VRAM ratio and Part D 'both cards' are therefore answered for this single card; see the discussion-board answer if one was given._"]
    (ROOT / "METRICS.md").write_text("\n".join(L))
    gpu_hours()
    print("wrote METRICS.md and GPU_HOURS.md")
