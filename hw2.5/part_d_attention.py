"""
Part D -- The cost of attention.

Naive scaled dot-product attention MATERIALIZES the full L x L score matrix:
    scores = (Q @ K^T) * scale      # shape (B, H, L, L)  <-- the quadratic term
    attn   = softmax(scores)        # another (B, H, L, L)
    out    = attn @ V
Peak memory therefore grows ~ B*H*L^2, which is what makes long context expensive.

The fused path (F.scaled_dot_product_attention with a Flash / mem-efficient
backend) never materializes the full L x L matrix: it tiles the sequence and
streams the softmax online, so memory grows ~ L instead of L^2.

FIXED CONFIG (state these in METRICS.md):
    batch_size = 1
    num_heads  = 8
    head_dim   = 64
    dtype      = fp16   (Flash attention requires fp16/bf16)
Change them with flags if you like, but report whatever you actually ran.

Run:
    python part_d_attention.py
Outputs: part_d_<UUID>.csv, part_d_<UUID>.png, RUN_LOG.txt line.
"""

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import benchlib as bl

SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384]


def naive_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B,H,L,L)
    attn = torch.softmax(scores, dim=-1)                    # (B,H,L,L)
    return torch.matmul(attn, v)


def fused_attention(q, k, v):
    # Prefer Flash, fall back to the memory-efficient kernel.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v)


def measure(fn, L, B, H, D, dtype, iters, warmup):
    """Return (sec_per_forward, peak_bytes, status)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        q = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        for _ in range(warmup):
            fn(q, k, v)
        torch.cuda.synchronize()
        sec, _ = bl.cuda_time(lambda: fn(q, k, v), warmup=0, iters=iters)
        peak = torch.cuda.max_memory_allocated()
        del q, k, v
        return sec, peak, "ok"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, "OOM"
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            return None, None, "OOM"
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    bl.require_cuda()
    info = bl.get_gpu_info()
    uuid = info["uuid"]
    B, H, D, dtype = args.batch, args.heads, args.head_dim, torch.float16
    print(f"GPU: {info['name']}  UUID={uuid}")
    print(f"config: batch={B} heads={H} head_dim={D} dtype=fp16\n")

    rows = []
    for label, fn in [("naive", naive_attention), ("fused", fused_attention)]:
        print(f"[{label}]")
        for L in SEQ_LENS:
            sec, peak, status = measure(fn, L, B, H, D, dtype, args.iters, args.warmup)
            rows.append({
                "uuid": uuid, "impl": label, "seq_len": L, "status": status,
                "sec_per_forward": f"{sec:.6e}" if sec else "",
                "latency_ms": round(sec * 1e3, 3) if sec else "",
                "peak_mem_MiB": round(peak / 1024**2, 1) if peak else "",
                "peak_mem_bytes": peak if peak else "",
            })
            if status == "ok":
                print(f"  L={L:6d}: {sec*1e3:8.2f} ms  peak {peak/1024**2:8.1f} MiB")
            else:
                print(f"  L={L:6d}: OOM")

    csv_path = f"part_d_{uuid}.csv"
    bl.write_csv(csv_path, rows, list(rows[0].keys()))

    # --- OOM boundary (from the coarse sweep) -------------------------------
    def boundary(impl):
        ok = [r["seq_len"] for r in rows if r["impl"] == impl and r["status"] == "ok"]
        oom = [r["seq_len"] for r in rows if r["impl"] == impl and r["status"] == "OOM"]
        return (max(ok) if ok else None), (min(oom) if oom else None)

    n_ok, n_oom = boundary("naive")
    f_ok, f_oom = boundary("fused")
    print(f"\nNaive OOM boundary: largest OK = {n_ok}, smallest OOM = {n_oom}")
    print(f"Fused OOM boundary: largest OK = {f_ok}, smallest OOM = {f_oom}")
    print("(These are the coarse tested lengths -- do NOT claim single-token "
          "resolution unless you searched at that resolution.)")

    # --- quadratic fit of naive peak memory vs L ----------------------------
    fit_L = np.array([r["seq_len"] for r in rows
                      if r["impl"] == "naive" and r["status"] == "ok"], dtype=float)
    fit_M = np.array([r["peak_mem_bytes"] for r in rows
                      if r["impl"] == "naive" and r["status"] == "ok"], dtype=float)
    coeffs = None
    if len(fit_L) >= 3:
        coeffs = np.polyfit(fit_L, fit_M, 2)  # a*L^2 + b*L + c
        a = coeffs[0]
        # Predicted quadratic coeff if the L x L score+prob matrices dominate:
        #   2 matrices * B*H * 2 bytes(fp16) = 4*B*H bytes per L^2
        predicted = 4 * B * H
        print(f"\nQuadratic fit  peak_mem(bytes) = {coeffs[0]:.3f} L^2 "
              f"+ {coeffs[1]:.1f} L + {coeffs[2]:.1f}")
        print(f"Measured L^2 coefficient a = {a:.4f} bytes/L^2  "
              f"(order-of-magnitude prediction ~{predicted} for 2 fp16 LxL matrices)")

    # --- speedup at each length where both ran ------------------------------
    print("\nFused speedup (naive_latency / fused_latency):")
    for L in SEQ_LENS:
        nrow = next((r for r in rows if r["impl"] == "naive" and r["seq_len"] == L
                     and r["status"] == "ok"), None)
        frow = next((r for r in rows if r["impl"] == "fused" and r["seq_len"] == L
                     and r["status"] == "ok"), None)
        if nrow and frow:
            sp = float(nrow["sec_per_forward"]) / float(frow["sec_per_forward"])
            print(f"  L={L:6d}: {sp:5.2f}x")

    # --- plot: peak memory vs L, with quadratic fit -------------------------
    plt.figure(figsize=(8, 5))
    for impl, mark in [("naive", "o"), ("fused", "s")]:
        xs = [r["seq_len"] for r in rows if r["impl"] == impl and r["status"] == "ok"]
        ys = [r["peak_mem_MiB"] for r in rows if r["impl"] == impl and r["status"] == "ok"]
        plt.plot(xs, ys, marker=mark, label=f"{impl} (measured)")
    if coeffs is not None:
        xx = np.linspace(fit_L.min(), fit_L.max(), 200)
        yy = np.polyval(coeffs, xx) / 1024**2
        plt.plot(xx, yy, "--", alpha=0.7, label="naive quadratic fit")
    plt.xlabel("Sequence length L")
    plt.ylabel("Peak memory (MiB)")
    plt.title(f"Part D: attention peak memory -- {info['name']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    png_path = f"part_d_{uuid}.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)

    bl.append_run_log("PartD", uuid,
                      f"attention B={B} H={H} D={D} fp16; naive OOM~{n_oom}, "
                      f"fused OOM~{f_oom}; -> {csv_path}, {png_path}")
    print(f"\nWrote {csv_path} and {png_path}. RUN_LOG.txt updated.")
    print("Three-sentence prompt for the write-up: the fused kernel never builds "
          "the full L x L\nscore or probability matrix -- it tiles Q/K/V and keeps a "
          "running softmax (max + sum)\nin SRAM, so it avoids the O(L^2) global-memory "
          "reads/writes and the O(L^2) allocation.")


if __name__ == "__main__":
    main()
