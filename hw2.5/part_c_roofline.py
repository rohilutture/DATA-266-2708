"""
Part C -- Bandwidth-bound vs compute-bound.

  * memory-bound op: large elementwise add  c = a + b
        bytes moved = 3 * numel * dtype_bytes   (read a, read b, write c)
        FLOPs       = numel                      (one add per element)
        effective bandwidth = bytes / time
  * compute-bound op: large square matmul
        FLOPs       = 2 * N^3
        bytes       = 3 * N^2 * dtype_bytes      (minimum traffic; ignores the
                      cache reuse that is exactly why matmul is compute-bound)

Arithmetic intensity (FLOP/byte) for each is compared against the card's
ridge point (peak compute / peak bandwidth) to place it on the roofline.

Run:
    python part_c_roofline.py --add-elems 268435456 --matmul-n 8192
Outputs: part_c_<UUID>.csv and a RUN_LOG.txt line.
"""

import argparse

import torch

import benchlib as bl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-elems", type=int, default=256 * 1024 * 1024,
                    help="elements per array for the memory-bound add (fp32)")
    ap.add_argument("--matmul-n", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    bl.require_cuda()
    info = bl.get_gpu_info()
    uuid = info["uuid"]
    peaks = bl.THEORETICAL_PEAK_TFLOPS.get(bl.match_peak_key(info["name"]), {})
    peak_bw = peaks.get("bandwidth_GBs")            # GB/s
    peak_fp32 = peaks.get("FP32")                   # TFLOPS (compare matmul in fp32)
    ridge = (peak_fp32 * 1e12) / (peak_bw * 1e9) if (peak_bw and peak_fp32) else None
    print(f"GPU: {info['name']}  UUID={uuid}")
    if ridge:
        print(f"  ridge point = {ridge:.2f} FLOP/byte "
              f"(peak {peak_fp32} TFLOPS / {peak_bw} GB/s)")

    rows = []

    # ---- memory-bound: elementwise add (fp32) ------------------------------
    n = args.add_elems
    dtype_bytes = 4
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    sec, iters = bl.cuda_time(lambda: torch.add(a, b, out=c),
                              warmup=args.warmup, iters=args.iters)
    bytes_moved = 3 * n * dtype_bytes
    gbs = bytes_moved / sec / 1e9
    ai_add = (n) / bytes_moved
    pct_bw = 100.0 * gbs / peak_bw if peak_bw else float("nan")
    rows.append({
        "uuid": uuid, "op": "elementwise_add_fp32", "size": n,
        "sec": f"{sec:.6e}", "iters": iters,
        "effective_GBs": round(gbs, 1),
        "pct_of_peak_BW": round(pct_bw, 1) if peak_bw else "",
        "arith_intensity_FLOP_per_byte": round(ai_add, 4),
        "achieved_TFLOPS": round(n / sec / 1e12, 4),
        "roofline_side": "memory-bound" if (ridge and ai_add < ridge) else "compute-bound",
    })
    print(f"  add : {gbs:7.1f} GB/s ({pct_bw:4.1f}% of BW), AI={ai_add:.4f} FLOP/byte")
    del a, b, c
    torch.cuda.empty_cache()

    # ---- compute-bound: square matmul (fp32) -------------------------------
    N = args.matmul_n
    a = torch.randn(N, N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, N, device="cuda", dtype=torch.float32)
    c = torch.empty(N, N, device="cuda", dtype=torch.float32)
    sec, iters = bl.cuda_time(lambda: torch.matmul(a, b, out=c),
                              warmup=args.warmup, iters=args.iters)
    flops = 2.0 * N**3
    bytes_mm = 3 * N * N * 4
    tflops = flops / sec / 1e12
    ai_mm = flops / bytes_mm
    rows.append({
        "uuid": uuid, "op": f"matmul_fp32_N{N}", "size": N,
        "sec": f"{sec:.6e}", "iters": iters,
        "effective_GBs": round(bytes_mm / sec / 1e9, 1),
        "pct_of_peak_BW": "",
        "arith_intensity_FLOP_per_byte": round(ai_mm, 2),
        "achieved_TFLOPS": round(tflops, 2),
        "roofline_side": "compute-bound" if (ridge and ai_mm > ridge) else "memory-bound",
    })
    print(f"  mm  : {tflops:7.1f} TFLOPS, AI={ai_mm:.1f} FLOP/byte "
          f"(min-traffic estimate)")
    del a, b, c
    torch.cuda.empty_cache()

    csv_path = f"part_c_{uuid}.csv"
    bl.write_csv(csv_path, rows, list(rows[0].keys()))
    bl.append_run_log("PartC", uuid,
                      f"roofline: add {n} elems, matmul N={N}; -> {csv_path}")
    print(f"\nWrote {csv_path}. RUN_LOG.txt updated.")
    print("Note: the matmul AI uses minimum DRAM traffic (3*N^2). Real traffic is\n"
          "lower per FLOP thanks to L2/shared-memory reuse -- which is *why* it sits\n"
          "on the compute side of the roofline. State that in your analysis.")


if __name__ == "__main__":
    main()
