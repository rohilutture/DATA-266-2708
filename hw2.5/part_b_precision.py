"""
Part B -- Precision and achieved throughput.

Dense NxN matmul at N = 1024, 4096, 8192, 16384 in FP32, TF32, FP16, BF16.
Reports achieved TFLOPS and achieved TFLOPS as a % of the card's dense
theoretical peak for that precision, and plots TFLOPS vs N (one line/precision).

matmul FLOP count = 2 * N^3  (N^2 outputs, each a length-N dot product = N
multiplies + N-1 adds ~ 2N flops).

Run:
    python part_b_precision.py --iters 50 --warmup 20
Outputs:
    part_b_<UUID>.csv, part_b_<UUID>.png, and a line appended to RUN_LOG.txt
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import benchlib as bl

SIZES = [1024, 4096, 8192, 16384]
PRECISIONS = ["FP32", "TF32", "FP16", "BF16"]


def make_matmul(N, precision, device):
    """Return an op() closure and its tensors for one (N, precision)."""
    if precision in ("FP32", "TF32"):
        dtype = torch.float32
        # FP32 vs TF32 is the SAME float32 data -- only this flag differs.
        use_tf32 = precision == "TF32"
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
    elif precision == "FP16":
        dtype = torch.float16
    elif precision == "BF16":
        dtype = torch.bfloat16
    else:
        raise ValueError(precision)

    a = torch.randn(N, N, device=device, dtype=dtype)
    b = torch.randn(N, N, device=device, dtype=dtype)
    c = torch.empty(N, N, device=device, dtype=dtype)

    def op():
        torch.matmul(a, b, out=c)

    return op, (a, b, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    bl.require_cuda()
    info = bl.get_gpu_info()
    uuid = info["uuid"]
    peaks = bl.THEORETICAL_PEAK_TFLOPS.get(bl.match_peak_key(info["name"]), {})
    print(f"GPU: {info['name']}  UUID={uuid}  driver={info['driver']}")

    rows = []
    for prec in PRECISIONS:
        for N in SIZES:
            op, tensors = make_matmul(N, prec, "cuda")
            sec, iters = bl.cuda_time(op, warmup=args.warmup, iters=args.iters)
            tflops = (2.0 * N**3) / sec / 1e12
            peak = peaks.get(prec)
            pct = (100.0 * tflops / peak) if peak else float("nan")
            rows.append(
                {
                    "uuid": uuid,
                    "precision": prec,
                    "N": N,
                    "iters": iters,
                    "sec_per_matmul": f"{sec:.6e}",
                    "achieved_TFLOPS": round(tflops, 2),
                    "peak_TFLOPS_dense": peak if peak else "",
                    "pct_of_peak": round(pct, 1) if peak else "",
                }
            )
            print(f"  {prec:4s} N={N:5d}: {tflops:8.1f} TFLOPS  ({pct:5.1f}% of peak)")
            del op, tensors
            torch.cuda.empty_cache()

    csv_path = f"part_b_{uuid}.csv"
    bl.write_csv(csv_path, rows, list(rows[0].keys()))

    # --- plot: achieved TFLOPS vs N, one line per precision -----------------
    plt.figure(figsize=(8, 5))
    for prec in PRECISIONS:
        xs = [r["N"] for r in rows if r["precision"] == prec]
        ys = [r["achieved_TFLOPS"] for r in rows if r["precision"] == prec]
        plt.plot(xs, ys, marker="o", label=prec)
    plt.xscale("log", base=2)
    plt.xlabel("Matrix size N (log2)")
    plt.ylabel("Achieved TFLOPS")
    plt.title(f"Part B: dense matmul throughput -- {info['name']}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    png_path = f"part_b_{uuid}.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)

    bl.append_run_log(
        "PartB",
        uuid,
        f"matmul precision sweep; iters={args.iters} warmup={args.warmup}; "
        f"-> {csv_path}, {png_path}",
    )
    print(f"\nWrote {csv_path} and {png_path}. RUN_LOG.txt updated.")
    print(
        "For the write-up: state where each precision plateaus (large N, where the\n"
        "GEMM is compute-bound and tensor cores stay fed) and why small N never does\n"
        "(kernel launch + memory traffic dominate; too little arithmetic per byte)."
    )


if __name__ == "__main__":
    main()
