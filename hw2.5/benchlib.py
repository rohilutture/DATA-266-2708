"""
benchlib.py -- shared utilities for the HW2.5 GPU benchmark suite.

Every measurement produced by the part_*.py scripts is stamped with the GPU
UUID, so each CSV row and every RUN_LOG.txt line is traceable to one card --
which is exactly what Table HW2.5.1 requires ("Every cell must be traceable to
a UUID-labelled run").

Install once on the workstation:
    pip install torch numpy matplotlib nvidia-ml-py
"""

import csv
import os
import subprocess
from datetime import datetime, timezone

import torch


# ---------------------------------------------------------------------------
# Theoretical DENSE tensor-core peaks (TFLOPS).
#
# READ THIS BEFORE TRUSTING THE "% of peak" COLUMN IN PART B:
#   * These are DENSE numbers. NVIDIA's marketing/whitepaper headline figures
#     are usually 2:4 STRUCTURED SPARSE, which is exactly 2x these. A plain
#     torch.matmul is dense, so it can never exceed the dense peak. Compare
#     against dense.
#   * The 16-bit paths depend on ACCUMULATE precision:
#       - TF32 always accumulates in FP32.
#       - BF16 tensor cores only accumulate in FP32.
#       - FP16 can accumulate in FP16 (2x throughput) or FP32 (1x). cuBLAS /
#         PyTorch default to FP32 accumulate for stability, so FP16 and BF16
#         share the dense peak below. If you deliberately force FP16 accumulate
#         you may double the FP16 entry -- say so in METRICS.md if you do.
#
# Sources for these reference values: TechPowerUp GPU database and the NVIDIA
# Ada / Blackwell architecture whitepapers. VERIFY them against the exact
# vendor document you cite in Part A before reporting percentages -- do not
# take them on my word.
# ---------------------------------------------------------------------------
THEORETICAL_PEAK_TFLOPS = {
    "RTX 4090": {          # Ada Lovelace, AD102, 4th-gen tensor cores, CC 8.9
        "FP32": 82.6,      # CUDA (shader) cores, not tensor cores
        "TF32": 82.6,      # tensor, FP32 accumulate
        "FP16": 165.2,     # tensor, FP32 accumulate (FP16 accumulate = 330.3)
        "BF16": 165.2,     # tensor, FP32 accumulate
        "FP8": 330.3,      # tensor, FP32 accumulate (FP16 accumulate = 660.6)
        "bandwidth_GBs": 1008.0,
        "vram_GiB": 24,
    },
    "RTX 5090": {          # Blackwell, GB202, 5th-gen tensor cores
        "FP32": 104.8,     # CUDA (shader) cores
        "TF32": 104.8,     # tensor, FP32 accumulate
        "FP16": 209.5,     # tensor, FP32 accumulate (FP16 accumulate = 419.0)
        "BF16": 209.5,     # tensor, FP32 accumulate
        "FP8": 419.0,      # tensor, FP32 accumulate (FP16 accumulate = 838.0)
        "FP4": 838.0,      # dense; Blackwell-only, tooling-dependent
        "bandwidth_GBs": 1792.0,
        "vram_GiB": 32,
    },
}


def match_peak_key(device_name):
    """Map a torch device name to a key in THEORETICAL_PEAK_TFLOPS."""
    n = (device_name or "").upper()
    if "5090" in n:
        return "RTX 5090"
    if "4090" in n:
        return "RTX 4090"
    return None


# ---------------------------------------------------------------------------
# GPU provenance
# ---------------------------------------------------------------------------
def get_gpu_info(index=0):
    """Return a dict of provenance for the active GPU (UUID, driver, etc.)."""
    info = {
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "capability": ".".join(map(str, torch.cuda.get_device_capability(index))),
        "uuid": "UNKNOWN",
        "driver": "UNKNOWN",
        "power_limit_W": None,
        "vram_total_GiB": round(
            torch.cuda.get_device_properties(index).total_memory / 1024**3, 2
        ),
    }

    def _s(x):
        return x.decode() if isinstance(x, bytes) else x

    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        info["uuid"] = _s(pynvml.nvmlDeviceGetUUID(h))
        info["name"] = _s(pynvml.nvmlDeviceGetName(h))
        info["driver"] = _s(pynvml.nvmlSystemGetDriverVersion())
        info["power_limit_W"] = round(
            pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0, 1
        )
        pynvml.nvmlShutdown()
        return info
    except Exception:
        pass  # fall back to nvidia-smi

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,power.limit",
                "--format=csv,noheader,nounits",
                "-i",
                str(index),
            ],
            text=True,
        ).strip()
        uuid, name, driver, plimit = [c.strip() for c in out.split(",")]
        info.update(
            {"uuid": uuid, "name": name, "driver": driver, "power_limit_W": float(plimit)}
        )
    except Exception as exc:  # keep going -- UUID just stays UNKNOWN
        print(f"[benchlib] could not read nvidia-smi provenance: {exc}")
    return info


# ---------------------------------------------------------------------------
# Timing -- CUDA events, never wall clock. Wall-clock timing of async CUDA
# kernels measures launch overhead, not execution. Warmup absorbs one-time
# kernel compilation / autotuning on the first call.
# ---------------------------------------------------------------------------
def cuda_time(fn, warmup=15, iters=50):
    """Mean seconds per call of fn(). Returns (seconds, iters)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    total_s = start.elapsed_time(end) / 1000.0  # elapsed_time is in ms
    return total_s / iters, iters


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_run_log(part, uuid, note, path="RUN_LOG.txt"):
    """One tab-separated line per run: timestamp, part, UUID, free-text note."""
    with open(path, "a") as f:
        f.write(f"{utc_now()}\t{part}\tUUID={uuid}\t{note}\n")


def write_csv(path, rows, fieldnames):
    """Append rows (list of dicts) to a CSV, writing the header if new."""
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def require_cuda():
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible to PyTorch. Check drivers / CUDA_VISIBLE_DEVICES.")
