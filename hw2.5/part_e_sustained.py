"""
Part E -- Sustained load and thermal behaviour.

A worker thread runs back-to-back matmuls for N minutes (default 20) while a
sampler thread records SM clock, memory clock, temperature, power draw and
utilisation every 5 s via NVML. The worker also logs its own throughput
(TFLOPS) in short windows so you can compare the first 30 s against the final
5 minutes and see throttling as a throughput drop, not just a clock drop.

Run (the real deliverable is 20 minutes):
    python part_e_sustained.py --minutes 20 --matmul-n 8192
Quick smoke test first:
    python part_e_sustained.py --minutes 1

Outputs:
    part_e_samples_<UUID>.csv   (5 s telemetry samples -- COMMIT THIS as the log)
    part_e_throughput_<UUID>.csv
    part_e_<UUID>.png
    RUN_LOG.txt line
"""

import argparse
import threading
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import benchlib as bl


def sampler_loop(stop_evt, samples, index, interval):
    """Poll NVML every `interval` seconds until stop_evt is set."""
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(index)
    t0 = time.time()
    while not stop_evt.is_set():
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            sm_clk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            mem_clk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
            samples.append({
                "elapsed_s": round(time.time() - t0, 2),
                "sm_clock_MHz": sm_clk, "mem_clock_MHz": mem_clk,
                "temp_C": temp, "power_W": round(power, 1), "util_pct": util,
            })
        except Exception as exc:
            samples.append({"elapsed_s": round(time.time() - t0, 2),
                            "sm_clock_MHz": "", "mem_clock_MHz": "",
                            "temp_C": "", "power_W": "", "util_pct": f"ERR:{exc}"})
        stop_evt.wait(interval)
    pynvml.nvmlShutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--matmul-n", type=int, default=8192)
    ap.add_argument("--interval", type=float, default=5.0, help="sampling period (s)")
    ap.add_argument("--chunk", type=int, default=50, help="matmuls between throughput samples")
    args = ap.parse_args()

    bl.require_cuda()
    info = bl.get_gpu_info()
    uuid = info["uuid"]
    N = args.matmul_n
    print(f"GPU: {info['name']}  UUID={uuid}")
    print(f"Sustaining fp16 matmul N={N} for {args.minutes} min, sampling every "
          f"{args.interval}s ...")

    # Persistent buffers so the worker never allocates inside the hot loop.
    a = torch.randn(N, N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, N, device="cuda", dtype=torch.float16)
    c = torch.empty(N, N, device="cuda", dtype=torch.float16)
    flops_per_chunk = 2.0 * N**3 * args.chunk

    samples = []
    throughput = []
    stop_evt = threading.Event()
    sampler = threading.Thread(target=sampler_loop,
                               args=(stop_evt, samples, info["index"], args.interval),
                               daemon=True)
    sampler.start()

    end_time = time.time() + args.minutes * 60.0
    t0 = time.time()
    # warmup so the first window isn't cold-start noise
    for _ in range(10):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()

    try:
        while time.time() < end_time:
            w0 = time.time()
            for _ in range(args.chunk):
                torch.matmul(a, b, out=c)
            torch.cuda.synchronize()            # bound the queue; time real work
            dt = time.time() - w0
            throughput.append({
                "elapsed_s": round(time.time() - t0, 2),
                "TFLOPS": round(flops_per_chunk / dt / 1e12, 2),
            })
    finally:
        stop_evt.set()
        sampler.join(timeout=args.interval + 2)

    # --- persist -------------------------------------------------------------
    for r in samples:
        r["uuid"] = uuid
    for r in throughput:
        r["uuid"] = uuid
    samp_path = f"part_e_samples_{uuid}.csv"
    thru_path = f"part_e_throughput_{uuid}.csv"
    if samples:
        bl.write_csv(samp_path, samples,
                     ["uuid", "elapsed_s", "sm_clock_MHz", "mem_clock_MHz",
                      "temp_C", "power_W", "util_pct"])
    bl.write_csv(thru_path, throughput, ["uuid", "elapsed_s", "TFLOPS"])

    # --- peak (first 30s) vs steady state (final 5 min) ----------------------
    total_s = args.minutes * 60.0
    first30 = [r["TFLOPS"] for r in throughput if r["elapsed_s"] <= 30]
    last5 = [r["TFLOPS"] for r in throughput if r["elapsed_s"] >= max(0, total_s - 300)]
    if first30 and last5:
        peak = sum(first30) / len(first30)
        steady = sum(last5) / len(last5)
        print(f"\nPeak (first 30s):        {peak:.1f} TFLOPS")
        print(f"Steady-state (last 5min): {steady:.1f} TFLOPS")
        print(f"Steady / peak:            {100*steady/peak:.1f}%")

    # --- throttle detection: first sample below 95% of the early clock ------
    if samples:
        clk = [(s["elapsed_s"], s["sm_clock_MHz"]) for s in samples
               if isinstance(s["sm_clock_MHz"], int)]
        if clk:
            early = max(c for _, c in clk[:3])
            throttled = next((t for t, c in clk if c < 0.95 * early), None)
            if throttled is not None:
                hot = next((s["temp_C"] for s in samples
                            if s["elapsed_s"] >= throttled), "?")
                print(f"Clock dropped below 95% of early peak at t={throttled}s "
                      f"(temp ~{hot} C). Cross-check against the power ceiling.")
            else:
                print("No clock throttling detected (SM clock stayed within 5% of "
                      "its early value).")

    # --- plot: clock and temperature vs time --------------------------------
    if samples:
        ts = [s["elapsed_s"] for s in samples if isinstance(s["sm_clock_MHz"], int)]
        clocks = [s["sm_clock_MHz"] for s in samples if isinstance(s["sm_clock_MHz"], int)]
        temps = [s["temp_C"] for s in samples if isinstance(s["sm_clock_MHz"], int)]
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(ts, clocks, color="tab:blue", label="SM clock (MHz)")
        ax1.set_xlabel("Elapsed (s)")
        ax1.set_ylabel("SM clock (MHz)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(ts, temps, color="tab:red", label="Temp (C)")
        ax2.set_ylabel("Temperature (C)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        plt.title(f"Part E: sustained load -- {info['name']}")
        fig.tight_layout()
        plt.savefig(f"part_e_{uuid}.png", dpi=140)

    bl.append_run_log("PartE", uuid,
                      f"sustained {args.minutes}min N={N}; -> {samp_path}, {thru_path}")
    print(f"\nWrote {samp_path}, {thru_path}, part_e_{uuid}.png. RUN_LOG.txt updated.")


if __name__ == "__main__":
    main()
