"""
analyze_e.py -- recompute Part E throttle onset and a thermal/power summary from
the CSVs part_e_sustained.py already wrote. No need to re-run the 20-min load.

Run in the folder that has the part_e_*.csv files:
    python analyze_e.py
"""
import csv
import glob
import sys


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(r, k):
    try:
        return float(r.get(k, ""))
    except (TypeError, ValueError):
        return None


samp_files = sorted(glob.glob("part_e_samples_*.csv"))
thru_files = sorted(glob.glob("part_e_throughput_*.csv"))
if not samp_files:
    sys.exit("No part_e_samples_*.csv found in this folder.")

srows = load(samp_files[-1])
clk = [(fnum(r, "elapsed_s"), fnum(r, "sm_clock_MHz")) for r in srows]
clk = [(t, c) for t, c in clk if t is not None and c is not None]
if not clk:
    sys.exit("No usable SM-clock samples in the CSV.")

total = max(t for t, _ in clk)

# cold-boost reference = highest clock in the first 30 s (ignores ramp-up)
warm = [c for t, c in clk if t <= 30]
ref = max(warm) if warm else max(c for _, c in clk)

# throttle onset = first sample AFTER 30 s below 95% of the reference
onset = next((t for t, c in clk if t > 30 and c < 0.95 * ref), None)

last5 = [c for t, c in clk if t >= total - 300]
steady_clk = sum(last5) / len(last5) if last5 else float("nan")

temps = [fnum(r, "temp_C") for r in srows if fnum(r, "temp_C") is not None]
powers = [fnum(r, "power_W") for r in srows if fnum(r, "power_W") is not None]

print(f"samples: {len(srows)} over {total:.0f}s  (file: {samp_files[-1]})")
print(f"cold-boost reference SM clock (max, first 30s): {ref:.0f} MHz")
print(f"steady SM clock (mean, last 5 min):             {steady_clk:.0f} MHz "
      f"({100*steady_clk/ref:.1f}% of cold-boost)")

if onset is not None:
    at = next(r for r in srows if fnum(r, "elapsed_s") == onset)
    print(f"THROTTLE ONSET: t={onset:.0f}s  "
          f"(clock {fnum(at,'sm_clock_MHz'):.0f} MHz, "
          f"temp {fnum(at,'temp_C'):.0f} C, power {fnum(at,'power_W'):.0f} W)")
else:
    print("THROTTLE ONSET: none (SM clock stayed within 5% of cold-boost after 30s)")

if temps:
    print(f"temperature: start {temps[0]:.0f} C -> max {max(temps):.0f} C")
if powers:
    tag = "POWER-LIMITED (at/near 575 W cap)" if max(powers) >= 560 else "below power cap"
    print(f"power: max {max(powers):.0f} W of 575 W -> {tag}")

if thru_files:
    trows = load(thru_files[-1])
    tp = [(fnum(r, "elapsed_s"), fnum(r, "TFLOPS")) for r in trows]
    tp = [(t, v) for t, v in tp if t is not None and v is not None]
    if tp:
        ttot = max(t for t, _ in tp)
        first30 = [v for t, v in tp if t <= 30]
        tl5 = [v for t, v in tp if t >= ttot - 300]
        if first30 and tl5:
            pk, st = sum(first30) / len(first30), sum(tl5) / len(tl5)
            print(f"throughput: peak(30s) {pk:.1f} -> steady(5min) {st:.1f} TFLOPS "
                  f"({100*st/pk:.1f}%)")

print("\nInterpretation hints:")
print(" - If throttle onset is 'none' but throughput still sagged a few %, the")
print("   drop is small clock drift, not a hard throttle -> report onset = none.")
print(" - If max power sits at ~575 W, you are power-limited, not thermal-limited;")
print("   the clock backs off to stay under the board power cap.")
