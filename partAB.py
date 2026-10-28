"""Part A (batch sweep to the memory wall) and Part B (memory techniques, one at a time, then combinations).

usage:
  python partAB.py all            # everything (A + B + combos)
  python partAB.py sweep NAME     # sweep one config to OOM
  python partAB.py train NAME     # full training run for accuracy at the HW4 batch size
  python partAB.py accum          # gradient-accumulation measurements
Each probe / training run executes in a fresh subprocess so an OOM cannot poison later measurements.
"""
import json, subprocess, sys
from pathlib import Path

import config as C
from common import BASE, RES, load, save

T = {
    "baseline": dict(BASE),
    "amp": dict(BASE, amp="bf16"),
    "ckpt": dict(BASE, ckpt=True),
    "optim8": dict(BASE, optim="adamw8bit"),
    "accum": dict(BASE, accum=C.ACCUM_STEPS),
    # best-combination candidates
    "amp+optim8": dict(BASE, amp="bf16", optim="adamw8bit"),
    "amp+ckpt": dict(BASE, amp="bf16", ckpt=True),
    "amp+ckpt+optim8": dict(BASE, amp="bf16", ckpt=True, optim="adamw8bit"),
}
SINGLES = ["baseline", "amp", "ckpt", "optim8"]
COMBOS = ["amp+optim8", "amp+ckpt", "amp+ckpt+optim8"]
HERE = Path(__file__).resolve()


def child(mode, tech, bs=None):
    arg = json.dumps(dict(mode=mode, tech=tech, bs=bs))
    p = subprocess.run([sys.executable, str(HERE), "_child", arg], capture_output=True, text=True)
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("RESULT:"):
            return json.loads(line[7:])
    blob = (p.stderr + p.stdout).lower()
    if "out of memory" in blob:
        return dict(oom=True, bs=bs)
    sys.stderr.write(p.stderr[-2000:])
    raise RuntimeError(f"child failed (mode={mode}, bs={bs})")


def sweep(name):
    tech, rows, bs = T[name], [], 1
    while bs <= C.BS_CAP:
        r = child("probe", tech, bs)
        r["bs"] = bs
        rows.append(r)
        print(f"[{name}] bs={bs:<6} " + ("OOM" if r.get("oom") else
              f"{r['throughput']:.0f} samp/s  peak {r['peak_mem_gb']:.2f} GiB  est epoch {r['est_epoch_s']:.1f}s"), flush=True)
        if r.get("oom"):
            break
        bs *= 2
    ok = [r for r in rows if not r.get("oom")]
    out = dict(name=name, tech=tech, rows=rows, max_batch=ok[-1]["bs"] if ok else 0,
               hit_cap=bool(rows) and not rows[-1].get("oom"))
    save(f"sweep_{name}.json", out)
    return out


def train(name, bs=C.HW4_BATCH, tag=None):
    r = child("full", T[name], bs)
    save(f"full_{tag or name}.json", r)
    print(f"[{name}] full run bs={bs}: test acc {r.get('test_acc', float('nan')):.4f}  "
          f"peak {r.get('peak_mem_gb', float('nan')):.2f} GiB  {r.get('throughput', 0):.0f} samp/s", flush=True)
    return r


def accum_measure():
    """Gradient accumulation: physical batch 8 x accum 16 = effective 128, vs a direct batch of 128."""
    eff = C.ACCUM_PHYSICAL * C.ACCUM_STEPS
    a = child("probe", T["accum"], C.ACCUM_PHYSICAL)
    direct = child("probe", T["baseline"], eff)
    full_a = child("full", T["accum"], C.ACCUM_PHYSICAL)
    full_direct = child("full", T["baseline"], eff)
    save("accum.json", dict(effective_batch=eff, physical=C.ACCUM_PHYSICAL, accum_steps=C.ACCUM_STEPS,
                            probe_accum=a, probe_direct=direct, full_accum=full_a, full_direct=full_direct,
                            n_opt_steps=C.PROBE_OPT_STEPS))
    print(f"[accum] effective {eff}: {a.get('time_n_opt_steps_s', float('nan')):.2f}s for {C.PROBE_OPT_STEPS} steps; "
          f"direct bs{eff}: " + ("OOM" if direct.get("oom") else f"{direct['time_n_opt_steps_s']:.2f}s"))


def run_all():
    for n in SINGLES + COMBOS:
        sweep(n)
    for n in SINGLES + COMBOS:
        train(n)
    accum_measure()


def _child_main(arg):
    import torch
    from common import probe, train_full
    a = json.loads(arg)
    if a["mode"] == "probe":
        r = probe(a["tech"], a["bs"])
    else:
        r = train_full(a["tech"], bs=a["bs"])
        for k in ("test_preds_final",):
            r.pop(k, None)
    print("RESULT:" + json.dumps(r))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "_child":
        _child_main(sys.argv[2])
    elif cmd == "all":
        run_all()
    elif cmd == "sweep":
        sweep(sys.argv[2])
    elif cmd == "train":
        train(sys.argv[2])
    elif cmd == "accum":
        accum_measure()
    else:
        sys.exit(__doc__)
