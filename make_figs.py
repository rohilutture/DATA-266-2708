"""Build every figure from results/<gpu>/*.json -> figures/*.png. Cards found under results/ are overlaid on Part A plots."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
TAGS = sorted(p.name for p in (ROOT / "results").iterdir() if p.is_dir())


def J(tag, name):
    p = ROOT / "results" / tag / name
    return json.loads(p.read_text()) if p.exists() else None


def ok_rows(sw):
    return [r for r in sw["rows"] if not r.get("oom")]


def partA():
    fig, ax = plt.subplots(figsize=(7, 4.5)); fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for t in TAGS:
        sw = J(t, "sweep_baseline.json")
        if not sw: continue
        rows = ok_rows(sw)
        ax.plot([r["bs"] for r in rows], [r["throughput"] for r in rows], "o-", label=t)
        ax2.plot([r["bs"] for r in rows], [r["peak_mem_gb"] for r in rows], "o-", label=t)
        oom = [r for r in sw["rows"] if r.get("oom")]
        if oom:
            ax.axvline(oom[0]["bs"], ls=":", c="red", lw=1)
    for a, y, f, n in ((ax, "samples / second", fig, "fig_partA_throughput.png"),
                       (ax2, "peak allocated memory (GiB)", fig2, "fig_partA_memory.png")):
        a.set_xscale("log", base=2); a.set_xlabel("batch size (fp32, HW4 settings)"); a.set_ylabel(y)
        a.grid(alpha=.3); a.legend(); f.tight_layout(); f.savefig(FIG / n, dpi=150)
    plt.close("all")


def partB():
    for t in TAGS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for n in ("baseline", "amp", "ckpt", "optim8", "amp+optim8", "amp+ckpt", "amp+ckpt+optim8"):
            sw = J(t, f"sweep_{n}.json")
            if sw:
                rows = ok_rows(sw)
                ax.plot([r["bs"] for r in rows], [r["throughput"] for r in rows], "o-", ms=3, label=n)
        ax.set_xscale("log", base=2); ax.set_xlabel("batch size"); ax.set_ylabel("samples / second")
        ax.set_title(f"Part B — throughput vs batch size ({t})"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / f"fig_partB_throughput_{t}.png", dpi=150); plt.close(fig)


def hw4a():
    for t in TAGS:
        r = J(t, "hw4a_baseline.json")
        if not r: continue
        ep = [l["epoch"] for l in r["logs"]]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(ep, [l["train_loss"] for l in r["logs"]], "o-", label="train"); ax[0].plot(ep, [l["val_loss"] for l in r["logs"]], "o-", label="val")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(ep, [l["val_acc"] for l in r["logs"]], "o-"); ax[1].axvline(r["best_epoch"], ls=":", c="g")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val accuracy"); ax[1].grid(alpha=.3)
        fig.suptitle(f"HW4.a training curves ({t})"); fig.tight_layout(); fig.savefig(FIG / f"fig_hw4a_curves_{t}.png", dpi=150); plt.close(fig)


def kv():
    for t in TAGS:
        k = J(t, "kv_cache.json")
        if not k: continue
        rows = [r for r in k["hf"]["rows"] if "analytic_kv_gb" in r]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([r["ctx"] for r in rows], [r["analytic_kv_gb"] for r in rows], "o-", label="analytic (config)")
        m = [r for r in rows if "measured_kv_gb" in r]
        ax.plot([r["ctx"] for r in m], [r["measured_kv_gb"] for r in m], "s--", label="measured (HF cache tensors)")
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("context length (tokens), batch 8")
        ax.set_ylabel("KV cache (GiB)"); ax.set_title(f"KV cache vs context ({t})"); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(FIG / f"fig_kv_{t}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    for f in (partA, partB, hw4a, kv):
        f()
    print("figures ->", FIG)
