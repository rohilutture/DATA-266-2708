"""Shared data / model / training / probing code."""
import json, math, os, random, re, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

import config as C

ROOT = Path(__file__).resolve().parent
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    # WDDM/WSL lets CUDA spill into system RAM instead of raising OOM; cap below physical VRAM
    # (the desktop holds ~2 GiB) so a too-large batch fails with a real OOM.
    torch.cuda.set_per_process_memory_fraction(0.9)


# ------------------------------------------------------------------ housekeeping
def gpu_tag() -> str:
    t = os.environ.get("HW45_GPU_TAG")
    if t:
        return t
    try:
        name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()[0]
    except Exception:
        name = "cpu-or-unknown"
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


RES = ROOT / "results" / gpu_tag()
RES.mkdir(parents=True, exist_ok=True)


def save(name, obj):
    (RES / name).write_text(json.dumps(obj, indent=2))


def load(name, default=None):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else default


def set_seed(s=C.SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def reset_peak():
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30 if DEV.type == "cuda" else 0.0


def peak_reserved_gb():
    return torch.cuda.max_memory_reserved() / 2**30 if DEV.type == "cuda" else 0.0


def gpu_name_and_vram():
    if DEV.type != "cuda":
        return "cpu", 0.0
    p = torch.cuda.get_device_properties(0)
    return p.name, p.total_memory / 2**30


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


# ------------------------------------------------------------------ data
_DATA = None


def load_rows():
    if C.CSV_PATH:
        import pandas as pd
        df = pd.read_csv(C.CSV_PATH)
        sub = df.iloc[C.SLICE:C.SLICE + C.N_ROWS]
        return list(sub["text"].astype(str)), list(map(int, sub["label"])), len(df)
    from datasets import load_dataset
    ds = load_dataset(C.DATASET_NAME, C.DATASET_CONFIG, split=C.DATASET_SPLIT)
    end = min(C.SLICE + C.N_ROWS, len(ds))
    sub = ds.select(range(C.SLICE, end))
    return list(sub["text"]), list(sub["label"]), len(ds)


def get_data():
    """Same rows, same 70/15/15 split, same SEED -> identical for HW4.a and every HW4.5 run."""
    global _DATA
    if _DATA is not None:
        return _DATA
    texts, labels, total = load_rows()
    n = len(texts)
    perm = np.random.RandomState(C.SEED).permutation(n)
    n_tr, n_va = int(C.SPLIT[0] * n), int(C.SPLIT[1] * n)
    parts = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained(C.MODEL_NAME)
    enc = tok(texts, padding="max_length", truncation=True, max_length=C.MAX_LEN, return_tensors="pt")
    y = torch.tensor(labels)
    d = {"info": {
        "dataset": C.CSV_PATH or f"{C.DATASET_NAME}/{C.DATASET_CONFIG}[{C.DATASET_SPLIT}]",
        "slice": C.SLICE, "row_range": f"{C.SLICE}..{C.SLICE + n - 1} (inclusive, 0-indexed)",
        "n_rows": n, "total_rows_in_source": total, "seed": C.SEED,
        "split_sizes": {k: int(len(v)) for k, v in parts.items()}}, "texts": texts, "labels": labels,
        "idx": {k: v.tolist() for k, v in parts.items()}}
    for k, ix in parts.items():
        ix_t = torch.as_tensor(ix)
        d[k] = (enc["input_ids"][ix_t], enc["attention_mask"][ix_t], y[ix_t])
    _DATA = d
    return d


# ------------------------------------------------------------------ model / optimiser
BASE = dict(amp=None, ckpt=False, accum=1, optim="adamw")   # the "HW4 settings"


def build_model(ckpt=False):
    from transformers import BertForSequenceClassification
    set_seed()
    m = BertForSequenceClassification.from_pretrained(C.MODEL_NAME, num_labels=C.NUM_LABELS)
    if ckpt:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return m.to(DEV)


def build_optim(model, name):
    ps = model.parameters()
    if name == "adamw":
        return torch.optim.AdamW(ps, lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    if name == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(ps, lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    if name == "adafactor":   # fallback "memory-efficient optimiser" if bitsandbytes is unavailable
        from transformers.optimization import Adafactor
        return Adafactor(ps, lr=C.LR, scale_parameter=False, relative_step=False, warmup_init=False)
    raise ValueError(name)


def autocast(amp):
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    return torch.autocast(device_type=DEV.type, dtype=dt, enabled=(dt is not None and DEV.type == "cuda"))


def make_scaler(amp):
    return torch.amp.GradScaler("cuda") if (amp == "fp16" and DEV.type == "cuda") else None


def backward_step_pieces(loss, scaler, accum):
    (scaler.scale(loss / accum) if scaler else loss / accum).backward()


def opt_step(opt, scaler):
    if scaler:
        scaler.step(opt); scaler.update()
    else:
        opt.step()
    opt.zero_grad(set_to_none=True)


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, split, amp, bs):
    model.eval()
    ids, att, y = (t.to(DEV) for t in get_data()[split])
    tot, correct, preds = 0.0, 0, []
    for i in range(0, len(y), bs):
        with autocast(amp):
            out = model(input_ids=ids[i:i + bs], attention_mask=att[i:i + bs], labels=y[i:i + bs])
        tot += out.loss.float().item() * len(y[i:i + bs])
        p = out.logits.argmax(-1)
        preds.append(p)
    preds = torch.cat(preds)
    return tot / len(y), (preds == y).float().mean().item(), preds.cpu().tolist()


# ------------------------------------------------------------------ full training run
def train_full(tech, bs=C.HW4_BATCH, epochs=C.EPOCHS, keep_best=False):
    d = get_data()
    amp, accum = tech["amp"], tech["accum"]
    model = build_model(tech["ckpt"])
    opt = build_optim(model, tech["optim"])
    scaler = make_scaler(amp)
    ids, att, y = (t.to(DEV) for t in d["train"])
    n = len(y)
    g = torch.Generator().manual_seed(C.SEED)
    sync(); reset_peak()
    wall0 = time.perf_counter()
    train_time, logs, best_state, best_acc, best_ep = 0.0, [], None, -1.0, -1
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        nb = math.ceil(n / bs)
        loss_sum = torch.zeros((), device=DEV)
        sync(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for i in range(nb):
            ix = perm[i * bs:(i + 1) * bs].to(DEV)
            with autocast(amp):
                loss = model(input_ids=ids[ix], attention_mask=att[ix], labels=y[ix]).loss
            loss_sum += loss.detach().float() * len(ix)
            backward_step_pieces(loss, scaler, accum)
            if (i + 1) % accum == 0 or i == nb - 1:
                opt_step(opt, scaler)
        sync(); ep_time = time.perf_counter() - t0
        train_time += ep_time
        vl, va, _ = evaluate(model, "val", amp, bs)
        logs.append(dict(epoch=ep + 1, train_loss=loss_sum.item() / n, val_loss=vl, val_acc=va, epoch_s=ep_time))
        print(f"  epoch {ep+1}/{epochs} train_loss={logs[-1]['train_loss']:.4f} val_loss={vl:.4f} "
              f"val_acc={va:.4f} ({ep_time:.1f}s)", flush=True)
        if keep_best and va > best_acc:
            best_acc, best_ep = va, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    wall = time.perf_counter() - wall0
    out = dict(tech=tech, bs=bs, epochs=epochs, logs=logs, wall_s=wall, train_s=train_time,
               throughput=n * epochs / train_time, peak_mem_gb=peak_gb(), peak_reserved_gb=peak_reserved_gb(),
               n_train=n, avg_epoch_s=train_time / epochs)
    tl, ta, tp = evaluate(model, "test", amp, bs)
    out.update(test_loss=tl, test_acc=ta, final_val_acc=logs[-1]["val_acc"], test_preds_final=tp)
    if keep_best:
        out["best_epoch"] = best_ep
        out["best_val_acc"] = best_acc
        model.load_state_dict(best_state)
        tl, ta, tp = evaluate(model, "test", amp, bs)
        out.update(test_acc_best=ta, test_preds_best=tp)
        out["_model"] = model
    return out


# ------------------------------------------------------------------ throughput / memory probe
def probe(tech, bs, n_opt=C.PROBE_OPT_STEPS, warmup=3):
    """Time `n_opt` optimiser steps (each = accum micro-batches of size bs) on random rows of the train split.
    Returns {"oom": True} if the card runs out of memory."""
    try:
        d = get_data()
        amp, accum = tech["amp"], tech["accum"]
        model = build_model(tech["ckpt"]); model.train()
        opt = build_optim(model, tech["optim"]); scaler = make_scaler(amp)
        ids, att, y = (t.to(DEV) for t in d["train"])
        n = len(y)
        gen = torch.Generator(device=DEV).manual_seed(C.SEED)
        for s in range(warmup + n_opt):
            if s == warmup:
                sync(); reset_peak(); t0 = time.perf_counter()
            for _ in range(accum):
                ix = torch.randint(0, n, (bs,), device=DEV, generator=gen)
                with autocast(amp):
                    loss = model(input_ids=ids[ix], attention_mask=att[ix], labels=y[ix]).loss
                backward_step_pieces(loss, scaler, accum)
            opt_step(opt, scaler)
        sync(); dt = time.perf_counter() - t0
        samples = bs * accum * n_opt
        return dict(oom=False, bs=bs, accum=accum, effective_bs=bs * accum, throughput=samples / dt,
                    sec_per_opt_step=dt / n_opt, time_n_opt_steps_s=dt, n_opt_steps=n_opt,
                    peak_mem_gb=peak_gb(), peak_reserved_gb=peak_reserved_gb(),
                    est_epoch_s=n / (samples / dt))
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if is_oom(e):
            return dict(oom=True, bs=bs, accum=tech["accum"])
        raise
