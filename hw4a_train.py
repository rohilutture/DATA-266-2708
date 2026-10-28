"""HW4.a part 1: BERT sentiment fine-tune on Twitter data. Also records the HW4 peak-memory / wall-clock baseline."""
import json, subprocess
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import config as C
from common import BASE, DEV, RES, get_data, gpu_name_and_vram, save, train_full


def main():
    d = get_data()
    name, vram = gpu_name_and_vram()
    print(f"GPU: {name} ({vram:.1f} GiB)  dataset: {d['info']['dataset']}  rows {d['info']['row_range']}")
    print("split sizes:", d["info"]["split_sizes"])
    save("data_split.json", {**d["info"], "indices": d["idx"]})   # exact rows for METRICS.md / reproducibility

    r = train_full(BASE, keep_best=True)
    model = r.pop("_model")

    # ---- epoch at which validation stopped improving, and what training past it cost
    logs = r["logs"]
    best = r["best_epoch"]
    extra_s = sum(l["epoch_s"] for l in logs[best:])
    r["extra_epochs_after_best"] = len(logs) - best
    r["extra_train_s_after_best"] = extra_s
    r["val_acc_change_after_best"] = logs[-1]["val_acc"] - r["best_val_acc"]

    # ---- confusion matrix & per-class P/R on the TEST split (best-val-epoch weights)
    y_true = d["test"][2].tolist()
    y_pred = r["test_preds_best"]
    cm = confusion_matrix(y_true, y_pred, labels=range(C.NUM_LABELS))
    p, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(C.NUM_LABELS), zero_division=0)
    worst = int(np.argmin(f1))
    test_texts = [d["texts"][i] for i in d["idx"]["test"]]
    wrong = [(test_texts[i], y_true[i], y_pred[i]) for i in range(len(y_true)) if y_true[i] == worst and y_pred[i] != worst]
    r["confusion_matrix"] = cm.tolist()
    r["per_class"] = [dict(label=C.LABEL_NAMES[k], precision=float(p[k]), recall=float(rec[k]), f1=float(f1[k]),
                           support=int(sup[k])) for k in range(C.NUM_LABELS)]
    r["worst_class"] = C.LABEL_NAMES[worst]
    r["worst_class_misclassified"] = [dict(text=t, true=C.LABEL_NAMES[a], pred=C.LABEL_NAMES[b]) for t, a, b in wrong[:5]]
    for k in ("test_preds_final", "test_preds_best"):
        r.pop(k)

    # ---- save and reload for custom inference
    out_dir = RES / "hw4a_model"
    model.save_pretrained(out_dir)
    from transformers import BertForSequenceClassification, BertTokenizer
    BertTokenizer.from_pretrained(C.MODEL_NAME).save_pretrained(out_dir)
    tok = BertTokenizer.from_pretrained(out_dir)
    reloaded = BertForSequenceClassification.from_pretrained(out_dir).to(DEV).eval()
    demo = ["I absolutely love this phone, best purchase all year!", "Traffic again. Great. Just what I needed.",
            "The meeting is at 3pm in room 204."]
    enc = tok(demo, padding=True, truncation=True, max_length=C.MAX_LEN, return_tensors="pt").to(DEV)
    with torch.no_grad():
        pred = reloaded(**enc).logits.argmax(-1).tolist()
    r["custom_inference"] = [dict(text=t, pred=C.LABEL_NAMES[k]) for t, k in zip(demo, pred)]

    save("hw4a_baseline.json", r)

    # ---- human-readable report with blanks for the one-line diagnoses
    L = ["# HW4.a — BERT sentiment fine-tune (auto-generated numbers; fill the DIAGNOSIS lines yourself)", "",
         f"GPU: {name}; dataset: {d['info']['dataset']}; rows {d['info']['row_range']}; seed {C.SEED}; "
         f"split {d['info']['split_sizes']}", "",
         "| epoch | train loss | val loss | val acc | epoch s |", "|---|---|---|---|---|"]
    L += [f"| {l['epoch']} | {l['train_loss']:.4f} | {l['val_loss']:.4f} | {l['val_acc']:.4f} | {l['epoch_s']:.1f} |" for l in logs]
    L += ["", f"Validation accuracy stopped improving after epoch **{best}** (best val acc {r['best_val_acc']:.4f}). "
          f"Training {r['extra_epochs_after_best']} more epoch(s) cost {extra_s:.1f}s of GPU time and changed val acc by "
          f"{r['val_acc_change_after_best']:+.4f}.", "",
          f"Test acc (best-val-epoch weights) {r['test_acc_best']:.4f}; (final-epoch weights) {r['test_acc']:.4f}.", "",
          "Confusion matrix (rows = true, cols = predicted; order: " + ", ".join(C.LABEL_NAMES) + "):", "```"]
    L += [str(row) for row in r["confusion_matrix"]] + ["```", "", "| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    L += [f"| {c['label']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c['support']} |" for c in r["per_class"]]
    L += ["", f"Worst class (lowest F1): **{r['worst_class']}**. Five misclassified examples:", ""]
    for i, e in enumerate(r["worst_class_misclassified"], 1):
        L += [f"{i}. \"{e['text']}\" — true: {e['true']}, predicted: {e['pred']}", "   DIAGNOSIS: <write one line here>"]
    L += ["", f"HW4 baseline measurements (used for HW4.5 only): peak GPU memory {r['peak_mem_gb']:.2f} GiB allocated "
          f"({r['peak_reserved_gb']:.2f} GiB reserved), wall-clock {r['wall_s']:.1f}s total / {r['train_s']:.1f}s training-only, "
          f"{r['throughput']:.0f} samples/s."]
    (RES / "HW4A_REPORT.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
