# AI_USE.md — HW6 (Self-Supervised / Contrastive, STL-10)

> Rohil Utture · DATA 266 · SID4=2708
> Adjust the wording so it sounds like me — a generic answer earns zero (Req. 0.5).

## 1. Which parts did I use an assistant for, and which did I write myself?
I used an assistant to scaffold the notebook — STL-10 loading and the shared 500-image
labeled subset, the ResNet-18 encoder/heads, the rotation and SimCLR datasets, the NT-Xent
loss, the frozen-encoder linear-probe helper, and the nearest-neighbour visualization. I set
my own parameters (SID4=2708 → SEED=2708), chose the 4090-scale config (96-px inputs, all
100k unlabeled images for rotation, 50k for SimCLR), ran everything, diagnosed the hang
below, read the results, and wrote the Part D comparison from my own numbers and figures.

## 2. One specific thing it produced that was wrong. Paste the wrong output.
The data pipeline it wrote used multiple DataLoader workers (`NUM_WORKERS = 6`). Part A
(which uses a built-in torchvision dataset) ran fine, but Part B **hung** — the cell sat for
~22 minutes with **no epoch output at all**:

```
def train_rotation():
    ...
    loader = dl(RotationDataset(unlab_tensor, ROT_IDX), ROT_BS, shuffle=True, drop_last=True)
    ...
# cell running 22m 33s, zero "[rot] epoch ..." lines printed
```

The cause: `RotationDataset` / `SimCLRDataset` are classes defined inside the notebook, and
on Windows the DataLoader uses spawn-based workers that have to pickle the dataset — which
fails/hangs for notebook-defined classes. Part A escaped it only because it wraps a built-in
`STL10` dataset.

## 3. How did I find out? What did the failure look like?
Part A finished normally but Part B produced no per-epoch log for 22 minutes, while a 4090
should do a rotation epoch in well under a minute. "Completes one stage, then dead silence on
the next with a custom Dataset" pointed straight at the worker/pickling path, not slow
training.

## 4. What did I change, and why does my version work?
I set `NUM_WORKERS = 0` in the Config cell and re-ran Parts B and C. With 0 workers the
dataset runs in the main process, so there is nothing to pickle to a child process and the
hang disappears — `[rot] epoch 0 ...` printed within a minute and both parts trained to
completion (Rotation 0.3935, SimCLR 0.5838 test accuracy). The trade-off is slower CPU-side
augmentation, but it is correct and reproducible. (A cleaner alternative would be to move the
two Dataset classes into a separate `.py` module so they pickle and workers can be used again.)

## If I wanted to list what else I verified
- Encoder actually frozen in the probes: only the linear layer updates (I set
  `requires_grad=False` and kept the encoder in `eval()`).
- NT-Xent pairs the right views: targets are `(i+N) % 2N`, so each view's positive is its
  own second augmentation.
- The three encoders learned different spaces: this shows up in both the linear-probe
  accuracies and the nearest-neighbour figures.
