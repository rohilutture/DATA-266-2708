# METRICS — DATA 266 Homework 1 (Rohil Utture, SID4 = 2708)

Parameters: `SEED=2708`, `HP_ID=2` (Learning-rate-high). Modified model = hidden **[64, 32]**, lr **0.003**, 30 epochs.
Split: 70 / 15 / 15, `random_state=SEED`, stratified. Metric: **test accuracy**.

## Task 2 — Diabetes NN: test accuracy over 3 training seeds (SEED, SEED+1, SEED+2)

Split is fixed at `random_state=SEED`; only the training seed varies across the three runs.

| Framework  | Model               | seed 2708 | seed 2709 | seed 2710 | **mean** | **std** |
|------------|---------------------|-----------|-----------|-----------|----------|---------|
| PyTorch    | Baseline            | 0.7281    | 0.7368    | 0.7368    | **0.7339** | 0.0041 |
| PyTorch    | Modified (HP_ID 2)  | 0.7544    | 0.7456    | 0.7632    | **0.7544** | 0.0072 |
| TensorFlow | Baseline            | 0.7632    | 0.7368    | 0.7544    | **0.7515** | 0.0109 |
| TensorFlow | Modified (HP_ID 2)  | 0.7368    | 0.7456    | 0.7368    | **0.7398** | 0.0041 |

Loss-curve summary (SEED run) — supports the overfitting read on the modified model:

| Framework  | Model    | final train BCE | final val BCE | min val BCE (epoch) |
|------------|----------|-----------------|---------------|---------------------|
| PyTorch    | Baseline | 0.402           | 0.438         | 0.437 (ep 15)       |
| PyTorch    | Modified | 0.318           | 0.515         | 0.438 (ep 6)        |
| TensorFlow | Baseline | 0.394           | 0.468         | 0.456 (ep 16)       |
| TensorFlow | Modified | 0.299           | 0.561         | 0.452 (ep 5)        |

## Task 3 — CUDA matmul (Tesla T4, sm_75, -std=c++14)

Profiler used: **nvprof** (Nsight-legacy). Its GPU-activities breakdown separates kernel from transfer:
matmulKernel ~ **90.1%**, CUDA memcpy DtoH ~ 5.8%, CUDA memcpy HtoD ~ 4.1%.

| Matrix size | CPU (ms)   | GPU kernel (ms) | H2D+D2H (ms) | Speedup (CPU / end-to-end) |
|-------------|------------|-----------------|--------------|----------------------------|
| 256         | 20.26      | 0.103           | 1.035        | 17.82x                     |
| 1024        | 3376.59    | 9.193           | 5.249        | 233.81x                    |
| 4096        | 631476.29  | 323.966         | 79.089       | 1566.73x                   |

Correctness (max abs diff vs CPU): 2.29e-05 / 9.16e-05 / 4.27e-04 for N = 256 / 1024 / 4096.

Crossover (Task 5): GPU end-to-end is beneficial at every tested size; the smallest measured beneficial
size is **N = 256** (17.82x), so the crossover lies **below 256**, not at zero — fixed kernel-launch and
O(N^2) transfer overhead would dominate for a tiny matrix. See cuda.ipynb §5 / report §3.
