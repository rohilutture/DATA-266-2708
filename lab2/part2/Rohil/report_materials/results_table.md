# Part 2 results (all numbers printed in `p2_finetune.ipynb`, saved in `eval/p2_results.json`)

Base model: Stable Diffusion 1.5. 20 fixed prompts (10 in-domain, 5 out-of-domain, 5 compositional) x seeds 0 and 1 = 40 images per model. Same sampler for every model (DPM-Solver++, 30 steps, CFG 7.5, 512x512).

## Training runs (same data order, steps, batch, learning rate, seed)

| | Run 1: LoRA r=4 | Run 2: LoRA r=32 |
|---|---|---|
| Trainable parameters | 797,184 (0.093% of UNet) | 6,377,472 (0.737% of UNet) |
| Learning rate | 1e-4 (cosine, 100 warmup) | 1e-4 (cosine, 100 warmup) |
| Steps | 2000 | 2000 |
| Batch size | 4 | 4 |
| Peak GPU memory (allocated) | 2.86 GB | 2.93 GB |
| Training time | 2505.5 s (0.696 GPU-hours) | 2503.4 s (0.695 GPU-hours) |
| GPU | Tesla T4 | Tesla T4 |

## Evaluation (baseline = base SD 1.5, final = run 2)

| Measurement | Base | Run 1 (r=4) | Run 2 (r=32, final) |
|---|---|---|---|
| **Primary:** KID, in-domain vs 122 test images (lower is better) | 0.1077 +/- 0.0136 | 0.0483 +/- 0.0088 | **0.0134 +/- 0.0056** |
| **Primary:** CLIP score, all 40 images | **32.24** | 31.35 | 30.71 |
| CLIP score, in-domain | **31.03** | 29.73 | 28.32 |
| CLIP score, out-of-domain | **33.85** | 33.15 | 33.67 |
| CLIP score, compositional | **33.06** | 32.79 | 32.52 |
| **Secondary:** LPIPS diversity between seeds, all prompts | **0.809** | 0.781 | 0.751 |
| LPIPS diversity, in-domain | **0.842** | 0.797 | 0.752 |
| **Human (final vs base):** win rate of final run, 25 blinded pairs, 3 raters | - | - | 29.3% (28%, 24%, 36%); majority vote 32%; mean pairwise agreement 0.84 |
| **Efficiency:** training GPU-hours / peak memory | - | 0.696 h / 2.86 GB | 0.695 h / 2.93 GB |
| Generation time per image (s) | 5.58 | 5.77 | 5.80 |
| NSFW-flagged outputs (of 40) | 0 | 3 | 9 |

KID uses only 20 generated in-domain images per model, so it is noisy; the +/- is the standard deviation over 100 random subsets.

## Memorization check (run 2, 10 in-domain images, seed 0, CLIP image cosine to nearest training image)

Per-prompt similarities: 0.902, 0.806, 0.886, 0.872, 0.890, 0.905, 0.745, 0.772, 0.762, 0.766. The three closest pairs are 0.905 (prompt 5), 0.902 (prompt 0) and 0.889 (prompt 4). For reference, real test images are on average 0.905 (max 0.970) similar to their nearest training image, so the generated images are no closer to the training set than genuinely new images are.
