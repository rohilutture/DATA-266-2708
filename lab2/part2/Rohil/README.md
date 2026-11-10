# Lab 2, Part 2 (Rohil): Fine-tuning Stable Diffusion 1.5 on Naruto BLIP captions

Two LoRA runs (rank 4 and rank 32) on `lambdalabs/naruto-blip-captions`, compared with the base model on 20 fixed prompts.

## Contents

| Path | What it is |
|---|---|
| `p2_finetune.ipynb` | Executed notebook with all outputs (Kaggle, Tesla T4, seed 266). First cell prints commit, versions, GPU and seed. |
| `requirements-p2.txt` | Pinned library versions used by the notebook. |
| `eval/` | Results (`p2_results.json/.csv`), prompts (`prompts_p2.json`), human-eval result, memorization check, loss curves, grids, NSFW review figure. |
| `human_eval/` | Rater letters (`ratings/*.csv`), answer key (`key.json`) and the blank sheet. The 25 pair images are not committed (see the submission zip). |
| `adapters/*/train_stats.json` | Training statistics for each run. The adapter weights are on Hugging Face, not here. |
| `images/metadata.jsonl` | Prompt, seed, model, NSFW flag and CLIP score for each of the 120 generated images. The images are on Hugging Face. |
| `report_materials/` | Draft report section, architecture diagram (SVG), results table, model card, figures, contributions and AI-usage templates. |

## Hugging Face

- Adapters: `rohil542/naruto-sd15-run1_lora_r4`, `rohil542/naruto-sd15-run2_lora_r32` (final run)
- Generated images: `rohil542/naruto-sd15-generations` (dataset, 120 rows)

## How to run

1. Open `p2_finetune.ipynb` on Kaggle (or Colab) with a T4 GPU.
2. Set `HF_USER` in the configuration cell and add an `HF_TOKEN` secret to push results.
3. Run all cells. Training takes about 0.7 GPU-hours per run; generation about 12 minutes.
4. Human evaluation needs real rater letters in the `RATINGS` cell (25 letters per rater, `L` or `R`).

## Results summary (from the notebook)

| | Base | Run 1 (r=4) | Run 2 (r=32, final) |
|---|---|---|---|
| KID, in-domain (lower is better) | 0.1077 | 0.0483 | 0.0134 |
| CLIP score | 32.24 | 31.35 | 30.71 |
| LPIPS diversity | 0.809 | 0.781 | 0.751 |
| Human win rate vs base (3 raters) | - | - | 29.3% |
| Training GPU-hours / peak memory | - | 0.696 h / 2.86 GB | 0.695 h / 2.93 GB |

Full table and discussion: `report_materials/results_table.md` and `report_materials/part2_report_section.md`.

## Notes

- Dataset license: not stated on the dataset card; the data are copyrighted anime frames, so the models are for non-commercial research only.
- `eval/prompts_p2.json` was written by the notebook before training, but it was committed to Git only after the runs.
- Weights and generated images are kept out of GitHub on purpose (they are on Hugging Face).
