# Part 3 - Multimodal multi-agent system

Files: `src/p3_agents.py` (source, percent format) -> `notebooks/p3_agents.ipynb` (build with `powershell tools/build_notebook.ps1 -Name p3_agents`),
`eval/p3_scenarios.json` (15 scenarios + rubrics) and `eval/p3_redteam.json` (10 red-team cases), both committed **before** the first run.

## Run (Kaggle T4/P100 or Colab T4, internet on)
1. Put the repo on the machine (`git clone`), `cd data266-lab2`.
2. Set `P2_LORA_REPO` to your Part 2 final adapter, e.g. `HF_USER/naruto-sd15-run2_lora_r32` (env var or the config cell). Add an `HF_TOKEN` secret if it is private.
3. Open `notebooks/p3_agents.ipynb`, Restart & Run All. Outputs land in `p3/` (`traces/*.json`, `outputs/*.png`, `eval/*.csv|png`).
4. Demo app: run with `RUN_APP=1` (the last code cells launch Gradio with a share link).
5. Human evaluation: fill `p3/human_eval/p3_review_sheet.csv` into `p3_rater_A.csv` and `p3_rater_B.csv` (columns `scenario,pass,should_accept`), optional `p3_resolved.csv`, then re-run section 8-9.

## Models (all open weights, run locally; no hosted API)
Qwen/Qwen2.5-VL-3B-Instruct (vision understanding, prompt rewriting, critic judge) · stable-diffusion-v1-5/stable-diffusion-v1-5 (img2img) ·
Part 2 LoRA (domain style, from Hugging Face) · timbrooks/instruct-pix2pix (edit) · openai/clip-vit-base-patch32 (critic similarity).

No trained weights are produced in Part 3; it loads the Part 2 adapter. Put that Hugging Face link in `links.txt` and in the Part 3 folder of the submission.
