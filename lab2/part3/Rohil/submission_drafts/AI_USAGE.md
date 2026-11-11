# AI_USAGE

Edit to match your actual use; state it plainly.

## Tools used in the build
- **Claude Code (Claude Sonnet 5.5)** was used to draft the Part 3 pipeline code (`src/p3_agents.py`), the notebook build script, the scenario and red-team JSON files, the README, and a first draft of the Part 3 report section and these files.
- **Models inside the system (not development tools):** Qwen2.5-VL-3B-Instruct, Stable Diffusion v1.5, InstructPix2Pix, CLIP ViT-B/32, and the Part 2 Naruto LoRA. No hosted LLM API is called at run time.

## What was checked by the team
- Reported numbers were copied from the Kaggle-executed notebook outputs and CSVs; none were typed by hand from memory.
- Evaluation thresholds were not tuned after seeing results. The R01 red-team failure was reported as scored.
- TODO: each member confirms they read, ran and can explain the code and the report section.

## What was not AI-generated
- TODO: human pass/fail and accept/revise judgments (both raters), the demo, and any text you rewrote.
