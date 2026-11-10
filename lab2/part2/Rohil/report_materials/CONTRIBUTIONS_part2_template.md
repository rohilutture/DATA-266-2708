# CONTRIBUTIONS (Part 2 section; merge into the repo-level CONTRIBUTIONS.md)

Fill in the TODO items. Do not invent recordings or dates; list only sessions that actually took place.

## Part 2: fine-tuning diffusion models

| Task | Led by | Done by |
|---|---|---|
| 2.1 Data preparation (cleaning, split, resize) | TODO | TODO |
| 2.2 Fine-tuning (two LoRA runs) | TODO | TODO |
| 2.3 Image generation (20 prompts, grids, HF dataset) | TODO | TODO |
| 2.4 Evaluation (KID, CLIP, LPIPS, human study) | TODO | TODO |
| Safety (license, NSFW review, memorization) | TODO | TODO |
| Model card and report section | TODO | TODO |

The partner who did not lead a task must also have run the complete Part 2 pipeline at least as a baseline: TODO (who ran it, when, which GPU, what result).

Human evaluation raters: evaluator1 = TODO, evaluator2 = TODO, evaluator3 = TODO (outside the team: yes / no).

## Meeting and knowledge-transfer recordings

| Date | Topic | Link |
|---|---|---|
| TODO | TODO (for example: Part 2 design, rank choice, evaluation plan) | TODO |

---

# AI_USAGE (Part 2 section; merge into the repo-level AI_USAGE.md)

- Claude Code (Anthropic; Claude desktop app): TODO confirm. Used to write and debug the Part 2 notebook (training loop, generation, metrics, human-evaluation pipeline), to fix a pandas bug in the metrics cell, and to draft the report text, model card and architecture diagram. All numbers in the report come from the executed notebook, and the failure-mode, safety and model-choice judgements were checked by the team against the generated images.
- Other tools: TODO.
