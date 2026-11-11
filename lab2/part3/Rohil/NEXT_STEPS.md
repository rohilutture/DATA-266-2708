# Part 3 next steps (deadline: Nov 17, 2026, 11:59 PM PT)

Status: Part 3 code, Kaggle run (commit 45e0dd2) and results are done and in this folder. Drafts are in `submission_drafts/`.
Kaggle numbers: without critique 13/15, median 8.78 s; with critique 14/15, median 29.35 s, mean revisions 1.07; red-team attack success 1/10 (R01).

- [ ] **Human evaluation (both members, independently).** On Kaggle, open `p3/human_eval/p3_review_sheet.csv` (the copy in `run_outputs` or a local download may be the local run's only). Each rater fills `pass` and `should_accept` and saves `p3_rater_A.csv` / `p3_rater_B.csv` (columns `scenario,pass,should_accept`); optional `p3_resolved.csv` for discussed disagreements. Rerun notebook sections 8-9 so critic agreement and pass agreement fill in.
- [ ] Replace the **[PENDING]** fields in `submission_drafts/Part3_report_section.md` with the human-evaluation numbers.
- [ ] Fill every TODO in `CONTRIBUTIONS.md` (team number, partner, who did what, meeting recordings), `AI_USAGE.md` (edit to your real use) and `links.txt`.
- [ ] **Demo rehearsal:** run with `RUN_APP=1`; practice three scenarios, including one that must fail safely (R03 or R04). Both members must be able to explain the code.
- [ ] **Team repo:** create private `data266-lab2-team##`, push the code, share with the instructor and ISAs, tag `p3-v1.0`. The notebook's commit line must match the tagged commit (Kaggle run is at 45e0dd2; otherwise rerun on Kaggle after final code, or note the commit in `links.txt`).
- [ ] **Final package:** one combined IEEE report PDF (max 12 pages per part, includes the GitHub link); zip with a folder per part, each with the executed notebook, weights (Part 3 trains none: list the Part 2 adapter HF link `rohil542/naruto-sd15-run2_lora_r32`), and required outputs; plus `CONTRIBUTIONS.md`, `AI_USAGE.md`, `links.txt`.
- [ ] Optional (+2): deploy the Gradio app as a Hugging Face Space.
