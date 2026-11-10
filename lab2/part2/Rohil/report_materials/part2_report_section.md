# Part 2: Fine-Tuning Stable Diffusion for Domain-Specific Image Generation (draft for the IEEE report)

Draft text for the Part 2 section. Move it into the IEEE template, rewrite it in your own words (the lab checks plagiarism across groups), and keep every number identical to the notebook output. Figures are in `figures/`; the diagram is `architecture_part2.svg` (export it to PDF/PNG for the report).

## A. Problem and approach

We fine-tune a pretrained text-to-image model so that it produces images in the style of one domain, and we compare two fine-tuning settings against the unmodified base model. The base model is Stable Diffusion 1.5, which stays frozen. The domain is the Naruto anime, using `lambdalabs/naruto-blip-captions`. Both runs train low-rank adapters (LoRA) on the attention projections of the UNet and differ only in rank (4 and 32). Training, generation and evaluation run on one 16 GB Tesla T4 (Kaggle). Figure 1 (`architecture_part2.svg`) shows the pipeline.

## B. Data (Task 2.1)

The dataset has 1221 image-caption pairs. Images were decoded and checked, and exact duplicates (MD5 of pixels) and near duplicates (difference hash with Hamming distance of at most 2) were searched for. We found 0 corrupted and 0 duplicate images, so all 1221 were kept. They were split 90/10 with seed 266 into 1099 training and 122 test images. The test images are used only for evaluation. Images were center-cropped to a square and resized to 512x512. Captions are short BLIP captions (mean 8.9 words).

Two limitations: (1) The captions are machine-generated and generic ("a man with a hat and glasses on"), so they carry no character names, actions or scene detail, and some are inaccurate. The dataset also covers a small cast and a few art styles. (2) The images are copyrighted anime frames, and neighbouring frames look alike, so the 90/10 split is not fully independent. This can make test images look easier than truly new ones and weakens the memorization check.

## C. Fine-tuning (Task 2.2)

Both runs use the same data order (a seeded generator), steps, batch size, learning rate and seed. VAE latents (and horizontally flipped latents) and CLIP text embeddings are computed once and cached. The UNet is kept in fp16 and frozen, the LoRA weights are trained in fp32 with mixed-precision autocast and gradient scaling (the T4 has no bf16), and gradient checkpointing is on. The loss is the mean squared error between the predicted and the true noise.

Table I. Training runs (from the notebook).

| | Run 1: LoRA r=4 | Run 2: LoRA r=32 |
|---|---|---|
| Trainable parameters | 797,184 | 6,377,472 |
| Learning rate / steps / batch | 1e-4 / 2000 / 4 | 1e-4 / 2000 / 4 |
| Peak GPU memory | 2.86 GB | 2.93 GB |
| Training time | 0.696 GPU-hours | 0.695 GPU-hours |

Memory and time are almost identical because the rank changes only the small adapter matrices; the frozen UNet dominates the cost. The training loss of both runs stays between 0.075 and 0.097 and shows no clear downward trend (`figures/loss_curves.png`). The two curves overlap because the same batches and noise were used. The loss is therefore not informative here, and we judge training by the generated images and the metrics. Both adapters are on Hugging Face (`rohil542/naruto-sd15-run1_lora_r4`, `rohil542/naruto-sd15-run2_lora_r32`).

## D. Image generation (Task 2.3)

We wrote 20 fixed prompts before generation: 10 in-domain (test-set captions chosen with seed 266), 5 out-of-domain (for example "a photograph of a golden retriever running on a sandy beach") and 5 compositional (for example "two cartoon characters, one in red armor and one in blue armor, standing back to back"). Each prompt was generated with seeds 0 and 1 for the base model and both runs (120 images), using the same sampler (DPM-Solver++, 30 steps, guidance scale 7.5, 512x512). Figure 2 (`figures/grid_base_vs_runs.png`) compares the three models for 8 prompts. All 120 images, with prompt, seed and model as metadata, are in the Hugging Face dataset `rohil542/naruto-sd15-generations`.

On in-domain prompts, both fine-tuned models turn the base model's photographs into anime frames, and rank 32 does this more strongly. On out-of-domain prompts such as the dog, the neon city, the ramen and the sports car, all models still produce photographs.

## E. Evaluation (Task 2.4)

Metrics. KID compares the 20 in-domain generations of each model with the 122 test images (lower is better; with so few generated images it is noisy, so we report its standard deviation over 100 subsets). CLIP score is 100 times the cosine between the CLIP embeddings (`openai/clip-vit-base-patch32`) of an image and its prompt. Diversity is the average LPIPS (AlexNet) distance between the two seeds of the same prompt. Efficiency is training GPU-hours and peak memory.

Table II. Baseline vs final system (full table in `results_table.md`).

| Measurement | Base | Run 1 (r=4) | Run 2 (r=32, final) |
|---|---|---|---|
| KID, in-domain (lower is better) | 0.1077 +/- 0.0136 | 0.0483 +/- 0.0088 | 0.0134 +/- 0.0056 |
| CLIP score (all prompts) | 32.24 | 31.35 | 30.71 |
| LPIPS diversity (all prompts) | 0.809 | 0.781 | 0.751 |
| Human win rate vs base | - | - | 29.3% |
| Training GPU-hours / peak memory | - | 0.696 h / 2.86 GB | 0.695 h / 2.93 GB |

What the table shows. Fine-tuning moves the generated images much closer to the test images: KID falls from 0.108 to 0.048 (r=4) and 0.013 (r=32), and the gaps are larger than the standard deviations. The cost is a lower CLIP score (32.24 to 30.71) and lower diversity (0.809 to 0.751), with the largest drop for rank 32. CLIP score drops most on in-domain prompts (31.03 to 28.32), partly because CLIP may rate stylised anime less favourably against generic captions, so this should be read cautiously.

Human evaluation. We built 25 image pairs (final run vs base model, left/right randomised, no labels) and three raters (both team members and one person outside the team) each chose the better image for domain style and prompt match. The final run won 29.3% of all ratings (per rater 28%, 24% and 36%; majority-vote win rate 32%), with mean pairwise agreement of 0.84. Raters chose the left image in most pairs while the fine-tuned model was on the right in 16 of 25 pairs, so position bias probably lowered its win rate. The pairs also mix in-domain, out-of-domain and compositional prompts, where a base-model photo can legitimately win on prompt match. [Add the category breakdown and the share of "L" picks from the notebook cell, and the exact instruction given to raters.]

## F. Failure modes

1. Compositional prompts are not followed (prompts 15, 16 and 18). For "one in red armor and one in blue armor" all three models draw one red figure and the second character is missing or cut off. For "a sword in the left hand and a scroll in the right hand" no model draws a scroll. For "green hair and yellow eyes", run 2 gives yellow hair and puts green only on the eyebrows and clothes.
2. Prompt ignored and domain bleed on an out-of-domain prompt (prompt 19, "a tiny cartoon character standing in front of a giant dragon"). The base model draws a child in front of a dragon statue, run 1 keeps the child but loses most of the dragon, and run 2 draws one brown cartoon animal with neither a dragon nor a small character.
3. Lower diversity. Average LPIPS between seeds falls from 0.809 to 0.781 (r=4) and 0.751 (r=32).
4. A black bar on the left edge of some images (for example prompts 10, 15 and 16) that also appears for the base model, so it is not caused by fine-tuning.

Run chosen: run 2 (r=32). It is the closest to the test images (KID 0.0134) and gives the most consistent anime style. It has the lowest CLIP score and diversity, the most NSFW-filter flags, more prompt-ignoring on out-of-domain and compositional prompts, and a human win rate below 50%. We chose style fidelity because the task is domain-specific generation; if prompt alignment mattered more, r=4 would be the safer choice.

## G. Safety analysis (C4, about one page)

License. The training images are copyrighted anime frames, and the dataset card on the Hugging Face Hub has no license metadata. The base model uses the CreativeML OpenRAIL-M license. We therefore publish the adapters for non-commercial research and education only, state this in the model card, and do not distribute the training images.

NSFW filter. We ran the Stable Diffusion safety checker on all 120 generated images, keeping the images and recording the flags. It flagged 0 of 40 base images, 3 of 40 from r=4 and 9 of 40 from r=32. We reviewed all 12 flagged images by eye (`figures/nsfw_flagged.png`): all show clothed anime characters, mostly close-up portraits with large flat skin-toned regions, and none is explicit. We consider them likely false positives of a checker designed for photographs, and we note that the filter should not be relied on alone in a deployed system. [Both members should confirm this review.]

Memorization check. For 10 in-domain images from run 2 (seed 0), we found the most similar training image by CLIP image similarity (`eval/memorization_check.csv`). The three closest pairs have cosine 0.905, 0.902 and 0.889 (`figures/memorization_top3.png`) and show different characters, poses and colours. For comparison, real test images have a mean cosine of 0.905 (max 0.970) to their closest training image. The generated images are thus no closer to the training set than genuinely new images, and we found no sign of copying. This check is limited to 10 images and to CLIP similarity, so it does not prove the absence of memorization.

## H. Reproducibility

Notebook `p2_finetune.ipynb` (executed, outputs saved); the first cell prints commit, library versions, GPU and seed 266; requirements in `requirements-p2.txt`; weights and images on Hugging Face. Note for the report: the prompt file `eval/prompts_p2.json` was written by the notebook before training, but it was committed to Git only after the runs, so the Git timestamp does not show it as pre-registered.
