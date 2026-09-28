# METRICS - HW5 LoRA / DialogSum

- SID4=2708, SEED=2708, SLICE=708
- Model: `google/flan-t5-small` | Dataset: `neil-code/dialogsum-test`
- Subset: 200 train / 50 eval | epochs=3, batch=8, lr=0.001

## ROUGE - baseline vs fine-tuned (r=16), eval subset

| model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---|---|---|---|
| baseline (no FT) | 0.1744 | 0.0580 | 0.1522 | 0.1542 |
| fine-tuned r=16 | 0.3764 | 0.1347 | 0.3059 | 0.3063 |

Fine-tuned r=16 final training loss: **1.4295**

## Rank sweep (Task 7)

| r | alpha | trainable params | final loss | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|---|
| 4 | 8 | 172,032 | 1.5348 | 0.3549 | 0.1357 | 0.3025 |
| 16 | 32 | 688,128 | 1.4295 | 0.3764 | 0.1347 | 0.3059 |
