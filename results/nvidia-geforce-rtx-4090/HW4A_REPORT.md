# HW4.a — BERT sentiment fine-tune (auto-generated numbers; fill the DIAGNOSIS lines yourself)

GPU: NVIDIA GeForce RTX 4090; dataset: cardiffnlp/tweet_eval/sentiment[train]; rows 0..4999 (inclusive, 0-indexed); seed 42; split {'train': 3500, 'val': 750, 'test': 750}

| epoch | train loss | val loss | val acc | epoch s |
|---|---|---|---|---|
| 1 | 0.9116 | 0.8248 | 0.6147 | 8.5 |
| 2 | 0.6091 | 0.7339 | 0.6813 | 7.8 |
| 3 | 0.3721 | 0.8790 | 0.6653 | 7.9 |
| 4 | 0.1948 | 1.0583 | 0.6520 | 7.9 |

Validation accuracy stopped improving after epoch **2** (best val acc 0.6813). Training 2 more epoch(s) cost 15.8s of GPU time and changed val acc by -0.0293.

Test acc (best-val-epoch weights) 0.6773; (final-epoch weights) 0.6573.

Confusion matrix (rows = true, cols = predicted; order: negative, neutral, positive):
```
[55, 41, 9]
[25, 254, 57]
[8, 102, 199]
```

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| negative | 0.625 | 0.524 | 0.570 | 105 |
| neutral | 0.640 | 0.756 | 0.693 | 336 |
| positive | 0.751 | 0.644 | 0.693 | 309 |

Worst class (lowest F1): **negative**. Five misclassified examples:

1. ""I may be Asian, but it doesn't mean I automatically like T-ara. I'm not even familiar with KPOP. #DirectionersFuneral"" — true: negative, predicted: neutral
   DIAGNOSIS: <write one line here>
2. "when you hungry af but Dunkin is closed on Sunday's &amp; conn don't open until 10" — true: negative, predicted: neutral
   DIAGNOSIS: <write one line here>
3. ""Well Jokes apart, today or tomorrow Yakub has to meet with with 72 Hoors. S"" — true: negative, predicted: neutral
   DIAGNOSIS: <write one line here>
4. "HOF president: Frank Gifford 'built the game': The passing of Frank Gifford cast a cloud over Sunday's proceed..." — true: negative, predicted: neutral
   DIAGNOSIS: <write one line here>
5. "@user 25 March 1972: An IRA volunteer was killed when two IRA units engaged each other in error on the Springfield Road in Belfast.??" — true: negative, predicted: neutral
   DIAGNOSIS: <write one line here>

HW4 baseline measurements (used for HW4.5 only): peak GPU memory 3.74 GiB allocated (4.01 GiB reserved), wall-clock 34.9s total / 32.1s training-only, 437 samples/s.