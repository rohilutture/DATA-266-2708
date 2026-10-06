# METRICS - HW6 Self-Supervised / Contrastive (STL-10)

- SID4=2708, SEED=2708
- Backbone: resnet18 (no pretrained) | img_size=96
- Labeled subset: 500 | rotation unlabeled: 100000 | simclr unlabeled: 50000 | tau=0.2

## Test accuracy (full 8000-image test set)

| part | model | test accuracy |
|---|---|---|
| A | Supervised (10% labels, end-to-end) | 0.4000 |
| B | Rotation SSL + linear probe | 0.3935 |
| C | SimCLR + linear probe | 0.5838 |
