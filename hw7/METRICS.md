# METRICS - HW7 AutoEncoders on FashionMNIST

- SID4=2708, SEED=2708
- Dataset: FashionMNIST | latent_dim=32 | epochs=20, batch=128, lr=0.001

## Test-set reconstruction (lower is better)

| model | MSE / pixel | BCE / pixel |
|---|---|---|
| FC-AE | 0.01005 | 0.27364 |
| VAE | 0.01282 | 0.28183 |

FC-AE optimizes reconstruction only; the VAE trades a little reconstruction accuracy
for a regularized latent space it can **generate** from (see hw7_vae_samples.png).
