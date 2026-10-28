# METRICS - HW9 Latent Diffusion (FashionMNIST)

- SID4=2708, SEED=2708
- VAE: stabilityai/sd-vae-ft-mse (frozen), scale=0.18215 | latent 4x16x16
- Diffusion: T=1000, beta in [0.0001, 0.02] linear
- U-Net: base=64, mults=[1, 2, 4], params=5,583,300, self-attn @16x16 + bottleneck
- Training: 50 epochs, lr=2e-05, batch=128, p_uncond=0.1, guidance w=3.0

## Results

| metric | value |
|---|---|
| VAE reconstruction MSE (pixels [-1,1]) | 0.000775 |
| Final training eps-MSE | 0.097272 |
