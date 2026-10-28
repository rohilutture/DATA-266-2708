# AI_USE.md — HW9 (Latent Diffusion on FashionMNIST)

Rohil Utture · DATA 266 · SID4=2708

## 1. Which parts did I use an assistant for, and which did I write myself?
I used an assistant to scaffold the full latent-diffusion pipeline: loading and freezing the
pretrained SD VAE (`stabilityai/sd-vae-ft-mse`), the FashionMNIST→latent encoding with the
0.18215 scaling factor, the diffusion schedule and the forward `q_sample`, the conditional
self-attention U-Net, the classifier-free-guidance training loop with 10% label dropout, and
the guided DDPM sampler. I set my own parameters (SID4=2708 → SEED=2708), chose to run the full
60k training set on my GPU, ran everything, read the reconstruction / loss / sample results,
and wrote the analysis from my own outputs and figures.

## 2. One specific thing it produced that was wrong
On this assignment the code ran correctly the first time — no shape error, no broken loss, no
bad metric. So instead of inventing a failure, below is what I checked to confirm the
implementation was actually right (as Req. 0.5 allows).

## What I verified instead, and how
- **U-Net wiring is correct.** The notebook asserts the U-Net output shape equals the latent
  shape (4×16×16) before any training, and it passed — so the down/up/skip channel counts and
  the attention blocks at 16×16 and the bottleneck are consistent. The model has 5,583,300
  parameters.
- **The VAE latent space is faithful.** Encoding → scaling by 0.18215 → unscaling → decoding
  reconstructs FashionMNIST almost losslessly: reconstruction MSE = 0.000775 (pixels in
  [-1,1]). This is what makes doing diffusion in the 4×16×16 latent (48× spatial compression)
  reasonable.
- **The forward process destroys signal as expected.** Decoding noised latents at
  t∈{0,200,500,800,1000} goes from a clean garment to high-frequency noise by t≈1000, i.e.
  ᾱ_t → 0, which is the distribution the sampler starts from.
- **CFG is wired the right way.** Using ε=(1+w)ε_cond − w·ε_uncond with w=3 produced
  class-specific samples (the 64-image grid cycles classes 0–9 and the garments are
  recognizable). Had I swapped the cond/uncond terms, guidance would have pushed away from the
  class and the samples would have been wrong — they weren't.
- **Training behaved normally.** The ε-MSE fell from 0.4085 (epoch 0) and plateaued around
  0.097 (final 0.0973), the expected shape for noise-prediction, since pure noise can't be
  predicted exactly.
- **Reproducibility.** Seeds (Python/NumPy/Torch) were set from my SID before training and
  sampling.
