# AI_USE.md — HW7 (GANs concepts + AutoEncoders on FashionMNIST)

> Rohil Utture · DATA 266 · SID4=2708
> Adjust the wording so it sounds like me.

## 1. Which parts did I use an assistant for, and which did I write myself?
I used an assistant to scaffold the Part 2 notebook — FashionMNIST loading, the FC-AE and VAE
model classes, the training loops, the evaluation (test MSE/BCE), and the reconstruction /
sampling plots — and to help organize the Part 1 GAN write-ups. I set my own parameters
(SID4=2708 → SEED=2708), ran the training on my GPU, read the results, and wrote the Part 2
comparison from my own numbers. For Part 1 I went through the GAN architecture, mode-collapse,
and variant answers and put them in my own words so I can explain them.

## 2. One specific thing it produced that was wrong
On this assignment the code it produced ran correctly the first time — I did not hit a tensor
shape error, a broken loss, or a bad metric. So instead of inventing a failure, below is what
I checked to make sure the implementation was actually right (as Req. 0.5 allows).

## What I verified instead, and how
- **Loss scaling is consistent.** The VAE's BCE and KL are both summed and then divided by the
  batch size, so they're on the same per-image scale. If they weren't, the KL would be
  negligible and the prior samples would be noise — but my `hw7_vae_samples.png` shows
  recognizable garments, so the KL is actually doing its job. Final VAE training ELBO was
  240.4/img = BCE 224.2 + KL 16.2.
- **KL term behaves correctly.** It should be 0 only when the encoder outputs `μ=0, logσ²=0`
  (i.e. exactly the `N(0,I)` prior) and positive otherwise. My KL stayed around 15–16 per
  image, which is sensible for a 32-dim latent that carries real information.
- **The two models behave as theory predicts.** The FC-AE reached a lower test reconstruction
  error than the VAE (MSE/px 0.01005 vs 0.01282), which is exactly what you'd expect because
  the FC-AE optimizes reconstruction alone while the VAE gives up a little accuracy for a
  regularized latent space. The reconstruction figure confirms the VAE outputs are slightly
  blurrier.
- **Decoder output range.** Both decoders end in a sigmoid so outputs are in [0,1], which is
  required for `binary_cross_entropy` against the [0,1] pixel targets — I confirmed training
  ran without a BCE domain error.
- **Reproducibility.** I set the Python/NumPy/Torch seeds from my SID (SEED=2708) before each
  model so the runs are repeatable.
