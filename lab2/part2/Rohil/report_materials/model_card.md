---
license: creativeml-openrail-m
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
datasets:
- lambdalabs/naruto-blip-captions
tags: [stable-diffusion, text-to-image, lora, diffusers, naruto]
---
# rohil542/naruto-sd15-run2_lora_r32

LoRA adapter (rank 32) for Stable Diffusion 1.5 that generates Naruto-style anime images.

```python
from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("rohil542/naruto-sd15-run2_lora_r32")
image = pipe("a man with a hat and glasses on", num_inference_steps=30, guidance_scale=7.5).images[0]
```

**Base model.** Stable Diffusion 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`), frozen. Only LoRA matrices on the attention projections (`to_q`, `to_k`, `to_v`, `to_out.0`) are trained.

**Data.** `lambdalabs/naruto-blip-captions`: 1221 image-caption pairs (BLIP captions, mean 8.9 words). 0 corrupted and 0 duplicate images were found. Split 90/10 with seed 266: 1099 training and 122 test images, cropped to 512x512. The dataset card states no license.

**Training.** 2000 steps, batch size 4, learning rate 1e-4 (cosine schedule, 100 warmup steps), AdamW, fp16 with gradient checkpointing, rank 32 and alpha 32, 6,377,472 trainable parameters (0.737% of the UNet). Seed 266. One Tesla T4: 0.695 GPU-hours, 2.93 GB peak memory. A second run with rank 4 (`rohil542/naruto-sd15-run1_lora_r4`) used identical settings.

**Results** (20 fixed prompts, seeds 0 and 1, same sampler as the base model):

| | Base SD 1.5 | This adapter (r=32) | r=4 adapter |
|---|---|---|---|
| KID vs test images, in-domain (lower is better) | 0.1077 | **0.0134** | 0.0483 |
| CLIP score (all prompts) | **32.24** | 30.71 | 31.35 |
| LPIPS diversity between seeds | **0.809** | 0.751 | 0.781 |
| Human win rate vs base (25 blinded pairs, 3 raters) | - | 29.3% | - |

KID is computed from only 20 generated images and is noisy.

**Limitations.**
- Narrow style and cast; the model imitates one anime art style and handles other domains poorly.
- Out-of-domain and compositional prompts can be partly ignored (for example it drops the second character in "one in red armor and one in blue armor").
- Lower diversity between seeds than the base model.
- Captions are machine-generated and short.
- The 90/10 split is not fully independent because neighbouring anime frames look alike.
- In our check, generated images were not closer to the training images than real test images are (CLIP cosine up to 0.905 vs 0.970 for real test images), but this does not rule out memorization of rarer samples.
- The Stable Diffusion safety checker flagged 9 of 40 outputs; all 9 were reviewed by eye and show clothed anime characters (likely false positives of a photo-trained checker).

**License and intended use.** The training images are copyrighted anime frames and the dataset card gives no license, so this adapter is for non-commercial research and education only. The CreativeML OpenRAIL-M license of the base model also applies. Do not use it to imitate or misrepresent the original work or for commercial purposes.
