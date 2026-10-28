# GPU-hour record

## Estimated from measured throughput

GPU-hours = epochs × n_train ÷ (samples/s) ÷ 3600.

### nvidia-geforce-rtx-4090
- 4 × 3500 ÷ 436 = 32 s = **0.0089 GPU-h** for the HW4.a fine-tune.
- For scale: the full 45,615-row train split would be 0.116 GPU-h at the same throughput.

## Actual lab session usage (fill in)

| date | GPU | start | end | hours | what ran |
|---|---|---|---|---|---|
| | | | | | |

Tip: `RUN_LOG.txt` has a timestamp for every stage; sum them for the real total.