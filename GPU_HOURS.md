# GPU-hour record

## Estimated from measured throughput

GPU-hours = epochs × n_train ÷ (samples/s) ÷ 3600.

### nvidia-geforce-rtx-4090
- 4 × 3500 ÷ 436 = 32 s = **0.0089 GPU-h** for the HW4.a fine-tune.
- For scale: the full 45,615-row train split would be 0.116 GPU-h at the same throughput.

## Actual lab session usage (fill in)

| date | GPU | start | end | hours | what ran |
|---|---|---|---|---|---|
| 2026-10-27 | RTX 4090 | 15:51 | 16:07 | 0.27 | first dry-run attempt (stopped: slow process startup from the Windows drive, venvs moved) |
| 2026-10-27 | RTX 4090 | 16:15 | 16:36 | 0.35 | `HW45_QUICK=1` dry run incl. vLLM fix reruns (passed) |
| 2026-10-27 | RTX 4090 | 16:36 | 17:00 | 0.40 | first full-run attempt (stopped: batch sweep spilled into system RAM, invalid) |
| 2026-10-27 | RTX 4090 | 17:21 | 17:46 | 0.41 | **final full run** (`run_all.sh`; source of all reported numbers) |

Final run: 17:21:04 to 17:45:38 per `RUN_LOG.txt` = 0.41 h. Total GPU session about 1.4 h; times of the aborted attempts are approximate.

Tip: `RUN_LOG.txt` has a timestamp for every stage; sum them for the real total.