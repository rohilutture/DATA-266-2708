# HW12 - GPTQ quantization & knowledge distillation

Files
- `HW12_GPTQ_Distillation_executed.ipynb` - executed notebook (outputs intact)
- `METRICS.md` - Table HW11.1 and Table HW11.2
- `RUN_LOG.txt` - console output of the run (filtered), training log, isolated measurements
- `AI_USE.md` - AI-use disclosure (incl. deviations: 1 epoch, workarounds)
- `measure_isolated.py`, `isolated_measurements.jsonl` - one-model-per-process latency / memory measurements
- `loss_trend_temperature_sweep.png`, `loss_trend_alpha_sweep.png`, `progress.log` - loss trends and sample training log

Model checkpoints (`artifacts/gpt2-gptq-4bit`, `artifacts/best_distilled_student`) are NOT in git (each > 100 MB, GitHub's file limit); they stay on the local machine in `Downloads/hw12_work/artifacts/`.

Commands used
```
python -m venv venv
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install transformers datasets pandas matplotlib scikit-learn jupyter
pip install gptqmodel --no-deps   # + its deps; see AI_USE.md for the pcre shim
jupyter nbconvert --to notebook --execute HW12_fast.ipynb --output HW12_executed.ipynb --ExecutePreprocessor.timeout=-1
python measure_isolated.py {base|gptq|teacher|student}
```
