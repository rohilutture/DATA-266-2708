#!/usr/bin/env bash
# Full pipeline. Everything is appended to RUN_LOG.txt with timestamps (deliverable).
#   PY_TRAIN : python of the training venv  (torch, transformers, datasets, bitsandbytes, sklearn, matplotlib)
#   PY_VLLM  : python of the vLLM venv      (vLLM pins its own torch, so keep it separate)
# Dry run first:  HW45_QUICK=1 ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY_TRAIN=${PY_TRAIN:-python}
PY_VLLM=${PY_VLLM:-python}
LOG=RUN_LOG.txt
stage() { echo "" | tee -a $LOG; echo "===== $(date -Is) :: $* =====" | tee -a $LOG; }
run()   { stage "$*"; "$@" 2>&1 | tee -a $LOG; }

stage "environment capture"; ./capture_env.sh 2>&1 | tee -a $LOG
run $PY_TRAIN hw4a_train.py                       # HW4.a fine-tune, curves, confusion matrix, HW4 baseline memory/time
run $PY_TRAIN partAB.py all                       # Part A sweep + Part B techniques + combos + accumulation
run $PY_VLLM  serve_bench.py --engine hf          # Table HW4.1 (HF pipeline baseline)
run $PY_VLLM  serve_bench.py --engine vllm        # Table HW4.1 (vLLM batch 1/8/32)
run $PY_VLLM  kv_cache.py                         # Part D KV cache vs context
run $PY_TRAIN make_figs.py
run $PY_TRAIN make_metrics.py
stage "done"
