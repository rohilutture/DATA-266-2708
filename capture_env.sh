#!/usr/bin/env bash
# Records the GPU UUID capture + environment. Run on the machine/GPU you measure on. Deliverable: uuid_capture_<tag>.txt
set -u
TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9]\+/-/g; s/^-//; s/-$//')
OUT="uuid_capture_${TAG}.txt"
{
  echo "# captured $(date -Is) on $(hostname)"
  echo "## nvidia-smi -L"; nvidia-smi -L
  echo; echo "## nvidia-smi (full)"; nvidia-smi
  echo; echo "## python / torch / cuda"
  "${PY_TRAIN:-python}" -c "import sys, torch; print(sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())" 2>&1
  echo; echo "## pip freeze (key packages)"; "${PY_TRAIN:-python}" -m pip freeze 2>/dev/null | grep -Ei '^(torch|transformers|datasets|bitsandbytes|vllm|scikit-learn|numpy)=='
} | tee "$OUT"
echo "wrote $OUT  (also take a screenshot of 'nvidia-smi -L' if the instructor wants one)"
