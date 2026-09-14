import torch
try:
    x = torch.empty(18_000_000_000, dtype=torch.float16, device="cuda")  # ~36 GB
    print("NO OOM -> sysmem fallback is STILL ON (36GB 'fit' on a 32GB card)")
    del x
except torch.cuda.OutOfMemoryError:
    print("OOM as expected -> fallback is OFF. Good.")