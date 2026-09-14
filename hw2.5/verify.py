import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))  # want (12, 0)
print("sm_120 native:", "sm_120" in torch.cuda.get_arch_list())                        # MUST be True
print("arch_list:", torch.cuda.get_arch_list())
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
print("matmul OK:", tuple(y.shape))