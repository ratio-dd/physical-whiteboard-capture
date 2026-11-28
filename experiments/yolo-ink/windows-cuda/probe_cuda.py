import torch

assert torch.cuda.is_available(), "CUDA unavailable"
device = torch.device("cuda:0")
x = torch.ones((4, 4), device=device)
y = x @ x
print({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "tensor_device": str(y.device),
    "tensor_sum": float(y.sum().item()),
})
