import torch

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS is not available")
        exit()


mps_device = torch.device("mps")

# Create a Tensor directly on the mps device
x = torch.ones(5, device=mps_device)

# Any operation happens on the GPU
y = x * 2

print(y)
