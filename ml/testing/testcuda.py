import torch, sys
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Built with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
x = torch.randn(2,2, device="cuda" if torch.cuda.is_available() else "cpu")
print("Tensor device:", x.device)
