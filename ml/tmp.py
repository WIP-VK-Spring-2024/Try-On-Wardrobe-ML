import torch


message = "CUDA IS AVAILABLE" if torch.cuda.is_available() else "ERROR! CUDA IS NOT AVAILABLE"

print(message)
