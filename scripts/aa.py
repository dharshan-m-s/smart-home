import torch
import numpy as np

print("NumPy:", np.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))