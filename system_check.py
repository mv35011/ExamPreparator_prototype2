import torch
import transformers
import torchvision
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
print(transformers.__version__)
print(torchvision.__version__)
print(torch.version.cuda)  # Should output '12.1'
print(torch.cuda.is_available())