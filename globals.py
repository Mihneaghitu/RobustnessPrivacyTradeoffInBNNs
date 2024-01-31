import torch

cuda_available = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda' if cuda_available else 'cpu')
