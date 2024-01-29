import torch

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
