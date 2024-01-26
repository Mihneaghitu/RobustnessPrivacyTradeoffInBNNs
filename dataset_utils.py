from typing import Tuple

import torch
import torchvision
from torchvision import transforms


def load_mnist() -> Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return train_data, test_data
