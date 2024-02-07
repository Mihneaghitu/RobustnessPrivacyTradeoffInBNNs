import os
from typing import Tuple

import torchvision
from torchvision import transforms


def load_mnist(rel_path: str = ".") -> Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    curr_dir = os.getcwd()
    os.chdir(curr_dir + "/" + rel_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    return train_data, test_data
