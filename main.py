from typing import Tuple

import torch
import torchvision
from torchvision import transforms

from deterministic.pipeline import test_mnist_vanilla, train_mnist_vanilla


def load_mnist() -> Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return train_data, test_data

def main():
    is_cuda_available = torch.cuda.is_available()
    device = torch.device('cuda:0' if is_cuda_available else 'cpu')
    print(device)

    train_data, test_data = load_mnist()
    vanilla_network = train_mnist_vanilla(train_data, device)
    torch.save(vanilla_network.state_dict(), 'vanilla_network.pt')
    accuracy = test_mnist_vanilla(vanilla_network, test_data, device)
    print(f'Accuracy: {accuracy}')

    return 0

if __name__ == '__main__':
    main()