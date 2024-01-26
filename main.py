from typing import Tuple

import torch
import torchvision
from torchvision import transforms

from dataset_utils import load_mnist
from deterministic.pipeline import test_mnist_vanilla, train_mnist_vanilla


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