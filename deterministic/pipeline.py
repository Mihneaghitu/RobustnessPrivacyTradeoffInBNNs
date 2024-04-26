import sys

sys.path.append('../')

import torch
import torchvision
from torch.utils.data import DataLoader

from dataset_utils import load_mnist
from deterministic.vanilla_net import VanillaNetLinear
from globals import TORCH_DEVICE


def train_mnist_vanilla(train_data: torchvision.datasets.mnist) -> VanillaNetLinear:
    vanilla_net = VanillaNetLinear()
    vanilla_net.to(TORCH_DEVICE)

    # hyperparameters
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.002
    num_epochs = 50
    batch_size = 64
    print_freq = 500

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialize the parameters with a standard normal --
    for param in vanilla_net.named_parameters():
        if 'weight' in param[0]:
            init_vals = torch.normal(mean=0.0, std=0.1, size=tuple(param[1].shape)).to(TORCH_DEVICE)
            param[1].data = torch.nn.parameter.Parameter(init_vals)

    running_loss = 0.0
    for epoch in range(num_epochs):
        losses  = []
        lr = max(lr * 0.95, 0.001)
        for i, data in enumerate(data_loader):
            batch_data_train, batch_target_train = data[0].to(TORCH_DEVICE), data[1].to(TORCH_DEVICE)
            # Forward pass
            y_hat = vanilla_net(batch_data_train)
            loss = criterion(y_hat, batch_target_train)
            vanilla_net.zero_grad()
            loss.backward()
            # Update the parameters using gradient descent
            with torch.no_grad():
                for param in vanilla_net.parameters():
                    new_val = vanilla_net.update_param(param, param.grad, lr)
                    param.copy_(new_val)

            losses.append(loss.item())
            running_loss += loss.item()
            if i % print_freq == print_freq - 1:    # print every 500 mini-batches
                print(f'[epoch {epoch + 1}, batch {i + 1}] loss: {running_loss / (batch_size * 500)}')
                running_loss = 0.0

    return vanilla_net

def test_mnist_vanilla(vanilla_net: VanillaNetLinear, test_data: torchvision.datasets.mnist):
    vanilla_net.eval()
    batch_size = 32
    data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)
    losses, correct, total = [], 0, 0

    for data, target in data_loader:
        batch_data_test, batch_target_test = data.to(TORCH_DEVICE), target.to(TORCH_DEVICE)
        y_hat = vanilla_net(batch_data_test)
        loss = torch.nn.functional.cross_entropy(y_hat, batch_target_test)
        losses.append(loss.item())
        # also compute accuracy
        _, predicted = torch.max(y_hat, 1)
        total += batch_target_test.size(0)
        correct += (predicted == batch_target_test).sum().item()

    return 100 * correct / total

def run_pipeline(test_only: bool = False):
    net = None
    curr_dir = __file__.rsplit('/', 2)[0]
    if not test_only:
        train, test = load_mnist(relative_path='../')
        net = train_mnist_vanilla(train)
        torch.save(net.state_dict(), curr_dir + '/vanilla_network.pt')
    else:
        net = VanillaNetLinear()
        net.load_state_dict(torch.load(curr_dir + 'vanilla_network.pt'))

    acc = test_mnist_vanilla(net, test)
    print(f'Accuracy of the network on the 10000 test images: {acc}%')
