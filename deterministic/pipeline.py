import torch
import torchvision
from torch.utils.data import DataLoader

from deterministic.vanilla_net import VanillaCNN


def train_mnist_vanilla(train_data: torchvision.datasets.mnist, device: torch.device) -> VanillaCNN:
    vanilla_net = VanillaCNN(input_channels=1)
    vanilla_net.to(device)

    # hyperparameters
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.05
    num_epochs = 5
    batch_size = 20
    print_freq = 500

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialize the parameters with a standard normal --
    # TODO -- fix low accuracy when initializing with standard normal
    # for param in vanilla_net.parameters():
    #     init_vals = torch.normal(mean=0.0, std=1.0, size=tuple(param.shape)).to(device)
    #     param.data = torch.nn.parameter.Parameter(init_vals)

    running_loss = 0.0
    for epoch in range(num_epochs):
        losses  = []
        for i, data in enumerate(data_loader):
            batch_data_train, batch_target_train = data[0].to(device), data[1].to(device)
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

def test_mnist_vanilla(vanilla_net: VanillaCNN, test_data: torchvision.datasets.mnist, device: torch.device):
    vanilla_net.eval()
    batch_size = 32
    data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)
    losses, correct, total = [], 0, 0

    for data, target in data_loader:
        batch_data_test, batch_target_test = data.to(device), target.to(device)
        y_hat = vanilla_net(batch_data_test)
        loss = torch.nn.functional.cross_entropy(y_hat, batch_target_test)
        losses.append(loss.item())
        # also compute accuracy
        _, predicted = torch.max(y_hat, 1)
        total += batch_target_test.size(0)
        correct += (predicted == batch_target_test).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

    return 100 * correct / total