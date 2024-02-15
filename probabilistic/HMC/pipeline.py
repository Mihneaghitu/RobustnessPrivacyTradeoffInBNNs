import copy
import sys

import torch
import torch.distributions as dist
import torchvision
from torch.utils.data import DataLoader

sys.path.append('../../')
from dataset_utils import load_mnist
from deterministic.vanilla_net import VanillaNetLinear
from globals import TORCH_DEVICE


def train_mnist_vanilla(train_data: torchvision.datasets.mnist, device: torch.device) -> VanillaNetLinear:
    vanilla_net = VanillaNetLinear()
    vanilla_net.to(device)

    # hyperparameters
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.001
    num_epochs = 5
    batch_size = 64
    print_freq = 500
    # make L = len(train_data) / batch_size
    L = len(train_data) // batch_size
    print(f'L: {L}')

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialize the parameters with a standard normal --
    # q -> current net params, current q -> start net params
    current_q = []
    for param in vanilla_net.named_parameters():
        if 'weight' in param[0]:
            init_vals = torch.normal(mean=0.0, std=0.1, size=tuple(param[1].shape)).to(device)
            current_q.append(copy.deepcopy(init_vals))
            param[1].data = torch.nn.parameter.Parameter(init_vals)

    running_loss_ce = 0.0
    for epoch in range(num_epochs):
        losses, p = [], []
        for param in vanilla_net.parameters():
            p.append(dist.Normal(0, 0.05).sample(param.shape).to(device))

        for i in range(L):
            # ------------------- START q + eps * p -------------------
            for idx, param in enumerate(vanilla_net.parameters()):
                new_val = vanilla_net.add_noise(param, p[idx], lr)
                with torch.no_grad():
                    param.copy_(new_val)
            # ------------------- END q + eps * p -------------------

            data = next(iter(data_loader))
            batch_data_train, batch_target_train = data[0].to(device), data[1].to(device)

            # ------------------- START p - eps * grad_U(q) -------------------
            if i == L - 1:
                break
            # Forward pass
            y_hat = vanilla_net(batch_data_train)
            closs = criterion(y_hat, batch_target_train)
            vanilla_net.zero_grad()
            closs.backward()

            prior_loss = torch.tensor(0.0).requires_grad_(True).to(device)
            for idx, param in enumerate(vanilla_net.parameters()):
                ll_grad = param.grad
                prior_loss = prior_loss + torch.neg(torch.mean(dist.Normal(0, 1).log_prob(param)))
                prior_grad = torch.autograd.grad(outputs=prior_loss, inputs=param)[0]
                potential_energy_update = ll_grad + prior_grad
                p[idx] = vanilla_net.update_param(p[idx], potential_energy_update, lr)

            vanilla_net.zero_grad()
            # ------------------- END p - eps * grad_U(q) -------------------

            losses.append(closs.item())
            running_loss_ce += closs.item()
            if i % print_freq == print_freq - 1:    # print every 500 mini-batches
                print(f'[epoch {epoch + 1}, batch {i + 1}] cross_entropy loss: {running_loss_ce / (batch_size * 500)}')
                running_loss_ce = 0.0

    return vanilla_net

def test_mnist_vanilla(vanilla_net: VanillaNetLinear, test_data: torchvision.datasets.mnist, device: torch.device):
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



is_cuda_available = torch.cuda.is_available()
device = torch.device('cuda:0' if is_cuda_available else 'cpu')
print(device)

train_data, test_data = load_mnist()
vanilla_network = train_mnist_vanilla(train_data, device)
torch.save(vanilla_network.state_dict(), 'vanilla_network_regularizer.pt')
accuracy = test_mnist_vanilla(vanilla_network, test_data, device)
print(f'Accuracy: {accuracy}')
