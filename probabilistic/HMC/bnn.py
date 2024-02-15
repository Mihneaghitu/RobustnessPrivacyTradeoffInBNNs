import sys
from abc import ABC, abstractmethod

import torch

sys.path.append('../../')
sys.path.append('../')

from globals import TORCH_DEVICE


class VanillaBNN(torch.nn.Module):
    def __init__(self):
        super(VanillaBNN, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )

        # calculate the number of weights and biases
        self.total_num_weights = 0
        self.total_num_biases = 0
        for param in self.named_parameters():
            if 'weight' in param[0]:
                self.total_num_weights += torch.prod(torch.tensor(param[1].shape))
            if 'bias' in param[0]:
                self.total_num_biases += torch.prod(torch.tensor(param[1].shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten the representation
        y = torch.nn.Flatten()(x)
        y = self.linear(y)

        return y

    def get_num_weights(self) -> int:
        return self.total_num_weights

    def get_num_params(self) -> int:
        return self.total_num_weights + self.total_num_biases

    # ---- for HMC ----

    def get_weights(self):
        params = torch.tensor([]).to(TORCH_DEVICE)
        for param in self.named_parameters():
            if 'weight' in param[0]:
                param_data = param[1].detach().clone().flatten()
                params = torch.cat((params, param_data))

        return params

    def set_weights(self, params_sample):
        for param in self.named_parameters():
            if 'weight' in param[0]:
                num_weights_layer = torch.prod(torch.tensor(param[1].shape))
                param[1].data = params_sample[:num_weights_layer].reshape(param[1].shape).detach().clone()
                params_sample = params_sample[num_weights_layer:]

    def set_zero_grads(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def get_weight_grads(self):
        params_grads = torch.tensor([]).to(TORCH_DEVICE)
        for param in self.named_parameters():
            if 'weight' in param[0]:
                param_grad_data = param[1].grad.detach().clone().flatten()
                params_grads = torch.cat((params_grads, param_grad_data))

        return params_grads