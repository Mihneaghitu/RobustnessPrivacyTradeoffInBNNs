import copy
from typing import List

import torch


class VanillaBnnLinear(torch.nn.Module):
    def __init__(self):
        super(VanillaBnnLinear, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            # torch.nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten the representation
        y = torch.nn.Flatten()(x)
        y = self.linear(y)

        return y

    def update_param(self, param: torch.Tensor, grad: torch.Tensor, learning_rate: float) -> torch.Tensor:
        return param - learning_rate * grad

    def add_noise(self, param: torch.Tensor, noise: torch.Tensor, learning_rate: float) -> torch.Tensor:
        return param + learning_rate * noise

    def set_params(self, new_params_values: List[torch.Tensor]) -> None:
        for idx, param in enumerate(self.parameters()):
            with torch.no_grad():
                param.copy_(new_params_values[idx])

    def get_params(self) -> List[torch.Tensor]:
        return [copy.deepcopy(param) for param in self.parameters()]
