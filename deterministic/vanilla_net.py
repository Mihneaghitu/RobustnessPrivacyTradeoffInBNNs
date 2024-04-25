from enum import Enum

import torch


class RegularizationMethod(Enum):
    NONE = 0
    L1 = 1
    L2 = 2
class VanillaNetLinear(torch.nn.Module):
    def __init__(self, regularization_method: RegularizationMethod = RegularizationMethod.NONE, alpha: float = 0.):
        super(VanillaNetLinear, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            # torch.nn.Softmax(dim=1)
        )
        self.regularization_method = regularization_method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten the representation
        y = torch.nn.Flatten()(x)
        y = self.linear(y)
        # add regularization
        reg = torch.tensor(0.)
        match self.regularization_method:
            case RegularizationMethod.L1:
                for param in self.parameters():
                    reg += torch.norm(param, 1)
            case RegularizationMethod.L2:
                for param in self.parameters():
                    reg += torch.norm(param, 2)
            case RegularizationMethod.NONE:
                pass

        y += self.alpha * reg

        return y

    def get_output_size(self) -> int:
        return self.linear[-1].out_features

    def get_input_size(self) -> int:
        return self.linear[0].in_features

    def update_param(self, param: torch.Tensor, grad: torch.Tensor, learning_rate: float) -> torch.Tensor:
        return param - learning_rate * grad

    def add_noise(self, param: torch.Tensor, noise: torch.Tensor, learning_rate: float) -> torch.Tensor:
        return param + learning_rate * noise
