from enum import Enum

import torch


class RegularizationMethod(Enum):
    NONE = 0
    L1 = 1
    L2 = 2
class VanillaNetLinear(torch.nn.Module):
    def __init__(self, regularization_method: RegularizationMethod = RegularizationMethod.NONE, alpha: float = 0.):
        super(VanillaNetLinear, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 10)

        self.regularization_method = regularization_method
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten the representation and propagate
        x = torch.nn.Flatten()(x)
        y = torch.nn.ReLU()(self.linear1(x))
        y = self.linear2(y)
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
