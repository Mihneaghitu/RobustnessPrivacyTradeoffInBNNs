import sys

sys.path.append('../')

from enum import Enum
from typing import Tuple

import torch

from globals import TORCH_DEVICE


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

    def ibp_forward(self, x: torch.Tensor, epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # flatten the representation and propagate
        x = torch.nn.Flatten()(x)
        z_inf, z_sup = torch.clamp(x - epsilon, 0, 1), torch.clamp(x + epsilon, 0, 1)

        # first layer
        z_inf, z_sup = self.__get_bounds_affine(self.linear1.weight, self.linear1.bias, z_inf, z_sup)
        z_inf, z_sup = self.__get_bounds_monotonic(torch.nn.ReLU(), z_inf, z_sup)

        # second layer -> logits
        z_inf, z_sup = self.__get_bounds_affine(self.linear2.weight, self.linear2.bias, z_inf, z_sup)

        return z_inf, z_sup

    def get_worst_case_logits(self, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
        z_inf, z_sup = self.ibp_forward(x, eps)

        lower_bound_mask = torch.eye(10).to(TORCH_DEVICE)[y]
        upper_bound_mask = 1 - lower_bound_mask
        worst_case_logits = z_inf * lower_bound_mask + z_sup * upper_bound_mask

        return worst_case_logits

    def get_output_size(self) -> int:
        return self.linear2.out_features

    def get_input_size(self) -> int:
        return self.linear1.in_features

    def update_param(self, param: torch.Tensor, grad: torch.Tensor, learning_rate: float) -> torch.Tensor:
        return param - learning_rate * grad

    def add_noise(self, param: torch.Tensor, noise: torch.Tensor, learning_rate: float) -> torch.Tensor:
        return param + learning_rate * noise

    def __get_bounds_affine(self, weights: torch.Tensor, bias: torch.Tensor, z_inf: torch.Tensor, z_sup: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        miu = (z_inf + z_sup) / 2
        std = (z_sup - z_inf) / 2
        weights = weights.T
        miu_new = miu @ weights + bias
        std_new = std @ torch.abs(weights)

        # new bounds after affine transformation
        return miu_new - std_new, miu_new + std_new

    def __get_bounds_monotonic(self, act: torch.nn.Module, z_inf: torch.Tensor, z_sup: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return act(z_inf), act(z_sup)

class IbpAdversarialLoss(torch.nn.Module):
    def __init__(self, net: VanillaNetLinear, base_criterion: torch.nn.Module, eps: float):
        super(IbpAdversarialLoss, self).__init__()
        self.net = net
        self.eps = eps
        self.base_criterion = base_criterion

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        worst_case_logits = self.net.get_worst_case_logits(x, y, self.eps)
        l_robust = self.base_criterion(worst_case_logits, y)

        return l_robust
