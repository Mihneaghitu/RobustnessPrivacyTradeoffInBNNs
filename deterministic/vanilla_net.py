import sys

sys.path.append('../')

from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Union

import torch
import torch.nn.functional as F

from globals import TORCH_DEVICE


class RegularizationMethod(Enum):
    NONE = 0
    L1 = 1
    L2 = 2

class VanillaNetLinear(torch.nn.Module, ABC):
    def __init__(self, num_classes: int):
        super(VanillaNetLinear, self).__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def ibp_forward(self, x: torch.Tensor, epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def get_input_size(self) -> Union[int, Tuple[int]]:
        pass

    @abstractmethod
    def get_output_size(self) -> int:
        pass

    @abstractmethod
    def get_worst_case_logits(self, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
        pass

    def update_param(self, param: torch.Tensor, grad: torch.Tensor, learning_rate: float) -> torch.Tensor:
        return param - learning_rate * grad

    def add_noise(self, param: torch.Tensor, noise: torch.Tensor, learning_rate: float) -> torch.Tensor:
        return param + learning_rate * noise

    def _get_bounds_affine(self, weights: torch.Tensor, bias: torch.Tensor, z_inf: torch.Tensor, z_sup: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = (z_inf + z_sup) / 2
        std = (z_sup - z_inf) / 2
        weights = weights.T
        mu_new = mu @ weights + bias
        std_new = std @ torch.abs(weights)

        # new bounds after affine transformation
        return mu_new - std_new, mu_new + std_new

    def _get_bounds_monotonic(self, act: torch.nn.Module, z_inf: torch.Tensor, z_sup: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return act(z_inf), act(z_sup)

    def _get_bounds_conv2d(self, weights: torch.Tensor, bias: torch.Tensor, z_inf: torch.Tensor, z_sup: torch.Tensor, stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = (z_inf + z_sup) / 2
        std = (z_sup - z_inf) / 2
        mu_new = F.conv2d(mu, weights, bias, stride=stride)
        std_new = F.conv2d(std, torch.abs(weights), stride=stride)

        lower_bound, upper_bound = mu_new - std_new, mu_new + std_new
        # new bounds after affine transformation
        return lower_bound, upper_bound

    def get_num_classes(self) -> int:
        return self.num_classes

class VanillaNetMnist(VanillaNetLinear):
    def __init__(self, regularization_method: RegularizationMethod = RegularizationMethod.NONE, alpha: float = 0.):
        super(VanillaNetMnist, self).__init__(10) # 10 classes
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
        z_inf, z_sup = self._get_bounds_affine(self.linear1.weight, self.linear1.bias, z_inf, z_sup)
        z_inf, z_sup = self._get_bounds_monotonic(torch.nn.ReLU(), z_inf, z_sup)

        # second layer -> logits
        z_inf, z_sup = self._get_bounds_affine(self.linear2.weight, self.linear2.bias, z_inf, z_sup)

        return z_inf, z_sup

    def get_output_size(self) -> int:
        return self.linear2.out_features

    def get_input_size(self) -> int:
        return self.linear1.in_features

    def get_worst_case_logits(self, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
        z_inf, z_sup = self.ibp_forward(x, eps)
        lower_bound_mask = torch.eye(self.num_classes).to(TORCH_DEVICE)[y]
        upper_bound_mask = 1 - lower_bound_mask
        worst_case_logits = z_inf * lower_bound_mask + z_sup * upper_bound_mask

        return worst_case_logits


class ConvNetPneumoniaMnist(VanillaNetLinear):
    def __init__(self):
        super(ConvNetPneumoniaMnist, self).__init__(1) # binary classification
        self.in_channels = 1
        self.latent_dim = 4800
        self.conv1 = torch.nn.Conv2d(self.in_channels, 16, kernel_size=4, stride=2) # dim = 16 x 13 x 13
        self.conv2 = torch.nn.Conv2d(16, 48, kernel_size=4, stride=1) # dim = 48 x 10 x 10
        self.linear1 = torch.nn.Linear(self.latent_dim, 100)
        self.linear2 = torch.nn.Linear(100, 1) # binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.ReLU()(self.conv1(x))
        y = torch.nn.ReLU()(self.conv2(y))
        y = torch.nn.Flatten()(y)
        y = torch.nn.ReLU()(self.linear1(y))
        y = self.linear2(y)

        return y

    def ibp_forward(self, x: torch.Tensor, epsilon: float) -> Tuple[torch.Tensor]:
        z_inf, z_sup = torch.clamp(x - epsilon, 0, 1), torch.clamp(x + epsilon, 0, 1)

        # first conv layer
        z_inf, z_sup = self._get_bounds_conv2d(self.conv1.weight, self.conv1.bias, z_inf, z_sup, stride=2)
        z_inf, z_sup = self._get_bounds_monotonic(torch.nn.ReLU(), z_inf, z_sup)

        # second conv layer
        z_inf, z_sup = self._get_bounds_conv2d(self.conv2.weight, self.conv2.bias, z_inf, z_sup, stride=1)
        z_inf, z_sup = self._get_bounds_monotonic(torch.nn.ReLU(), z_inf, z_sup)

        # flatten the representation
        z_inf, z_sup = torch.nn.Flatten()(z_inf), torch.nn.Flatten()(z_sup)

        # first fully connected layer
        z_inf, z_sup = self._get_bounds_affine(self.linear1.weight, self.linear1.bias, z_inf, z_sup)
        z_inf, z_sup = self._get_bounds_monotonic(torch.nn.ReLU(), z_inf, z_sup)

        # second fully connected layer -> logits
        z_inf, z_sup = self._get_bounds_affine(self.linear2.weight, self.linear2.bias, z_inf, z_sup)

        return z_inf, z_sup

    def get_output_size(self) -> int:
        return self.linear2.out_features

    def get_input_size(self) -> Tuple[int]:
        return (self.in_channels, 28, 28)

    def get_worst_case_logits(self, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
        z_inf, z_sup = self.ibp_forward(x, eps)

        # Because it is binary classification, and thus the output is a single neuron, to worst case logit is the lower bound
        # when the target is 1, and the upper bound when the target is 0
        return torch.where(y == 1, z_inf, z_sup)

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
