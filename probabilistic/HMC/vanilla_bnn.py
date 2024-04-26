import copy
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from globals import TORCH_DEVICE


class VanillaBnnLinear(torch.nn.Module, ABC):
    def __init__(self):
        super(VanillaBnnLinear, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def ibp_forward(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        pass

    @abstractmethod
    def get_input_size(self) -> int:
        pass

    @abstractmethod
    def get_output_size(self) -> int:
        pass

    def hmc_forward(self, x: torch.Tensor, posterior_samples: List[torch.Tensor]) -> torch.Tensor:
        # We need to do a forward pass for each sample in the posterior
        # First dim is the batch size
        y_hat = torch.zeros(x.size(0), self.get_output_size()).to(TORCH_DEVICE)
        for sample in posterior_samples:
            self.set_params(sample)
            self.eval()
            y_hat += self.forward(x)

        y_hat = torch.div(y_hat, torch.tensor(len(posterior_samples)))

        return y_hat


    def get_worst_case_logits(self, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
        z_inf, z_sup = self.ibp_forward(x, eps)

        lower_bound_mask = torch.eye(10).to(TORCH_DEVICE)[y]
        upper_bound_mask = 1 - lower_bound_mask
        worst_case_logits = z_inf * lower_bound_mask + z_sup * upper_bound_mask

        return worst_case_logits

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

    def _get_bounds_affine(self, weights: torch.Tensor, bias: torch.Tensor, z_inf: torch.Tensor, z_sup: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        miu = (z_inf + z_sup) / 2
        std = (z_sup - z_inf) / 2
        weights = weights.T
        miu_new = miu @ weights + bias
        std_new = std @ torch.abs(weights)

        # new bounds after affine transformation
        return miu_new - std_new, miu_new + std_new

    def _get_bounds_monotonic(self, act: torch.nn.Module, z_inf: torch.Tensor, z_sup: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return act(z_inf), act(z_sup)

class VanillaBnnMnist(VanillaBnnLinear):
    def __init__(self):
        super(VanillaBnnMnist, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten the representation
        x_start = torch.nn.Flatten()(x)
        y = torch.nn.ReLU()(self.linear1(x_start))
        y = self.linear2(y)

        return y

    def ibp_forward(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        activation = torch.nn.ReLU()
        # flatten the representation
        x_start = torch.nn.Flatten()(x)
        z_inf, z_sup = x_start - eps, x_start + eps
        z_inf = torch.clamp(z_inf, 0, 1)
        z_sup = torch.clamp(z_sup, 0, 1)

        # first layer
        z_inf, z_sup = self._get_bounds_affine(self.linear1.weight, self.linear1.bias, z_inf, z_sup)
        z_inf, z_sup = self._get_bounds_monotonic(activation, z_inf, z_sup)

        # second layer -> logits
        z_inf, z_sup = self._get_bounds_affine(self.linear2.weight, self.linear2.bias, z_inf, z_sup)


        return z_inf, z_sup

    def get_input_size(self) -> int:
        return self.linear1.in_features

    def get_output_size(self) -> int:
        return self.linear2.out_features

class VanillaBnnFashionMnist(VanillaBnnLinear):
    def __init__(self):
        super(VanillaBnnFashionMnist, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten the representation
        x_start = torch.nn.Flatten()(x)
        y = torch.nn.ReLU()(self.linear1(x_start))
        y = self.linear2(y)

        return y

    def ibp_forward(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        activation = torch.nn.ReLU()
        # flatten the representation
        x_start = torch.nn.Flatten()(x)
        z_inf, z_sup = x_start - eps, x_start + eps
        z_inf = torch.clamp(z_inf, 0, 1)
        z_sup = torch.clamp(z_sup, 0, 1)

        # first layer
        z_inf, z_sup = self._get_bounds_affine(self.linear1.weight, self.linear1.bias, z_inf, z_sup)
        z_inf, z_sup = self._get_bounds_monotonic(activation, z_inf, z_sup)

        # second layer -> logits
        z_inf, z_sup = self._get_bounds_affine(self.linear2.weight, self.linear2.bias, z_inf, z_sup)

        return z_inf, z_sup

    def get_input_size(self) -> int:
        return self.linear1.in_features

    def get_output_size(self) -> int:
        return self.linear2.out_features

class IbpAdversarialLoss(torch.nn.Module):
    def __init__(self, net: VanillaBnnLinear, base_criterion: torch.nn.Module, eps: float):
        super(IbpAdversarialLoss, self).__init__()
        self.net = net
        self.eps = eps
        self.base_criterion = base_criterion

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        worst_case_logits = self.net.get_worst_case_logits(x, y, self.eps)
        l_robust = self.base_criterion(worst_case_logits, y)

        return l_robust
