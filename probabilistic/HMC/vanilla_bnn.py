import copy
from typing import List, Tuple

import torch
import torch.distributions as dist

from globals import TORCH_DEVICE


class VanillaBnnLinear(torch.nn.Module):
    def __init__(self):
        super(VanillaBnnLinear, self).__init__()
        self.linear1 = torch.nn.Linear(784, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, 10)
        # torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten the representation
        x_start = torch.nn.Flatten()(x)
        y = torch.nn.ReLU()(self.linear1(x_start))
        y = torch.nn.ReLU()(self.linear2(y))
        y = self.linear3(y)

        return y

    def ibp_forward(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        activation = torch.nn.ReLU()
        # flatten the representation
        x_start = torch.nn.Flatten()(x)
        z_inf, z_sup = x_start - eps, x_start + eps
        z_inf = torch.clamp(z_inf, 0, 1)
        z_sup = torch.clamp(z_sup, 0, 1)

        # first layer
        z_inf, z_sup = self.__get_bounds_affine(self.linear1.weight, self.linear1.bias, z_inf, z_sup)
        z_inf, z_sup = self.__get_bounds_monotonic(activation, z_inf, z_sup)

        # second layer
        z_inf, z_sup = self.__get_bounds_affine(self.linear2.weight, self.linear2.bias, z_inf, z_sup)
        z_inf, z_sup = self.__get_bounds_monotonic(activation, z_inf, z_sup)

        # third layer -> logits
        z_inf, z_sup = self.__get_bounds_affine(self.linear3.weight, self.linear3.bias, z_inf, z_sup)

        return z_inf, z_sup

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

    def get_input_size(self) -> int:
        return self.linear1.in_features

    def get_output_size(self) -> int:
        return self.linear3.out_features

    # ------------------- Private methods -------------------
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
    def __init__(self, net: VanillaBnnLinear, base_criterion: torch.nn.Module, eps: float):
        super(IbpAdversarialLoss, self).__init__()
        self.net = net
        self.eps = eps
        self.base_criterion = base_criterion

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z_inf, z_sup = self.net.ibp_forward(x, self.eps)

        #! inference: if z_inf > z_sup of all the rest provably robust
        lower_bound_mask = torch.eye(10).to(TORCH_DEVICE)[y]
        upper_bound_mask = 1 - lower_bound_mask
        worst_case_logits = z_inf * lower_bound_mask + z_sup * upper_bound_mask

        l_robust = self.base_criterion(worst_case_logits, y)

        return l_robust
