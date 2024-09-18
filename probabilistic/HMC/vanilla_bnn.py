import copy
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from globals import TORCH_DEVICE


class VanillaBnnLinear(torch.nn.Module, ABC):
    def __init__(self, num_classes: int):
        super(VanillaBnnLinear, self).__init__()
        self.num_classes = num_classes

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

    def hmc_forward(self, x: torch.Tensor, posterior_samples: List[torch.Tensor], last_layer_act: callable = None) -> torch.Tensor:
        # We need to do a forward pass for each sample in the posterior
        # First dim is the batch size
        y_hat = torch.zeros(x.size(0), self.get_output_size()).to(TORCH_DEVICE)
        for sample in posterior_samples:
            self.set_params(sample)
            self.eval()
            if last_layer_act is not None:
                y_hat += last_layer_act(self.forward(x))
            else:
                y_hat += self.forward(x)
        y_hat = y_hat / len(posterior_samples)

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

    def get_num_classes(self) -> int:
        return self.num_classes

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

    def _get_bounds_conv2d(self, weights: torch.Tensor, bias: torch.Tensor, z_inf: torch.Tensor, z_sup: torch.Tensor, stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
        miu = (z_inf + z_sup) / 2
        std = (z_sup - z_inf) / 2
        miu_new = torch.nn.functional.conv2d(miu, weights, bias, stride=stride)
        std_new = torch.nn.functional.conv2d(std, torch.abs(weights), stride=stride)

        lower_bound, upper_bound = miu_new - std_new, miu_new + std_new

        return lower_bound, upper_bound


class VanillaBnnMnist(VanillaBnnLinear):
    def __init__(self, layer_sizes: List[int] = None):
        super(VanillaBnnMnist, self).__init__(10) # 10 classes
        # apparently when you have a mutable default argument, it can be modified with each call to the function
        if layer_sizes is None:
            layer_sizes = [512]
        self.linears, prev_size = torch.nn.ModuleList(), 784

        # create hidden layers
        for layer_size in layer_sizes:
            self.linears.append(torch.nn.Linear(prev_size, int(layer_size)))
            prev_size = int(layer_size)
        # create output layer
        self.linears.append(torch.nn.Linear(int(layer_sizes[-1]), 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten the representation
        x_curr = torch.nn.Flatten()(x)
        for layer in self.linears[:-1]:
            x_curr = torch.nn.ReLU()(layer(x_curr))
        y = self.linears[-1](x_curr)

        return y

    def ibp_forward(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        activation = torch.nn.ReLU()
        # flatten the representation
        x_start = torch.nn.Flatten()(x)
        z_inf, z_sup = x_start - eps, x_start + eps
        z_inf = torch.clamp(z_inf, 0, 1)
        z_sup = torch.clamp(z_sup, 0, 1)

        # hidden layers
        for layer in self.linears[:-1]:
            z_inf, z_sup = self._get_bounds_affine(layer.weight, layer.bias, z_inf, z_sup)
            z_inf, z_sup = self._get_bounds_monotonic(activation, z_inf, z_sup)

        # output layer
        z_inf, z_sup = self._get_bounds_affine(self.linears[-1].weight, self.linears[-1].bias, z_inf, z_sup)


        return z_inf, z_sup

    def get_input_size(self) -> int:
        return self.linears[0].in_features

    def get_output_size(self) -> int:
        return self.linears[-1].out_features

class VanillaBnnFashionMnist(VanillaBnnLinear):
    def __init__(self):
        super(VanillaBnnFashionMnist, self).__init__(10) # 10 classes
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

class ConvBnnPneumoniaMnist(VanillaBnnLinear):
    def __init__(self, dim_ratio: int = 1):
        super(ConvBnnPneumoniaMnist, self).__init__(1) # binary classification
        self.in_channels = 1
        self.latent_dim = int(4800 * dim_ratio)
        self.conv1 = torch.nn.Conv2d(self.in_channels, int(16 * dim_ratio), kernel_size=4, stride=2)
        self.conv2 = torch.nn.Conv2d(int(16 * dim_ratio), int(48 * dim_ratio), kernel_size=4, stride=1)
        self.linear1 = torch.nn.Linear(self.latent_dim, 100)
        self.linear2 = torch.nn.Linear(100, 1) # binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.ReLU()(self.conv1(x))
        y = torch.nn.ReLU()(self.conv2(y))
        y = torch.nn.Flatten()(y)
        y = torch.nn.ReLU()(self.linear1(y))
        y = self.linear2(y)

        return y

    def ibp_forward(self, x: torch.Tensor, eps: float) -> Tuple[torch.Tensor]:
        z_inf, z_sup = torch.clamp(x - eps, 0, 1), torch.clamp(x + eps, 0, 1)

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
    def __init__(self, net: VanillaBnnLinear, base_criterion: torch.nn.Module, eps: float):
        super(IbpAdversarialLoss, self).__init__()
        self.net = net
        self.eps = eps
        self.base_criterion = base_criterion

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        worst_case_logits = self.net.get_worst_case_logits(x, y, self.eps)
        l_robust = self.base_criterion(worst_case_logits, y)

        return l_robust
