import torch
from torch.nn.modules.module import Module


class VanillaNetLinear(torch.nn.Module):
    def __init__(self):
        super(VanillaNetLinear, self).__init__()
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

    def get_output_size(self) -> int:
        return self.linear[-1].out_features

    def get_input_size(self) -> int:
        return self.linear[0].in_features

    def update_param(self, param: torch.Tensor, grad: torch.Tensor, learning_rate: float) -> torch.Tensor:
        return param - learning_rate * grad

    def add_noise(self, param: torch.Tensor, noise: torch.Tensor, learning_rate: float) -> torch.Tensor:
        return param + learning_rate * noise

class RegularizedVanillaNetLinear(VanillaNetLinear):
    def __init__(self, alpha: float):
        super(RegularizedVanillaNetLinear, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten the representation
        y = torch.nn.Flatten()(x)
        y = self.linear(y)
        # add l2 regularization
        l2_reg = torch.tensor(0.)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        y += self.alpha * l2_reg

        return y
