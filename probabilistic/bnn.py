from abc import ABC, abstractmethod

import torch


class BNN(ABC):
    @abstractmethod
    def init_weights(self, sample: torch.Tensor):
        pass

    def get_num_weights(self) -> int:
        pass

class VanillaBNN(BNN, torch.nn.Module):
    def __init__(self):
        super(VanillaBNN, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            torch.nn.Softmax(dim=1)
        )

        # calculate the number of weights
        self.total_num_weights = 0
        for param in self.named_parameters():
            if 'weight' in param[0]:
                self.total_num_weights += torch.prod(torch.tensor(param[1].shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten the representation
        y = torch.nn.Flatten()(x)
        y = self.linear(y)

        return y

    def get_num_weights(self) -> int:
        return self.total_num_weights

    def init_weights(self, sample: torch.Tensor):
        curr_idx = 0
        for param in self.named_parameters():
            if 'weight' in param[0]:
                num_weights_layer = torch.prod(torch.tensor(param[1].shape))
                param[1].data = sample[curr_idx : curr_idx + num_weights_layer].reshape(param[1].shape)
                curr_idx += num_weights_layer