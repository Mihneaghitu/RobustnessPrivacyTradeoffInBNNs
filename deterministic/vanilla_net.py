import torch
import torch.nn.functional as F


class VanillaCNN(torch.nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        # num_c1_channels = 6
        # self.conv1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(input_channels, num_c1_channels, kernel_size=5), # 28x28x1 -> 24x24x6
        #     torch.nn.MaxPool2d(2), # 24x24x6 -> 12x12x6
        #     torch.nn.ReLU()
        # )
        # num_c2_channels = 16
        # self.conv2 = torch.nn.Sequential(
        #     torch.nn.Conv2d(num_c1_channels, num_c2_channels, kernel_size=3), # 12x12x6 -> 10x10x16
        #     torch.nn.MaxPool2d(2), # 10x10x16 -> 5x5x16
        #     torch.nn.ReLU()
        # )
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
        # y = self.conv1(y)
        # y = self.conv2(y)
        y = self.linear(y)

        return y

    def update_param(self, param: torch.Tensor, grad: torch.Tensor, learning_rate: float) -> torch.Tensor:
        return param - learning_rate * grad