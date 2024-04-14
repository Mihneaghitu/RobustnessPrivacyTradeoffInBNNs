import os
from typing import Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


def load_mnist(rel_path: str = ".") -> Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    curr_dir = os.getcwd()
    os.chdir(curr_dir + "/" + rel_path)
    print(f"Current directory: {curr_dir}")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    return train_data, test_data

def get_marginal_distributions(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    num_features = 1
    # [0][0] input, [0][1] target
    for dim in dataset[0][0].shape:
        num_features *= dim

    values_per_feature = torch.zeros(num_features, len(dataset), dtype=torch.float32)

    for i, (elem, _) in enumerate(dataset):
        values_per_feature[:, i] = elem.flatten()

    marginal_means = torch.mean(values_per_feature, dim=1)
    marginal_stds = torch.std(values_per_feature, dim=1)

    # For now, assume that the data follows a normal distribution
    return marginal_means, marginal_stds

# -------------------- Not used for now --------------------
def __to_discrete_distribution(nums: np.ndarray, bins: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    # split values into bins
    intervals = np.linspace(nums.min(), nums.max(), bins)
    # get the index of the interval each value belongs to
    idx = np.digitize(nums, intervals)
    # get the count of each interval
    counts = np.bincount(idx)
    # normalize the counts
    counts = counts / counts.sum()
    # get the center of each interval
    ordered_domain_values = (intervals[1:] + intervals[:-1]) / 2

    # return the ordered domain values and the normalized counts
    return ordered_domain_values, counts[1:-1]
# -------------------- Not used for now --------------------
