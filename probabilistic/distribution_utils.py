import numpy as np


def to_discrete_distribution(nums: np.ndarray, bins: int = 150) -> (np.ndarray, np.ndarray):
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