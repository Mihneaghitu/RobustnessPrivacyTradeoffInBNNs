import numpy as np


def to_discrete_distribution(nums: np.ndarray) -> (np.ndarray, np.ndarray):
    # Make dictionary mapping a number to its probability
    weight_dict = {}
    for _, w in enumerate(nums):
        # This should be illegal
        w_repr = str(w)
        if w_repr in weight_dict:
            weight_dict[w_repr] += 1
        else:
            weight_dict[w_repr] = 1

    ordered_weight_dict = dict(sorted(weight_dict.items(), key=lambda x: x[0]))

    domain_values = np.array(list(ordered_weight_dict.keys())).astype(float)
    range_values = np.array(list(ordered_weight_dict.values()))
    range_values = range_values / range_values.sum()

    return domain_values, range_values