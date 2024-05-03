from dataclasses import dataclass

from torch.nn import Module


@dataclass
class Hyperparameters:
    num_epochs: int
    lr: float
    criterion: Module
    lr_decay_magnitude: float = 0.1 # what the value after all the training should be as a percentage of the initial lr
    batch_size: int = 64
    warmup_lr: float = 0.1
    run_dp: bool = False
    grad_norm_bound: float = -1
    dp_sigma: float = 1.0
    alpha: float = 0.75
    eps: float = 0.1
