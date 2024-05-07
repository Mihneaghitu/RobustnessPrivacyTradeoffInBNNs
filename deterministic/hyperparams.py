from dataclasses import dataclass

from torch.nn import CrossEntropyLoss, Module


@dataclass
class Hyperparameters:
    num_epochs: int
    lr: float
    criterion: Module = CrossEntropyLoss()
    lr_decay_magnitude: float = 0.1 # what the value after all the training should be as a percentage of the initial lr
    batch_size: int = 64
    warmup_lr: float = 0.1
    run_dp: bool = False
    grad_norm_bound: float = -1
    dp_sigma: float = 1.0
    alpha: float = 0.75
    eps: float = 0.1
    eps_warmup_itrs: int = 2000
    alpha_warmup_itrs: int = 10000
    decay_epoch_start: int = 25
    warmup_itr_start: int = 3000
