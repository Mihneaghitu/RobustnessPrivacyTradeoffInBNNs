from dataclasses import dataclass

from torch.nn import Module


@dataclass
class HyperparamsHMC:
    num_epochs: int
    num_burnin_epochs: int
    step_size: float
    lf_steps: int
    criterion: Module
    batch_size: int
    num_chains: int = 2
    warmup_step_size: float = 0.1
    momentum_std: float = 1.0
    prior_mu: float = 0.0
    prior_std: float = 1.0
    ll_std: float = 1.0
    run_dp: bool = False
    grad_norm_bound: float = -1
    dp_sigma: float = 1.0
    # alpha and (1 - alpha) are the normal and, respectively, the robust learning rates
    alpha: float = 0.75
    # eps is the perturbation bounding box radius used for adversarial training
    eps: float = 0.1
