from dataclasses import dataclass

from torch.nn import CrossEntropyLoss, Module


@dataclass
class HyperparamsHMC:
    num_epochs: int
    num_burnin_epochs: int
    step_size: float
    lf_steps: int
    batch_size: int
    criterion: Module = CrossEntropyLoss(reduction='mean')
    num_chains: int = 1
    decay_epoch_start: int = 25
    lr_decay_magnitude: float = 0.1
    warmup_step_size: float = 0.1
    momentum_std: float = 1.0
    prior_mu: float = 0.0
    prior_std: float = 1.0
    alpha_warmup_epochs: int = 0 #! needs to be less than the burnin epochs
    eps_warmup_epochs: int = 0 #! needs to be less than the burnin epochs
    # ------------ Adversarial Training Params ------------
    alpha: float = 0.75 # trade-off between the two objectives
    eps: float = 0.1 # perturbation radius
    # ------------ DP Params ------------
    run_dp: bool = False
    grad_clip_bound: float = -1.0 # b_g in the paper
    acceptance_clip_bound: float = -1.0 # b_l in the paper
    tau_g: float = -1.0
    tau_l: float = -1.0
