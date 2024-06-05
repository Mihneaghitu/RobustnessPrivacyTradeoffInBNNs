import sys
from typing import Callable, Tuple

import torch

sys.path.append('../../')

from deterministic.membership_inference_dnn import (MembershipInferenceAttack,
                                                    RecordSynthesizer)
from probabilistic.HMC.vanilla_bnn import VanillaBnnLinear


class MembershipInferenceAttackBnn(MembershipInferenceAttack):
    def __init__(self, target_net: VanillaBnnLinear, num_classes: int, distrib_moments: Tuple[torch.Tensor, torch.Tensor],
                 posterior_samples: torch.Tensor) -> None:
        super().__init__(target_net, num_classes, distrib_moments)
        # Partial function so we keep the forward function uniform (i.e. takes the input tensor and returns the prediction)
        self.fwd_func = lambda x: target_net.hmc_forward(x, posterior_samples)
        self.record_synthesizer = RecordSynthesizerBnn(target_net, num_classes, distrib_moments, self.fwd_func, posterior_samples)
        self.posterior_samples = posterior_samples

    #! FOR THE test_attack_models() function:
    # I'm not sure how it should behave. On one hand, it would be simple to just use the bnn forward function and leave the rest as is.
    # On the other hand, the first DSET_SIZE / (batch_size * lf_steps) - 1 samples do not see the whole training dataset.

class RecordSynthesizerBnn(RecordSynthesizer):
    def __init__(self, target_net: VanillaBnnLinear, num_classes: int, distrib_moments: Tuple[torch.Tensor, torch.Tensor],
                 fwd_func: Callable[[torch.Tensor], torch.Tensor], posterior_samples: torch.Tensor, pos_training_samples: int = 5000) -> None:
        super().__init__(target_net, num_classes, distrib_moments, fwd_func, pos_training_samples)
        self.posterior_samples = posterior_samples
