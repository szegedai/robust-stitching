from typing import Callable
from src.attacks import Attack

import torch


class FGSM(Attack):

    def __init__(self,
                 eps: float = 0.3,
                 targeted: bool = False,
                 loss_fn: torch.nn.Module = ...,
                 target_transform_fn: Callable = None
    ) -> None:
        super().__init__(loss_fn, target_transform_fn)
        self.eps = eps
        self.targeted = targeted

    def _compute_adv_samples(self, 
                             model: torch.nn.Module, 
                             inputs: torch.Tensor, 
                             targets: torch.Tensor
    ) -> torch.Tensor:
        # Get outputs
        inputs.requires_grad = True
        outputs = model(inputs)

        # Get loss and cost
        cost = None
        if self.targeted:
            output_targets = self.target_transform(outputs)
            cost = -self.loss_fn(targets, output_targets)
        else:
            cost = self.loss_fn(targets, outputs)

        # Get gradient and create adversarial inputs
        grad = torch.autograd.grad(cost,
                                   inputs,
                                   retain_graph=False,
                                   create_graph=False)[0]

        adv_inputs = inputs + self.eps * grad.sign()
        adv_inputs = torch.clamp(adv_inputs, self.min, self.max).detach()

        return adv_inputs
