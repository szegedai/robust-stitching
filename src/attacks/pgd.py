from typing import Callable
from src.attacks import Attack

import torch


class PGD(Attack):

    def __init__(self,
                 eps: float = 0.3,
                 alpha: float = 0.1,
                 n_steps: int = 10,
                 random_start: bool = False,
                 targeted: bool = False,
                 loss_fn: "torch.nn.Module" = ...,
                 target_transform_fn: Callable = None,
                 min: float = 0.0,
                 max: float = 1.0,
                 attack_ratio: float = 1.0
    ) -> None:
        super().__init__(loss_fn, target_transform_fn, min, max, attack_ratio)
        self.eps = eps
        self.alpha = alpha
        self.n_steps = n_steps
        self.random_start = random_start
        self.targeted = targeted


    def _compute_adv_samples(self,
                             model: torch.nn.Module,
                             inputs: torch.Tensor,
                             targets: torch.Tensor
    ) -> torch.Tensor:

        adv_inputs = inputs.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_inputs = adv_inputs + torch.empty_like(adv_inputs) \
                                           .uniform_(-self.eps, self.eps)
            adv_inputs = torch.clamp(adv_inputs, min=0, max=1).detach()

        for _ in range(self.n_steps):
            adv_inputs.requires_grad = True
            outputs = model(adv_inputs)

            # Get loss and cost
            cost = None
            if self.targeted:
                output_targets = self.target_transform(outputs)
                cost = -self.loss_fn(targets, output_targets)
            else:
                cost = self.loss_fn(outputs, targets)

            # Update adversarial images
            grad = torch.autograd.grad(cost,
                                       adv_inputs,
                                       retain_graph=False,
                                       create_graph=False)[0]

            adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_inputs-inputs, min=-self.eps, max=self.eps)
            adv_inputs = torch.clamp(inputs + delta, min=0, max=1).detach()

        return adv_inputs
