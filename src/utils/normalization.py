from typing import Tuple

import torch


def norm_tensors(n_channels: int) -> Tuple[torch.tensor, torch.tensor]:
    if n_channels == 1:
        # MNIST mu and sigma
        return torch.tensor([0.1307]).view(1, 1, 1, 1), \
               torch.tensor([0.3081]).view(1, 1, 1, 1)
    else:
        return torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1), \
               torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)