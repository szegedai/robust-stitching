from typing import Any, List, Type, Union

import torch
from src.models.classifiers.resnet import BasicBlock, Bottleneck, ResNet


class EngstromRN50(ResNet):
    def __init__(self, 
                 n_classes: int = 10, 
                 n_channels: int = 3, 
                 block: Type[Union[BasicBlock, Bottleneck]] = Bottleneck, 
                 layers: List[int] = [3, 4, 6, 3], 
                 name: str = 'EngstromRN50',
                 **kwargs: Any
    ) -> None:
        super().__init__(n_classes, 
                         n_channels, 
                         block, 
                         layers, 
                         name, 
                         **kwargs)
        mu = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        sigma = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mu) / self.sigma
        return self._forward_impl(x)
