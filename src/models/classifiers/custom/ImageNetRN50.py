import torch
from torchvision.models import resnet50
from robustbench.model_zoo.architectures.utils_architectures import normalize_model

from src.models.classifiers import GeneralNet


IMAGENET_MU = (0.485, 0.456, 0.406)
IMAGENET_SIGMA = (0.229, 0.224, 0.225)


class ImageNetRN50(GeneralNet):
    def __init__(self, 
                 n_classes: int = 1000, 
                 n_channels_in: int = 3, 
                 model_path: str = None, 
                 seed: int = None
    ) -> None:
        super().__init__(n_classes, 
                         n_channels_in, 
                         model_path, 
                         seed)

        self.model = normalize_model(resnet50(), IMAGENET_MU, IMAGENET_SIGMA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
