import torch
from vit_pytorch import ViT, SimpleViT

from src.models.classifiers.general import GeneralNet


class VisionTransformer(GeneralNet):
    """Adapter for the Vision Transformer implementation of the vit-pytorch 
    package: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, 
                 n_classes: int, 
                 n_channels_in: int, 
                 image_size: int = 32,
                 patch_size: int = 4,
                 dim: int = 1024,
                 depth: int = 6,
                 heads: int = 16,
                 mlp_dim: int = 2048,
                 model_path: str = None, 
                 seed: int = None
    ) -> None:
        super().__init__(n_classes, n_channels_in, model_path, seed)
        self.model = SimpleViT(image_size=image_size,
                               patch_size=patch_size,
                               num_classes=n_classes,
                               dim=dim,
                               depth=depth,
                               heads=heads,
                               mlp_dim=mlp_dim,
                               channels=n_channels_in)

        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def name(self):
        ''' Name of the model '''
        return f"vit_{self.patch_size}_{self.dim}_{self.depth}_{self.heads}_{self.mlp_dim}"

