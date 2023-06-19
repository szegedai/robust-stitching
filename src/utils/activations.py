from typing import Callable, List, Tuple
import torch
from torch import nn
from src.dataset.utils import _get_data_loader
from src.utils import config


def _activation_range_hook(min_holder: List[float], 
                           max_holder: List[float]
) -> Callable:
    """Forward hook that stores the minimum and maximum activations.

    Args:
        min_holder (list): list holding the minimum activations
        max_holder (list): list holding the maximum activations
    
    Returns:
        callable: the forward hook
    """
        
    def hook(model: nn.Module, 
             input: torch.Tensor, 
             output: torch.Tensor
    ) -> None:
        """Appends the minimum and maximum activations to the holders."""
        min_holder.append(output.min().detach().item())
        max_holder.append(output.max().detach().item())
    
    return hook


def get_activation_range(model: nn.Module, 
                         layer_name: str, 
                         dataset_name: str
) -> Tuple[float, float]:
    """Returns the minimum and maximum activations of a layer in a model.
    
    Args:
        model (nn.Module): The model to measure
        layer_name (str): The layer in the model to measure
        dataset_name (str): The dataset used in measuring activations
    
    Returns:
        tuple[float, float]: minimum and maximum activations of the given layer
    """

    train_data_loader = _get_data_loader(dataset_name, "train", 32, 0)
    
    min_batch_activations = []
    max_batch_activations = []
    
    model.eval()
    layer = getattr(model, layer_name)
    handle = layer.register_forward_hook(_activation_range_hook(
                                            min_batch_activations, 
                                            max_batch_activations))

    for images, _ in train_data_loader:
        images = images.to(config.device)
        model(images)

    handle.remove()

    return min(min_batch_activations), max(max_batch_activations)