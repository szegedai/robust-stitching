from typing import Dict

import torch
from torch import nn

from src.dataset.utils import _get_data_loader


def output_label_ratio(model: nn.Module, 
                       dataset_name: str, 
                       batch_size: int = 32, 
                       train: bool = False
) -> Dict[int, float]:

    # Get device
    device = next(model.parameters()).device

    # Load dataset
    dataset_type = 'train' if train else 'val'
    data_loader = _get_data_loader(dataset_name, 
                                   dataset_type, 
                                   batch_size=batch_size, 
                                   seed=0)

    n_samples = 0
    output_label_counter = {}

    model.eval()
    for inputs, _ in data_loader:
        inputs = inputs.to(device)
        outputs = torch.argmax(model(inputs), dim=1).detach().cpu().numpy().flatten()

        n_samples += len(outputs)

        for pred_label in outputs:
            if pred_label in output_label_counter:
                output_label_counter[pred_label] += 1
            else:
                output_label_counter[pred_label] = 1

    # ratio
    for label in output_label_counter:
        output_label_counter[label] /= n_samples
    
    return output_label_counter
