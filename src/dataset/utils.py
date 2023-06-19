import torch
import torch.utils.data as data_utils
from torch.utils.data import Subset

from src.dataset import get_datasets
from src.utils import config

def _get_data_loader(dataset_name,
                     dataset_type,
                     batch_size=256,
                     seed=None):
    ''' Get pytorch dataloader given the dataset_name and the type
        (train or val)'''

    if seed is not None:
        torch.manual_seed(seed)

    dataset = get_datasets(dataset_name)[dataset_type]
    if config.debug:
        debug_size = min(3 * batch_size + 1, len(dataset))
        dataset = data_utils.Subset(dataset, torch.arange(debug_size))
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                            #   num_workers=0 if config.debug else 4,
                                              num_workers=0, # TODO: check if necessary due to hardware limitations
                                              pin_memory=True,
                                              drop_last=False)
    return data_loader


def order_dataset_by_label(dataset: torch.utils.data.Dataset) -> Subset:
    """Sorts the dataset by labels in ascending order"""
    sorted_indices = []

    for label_idx in range(0, max(dataset.targets) + 1):
        indices = [idx for idx, label in enumerate(dataset.targets) if label == label_idx]
        sorted_indices.extend(indices)

    sorted_dataset = Subset(dataset, indices=sorted_indices)

    return sorted_dataset
