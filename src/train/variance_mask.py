from torch.utils.data import DataLoader
import torch
from dotmap import DotMap
import numpy as np
import os
from tqdm import tqdm
from packaging import version

import sys
if './' not in sys.path:
    sys.path.append('./')

from src.utils import config


def get_frank(data_dict, percentile=10, mode='after', replace_mean=False, apply_mask=True):
    frank_model, mask, mean = get_variance_mask_and_mean_from_data_dict(
        data_dict, mode, percentile=percentile)

    if replace_mean:
        weights = mean
    else:
        weights = frank_model.transform.transform.weight.detach().cpu().numpy()
    bias = frank_model.transform.transform.bias

    if apply_mask:
        mask = mask.reshape(weights.shape)
        weights = weights * mask

    frank_model.transform.load_trans_matrix(weights, bias)
    return frank_model


def get_variance_mask_and_mean_from_data_dict(data_dict,
                                              mode='after',
                                              percentile=10):
    from src.train import FrankModelTrainer
    from src.dataset import get_datasets
    model_trainer = FrankModelTrainer.from_data_dict(data_dict, mode)
    model_trainer.optimizer_name = 'sgd'
    model_trainer.optimizer = model_trainer._get_optimizer()
    datasets = get_datasets(data_dict['params']['dataset'])
    mask, mean = get_varaince_mask_and_mean(datasets,
                                            model_trainer,
                                            percentile=percentile)
    return model_trainer.model, mask, mean


def get_varaince_mask_and_mean(datasets, model_trainer, percentile=10):
    mask_obj = VarianceMask(datasets, model_trainer, percentile=percentile)
    return mask_obj()


class VarianceMask:
    def __init__(self,
                 datasets,
                 model_trainer,
                 percentile=10,
                 gradient_noise=None,
                 batch_size=100,
                 n_workers=4,
                 drop_last=True):

        self.datasets = datasets
        self.model_trainer = model_trainer
        self.percentile = percentile
        self.gradient_noise = gradient_noise
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.drop_last = drop_last

        self.model = model_trainer.model
        self.data_loaders = self._create_data_loaders(self.datasets,
                                                      self.gradient_noise,
                                                      self.n_workers,
                                                      self.drop_last)

        self.last_matrix = self.trans_matrix
        self.matrices = []

    @property
    def trans_matrix(self):
        return self.model.transform.transform.weight.detach().clone().cpu()

    def update(self):
        self.matrices.append(self.trans_matrix)

    def __call__(self):

        self.model.to(config.device)
        self.model.eval()

        data_loader = self.data_loaders['train']
        for (inputs, labels) in tqdm(data_loader):
            # Make a train step and update differences
            self.model_trainer.train_step(inputs, labels)
            self.update()

        self.matrices = torch.stack(self.matrices).detach().cpu().numpy()
        std_matrix = (self.matrices[1:] - self.matrices[:-1]).std(axis=0)
        mean_matrix = self.matrices.mean(axis=0)
        threshold = np.percentile(std_matrix, self.percentile)
        mask = np.where(std_matrix < threshold, np.ones_like(std_matrix),
                        np.zeros_like(std_matrix))
        mask = mask.reshape(mask.shape[:2])
        mean_matrix = mean_matrix.reshape(mean_matrix.shape[:2])
        return mask, mean_matrix

    def _create_data_loaders(self, datasets, gradient_noise, n_workers,
                             drop_last):

        if self.gradient_noise is not None:
            torch.manual_seed(gradient_noise)

        high_end_version = version.parse(
            torch.__version__) >= version.parse("1.7.0")

        common_settings = dict(batch_size=self.batch_size,
                               num_workers=n_workers,
                               pin_memory=True,
                               drop_last=drop_last)
        if high_end_version and n_workers > 0:
            common_settings['prefetch_factor'] = 10

        train = DataLoader(datasets.train, shuffle=True, **common_settings)
        val = DataLoader(datasets.val, **common_settings)

        data_loaders = DotMap({'train': train, 'val': val})
        return data_loaders


def cli_main(args=None):
    import argparse
    import pickle

    def_pickle = 'results/30epoch_all/matrix/conv5-conv52021-04-14 16:41:20.291845.pkl'
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('-p',
                        '--pickle',
                        help='Path to pickle',
                        default=def_pickle)
    args = parser.parse_args(args)

    data_dict = pickle.load(open(args.pickle, "br"))
    frank_model = get_frank(data_dict, apply_mask=False, replace_mean=True)
    print(frank_model.transform.transform.weight.detach().cpu().numpy())


if __name__ == '__main__':
    cli_main()