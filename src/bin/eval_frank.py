import sys
import argparse
import torch

import pickle
import os

if './' not in sys.path:
    sys.path.append('./')

from src.models import get_model, get_frank_from_data_dict
from src.dataset import get_n_classes_and_channels, get_datasets

from src.utils.eval_values import eval_net

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('pickle', help='Path to pickle')
    parser.add_argument('mode', help='Which matrix to use', choices=['before', 'after', 'ps_inv'])
    return parser.parse_args(args)

def get_frank_model_object(data_dict, mode):
    # This function is deprected. Replaced by the following function
    import warnings
    warnings.warn('The function eval_frank.get_frank_model_object is '\
                  'deprecated. Please use src.models.frankenstein.get_'\
                  'frank_from_data_dict instead')
    return get_frank_from_data_dict(data_dict, mode)

def get_frankenstein(data_dict, mode):
    frank_model = get_frank_from_data_dict(data_dict, mode)
    return frank_model.model

def eval_frank(data_dict, mode, verbose=False):
    # Init variables
    data_name = data_dict['params']['dataset']

    # Load frankenstein
    model = get_frankenstein(data_dict, mode)

    # Calculate loss and accuracy
    mean_loss, mean_acc, hits = eval_net(model, data_name, verbose=verbose)

    # Print if requested
    if verbose:
        print('Loss: {:.3f} | Accuracy: {:2.2f}%'.format(mean_loss, mean_acc*100.))
    
    return mean_loss, mean_acc, hits

def eval_frank_from_pickle(pickle_path, mode, verbose=False):
    data_dict = pickle.load(open(pickle_path, "br"))
    return eval_frank(data_dict, mode, verbose)

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    eval_frank_from_pickle(args.pickle, args.mode, verbose=True)


if __name__ == '__main__':
    main()
