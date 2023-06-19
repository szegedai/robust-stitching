import argparse
import sys
from typing import Union


if './' not in sys.path:
    sys.path.append('./')

from src.dataset import get_datasets
from src.train.trainer import Trainer
from src.train.classification_model_trainer import ClassificationModelTrainer
from src.train.autoencoder_model_trainer import AutoencoderModelTrainer
from src.train.robust_autoencoder_model_trainer import RobustAutoencoderModelTrainer
from src.train.trades_classification_model_trainer import TRADESClassificationModelTrainer
from src.train.madry_classification_model_trainer import MadryClassificationModelTrainer


def _float_parse_type(eps: Union[str, int, float]) -> float:
    try:
        return float(eps)
    except ValueError:
        return eval(eps)


def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('-m',
                        '--model',
                        help='Name of model',
                        type=str,
                        default='lenet')
    # Choices were: ['lenet', 'tiny10', 'inceptionv1', 'resnet20']

    parser.add_argument(
        '-d',
        '--dataset',
        help='Name of dataset',
        choices=['fashion', 'mnist', 'cifar10', 'cifar100', 'celeba', 'svhn'],
        default='mnist')
    parser.add_argument('-i',
                        '--init-noise',
                        help='Starting position of nn (seed)',
                        type=int,
                        default=0)
    parser.add_argument('-g',
                        '--gradient-noise',
                        help='Order of batches (seed)',
                        type=int,
                        default=0)
    parser.add_argument('-s',
                        '--save-frequency',
                        help='How often to save in number of iterations',
                        type=int,
                        default=10000)
    parser.add_argument('-o',
                        '--out-dir',
                        help='Folder to save networks to',
                        default='snapshots')
    parser.add_argument('-e',
                        '--epochs',
                        help='Number of epochs to train',
                        type=int,
                        default=300)
    parser.add_argument('-b',
                        '--batch-size',
                        help='Batch size',
                        type=int,
                        default=128)
    parser.add_argument('-lr',
                        '--lr',
                        help='Learning rate',
                        type=float,
                        default=1e-1)
    parser.add_argument('-wd',
                        '--weight-decay',
                        help='Weight decay',
                        type=float,
                        default=1e-4)
    parser.add_argument('-opt',
                        '--optimizer',
                        help='Optimizer',
                        choices=['adam', 'sgd'],
                        default='sgd')
    parser.add_argument('--robust',
                        help='Robust training True/False',
                        action='store_true')
    parser.add_argument('--epsilon',
                        help='Epsilon for robust training',
                        type=_float_parse_type,
                        default=8/255)
    parser.add_argument('--step-size',
                        help='Optimization step size for attacks',
                        type=_float_parse_type,
                        default=2/255)
    parser.add_argument('--perturb-steps',
                        help='Perturbation steps for robust training',
                        type=int,
                        default=10)
    parser.add_argument('--beta',
                        help='Beta parameter of TRADES loss',
                        type=int,
                        default=1)
    parser.add_argument('--madry',
                        help='Robust training True/False',
                        action='store_true')
    parser.add_argument('--attack-ratio',
                        help='Ratio of adversarial inputs for Madry training',
                        type=float,
                        default=1)
    return parser.parse_args(args)


def run(conf):

    # Setup
    model_trainer = None
    if "autoencoder" in conf.model.lower():
        if conf.robust:
            model_trainer = RobustAutoencoderModelTrainer.from_arg_config(conf)
        else:
            model_trainer = AutoencoderModelTrainer.from_arg_config(conf)
    else:
        if conf.robust:
            model_trainer = TRADESClassificationModelTrainer.from_arg_config(conf)
        elif conf.madry:
            model_trainer = MadryClassificationModelTrainer.from_arg_config(conf)
        else:
            model_trainer = ClassificationModelTrainer.from_arg_config(conf)

    datasets = get_datasets(conf.dataset)

    # Print model on screen
    model_trainer.summarize(datasets['train'][0][0].shape)

    # Train
    trainer = Trainer(
        datasets,
        model_trainer,
        gradient_noise=conf.gradient_noise,
        batch_size=conf.batch_size,
        n_workers=4,
        drop_last=True,
        save_folder=conf.out_dir,
    )

    trainer.train(conf.epochs, conf.save_frequency)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    run(args)


if __name__ == '__main__':
    main()