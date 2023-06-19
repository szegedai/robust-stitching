import sys
import argparse
from typing import Dict

if './' not in sys.path:
    sys.path.append('./')

import numpy as np
from torch import nn
from art.attacks.evasion import FastGradientMethod, AutoAttack, AutoProjectedGradientDescent
from art.attacks import EvasionAttack
from art.estimators.classification import PyTorchClassifier

from src.train import ClassificationModelTrainer
from src.models import get_info_from_path
from src.dataset import get_n_classes_and_channels
from src.dataset import get_datasets

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('m', help='Path to model')
    return parser.parse_args(args)


def eval_on_attack(model: PyTorchClassifier, 
                   attack: EvasionAttack, 
                   x: np.ndarray, 
                   y: np.ndarray
) -> float:
    x_adv = attack.generate(x, y)
    pred = model.predict(x_adv)
    return np.sum(np.argmax(pred, axis=1) == y) / len(y)


def get_attacks(model: PyTorchClassifier, 
                str_dataset: str = "mnist"
) -> Dict[str, EvasionAttack]:
    # Dataset-independent parameters
    max_iter = 20
    n_random_init = 1
    batch_size = 64
    norm = np.inf

    # Dataset-dependent parameters
    # Linf attack eps
    eps = 0.3
    eps_step = 0.1
    if str_dataset.lower() in ["cifar10", "svhn"]:
        eps = 8/255 
        eps_step = 2/255
    elif str_dataset.lower() == "imagenet":
        eps = 4/255 
        eps_step = 1/255

    fastgrad = FastGradientMethod(estimator=model, eps=eps)

    autopgd_ce = AutoProjectedGradientDescent(estimator=model,
                                              norm=norm,
                                              eps=eps,
                                              eps_step=eps_step,
                                              max_iter=max_iter,
                                              targeted=False,
                                              nb_random_init=n_random_init,
                                              batch_size=batch_size,
                                              loss_type="cross_entropy",
                                              verbose=False)

    autopgd_logits = AutoProjectedGradientDescent(estimator=model,
                                                  norm=norm,
                                                  eps=eps,
                                                  eps_step=eps_step,
                                                  max_iter=max_iter,
                                                  targeted=False,
                                                  nb_random_init=n_random_init,
                                                  batch_size=batch_size,
                                                  loss_type="difference_logits_ratio",
                                                  verbose=False)

    auto_attack = AutoAttack(estimator=model,
                             eps=eps,
                             eps_step=eps_step,
                             batch_size=batch_size,
                             norm=norm,
                             attacks=[autopgd_ce, autopgd_logits])

    attacks = {
        'fastgrad': fastgrad,
        # 'autopgd_ce': autopgd_ce,
        # 'autopgd_logits': autopgd_logits,
        'autoattack': auto_attack,
    }

    return attacks


def eval_robustness(model_trainer: ClassificationModelTrainer,
                    str_dataset: str,
                    verbose: bool = True
) -> Dict[str, float]:
    """Evaluates the accuracy of a model on adversarial test examples"""
    # Model setup
    n_classes, n_channels = get_n_classes_and_channels(str_dataset)

    model = PyTorchClassifier(model=model_trainer.model,
                              loss=nn.CrossEntropyLoss(),
                              input_shape=(n_channels, 32, 32),
                              nb_classes=n_classes,
                              clip_values=(0.0, 1.0))

    # Dataset setup
    dataset = get_datasets(str_dataset.lower())['val']
    clean_val_x = []
    clean_val_y = []

    imagenet_counter = 0

    for x in dataset:
        clean_val_x.append(x[0].numpy())
        clean_val_y.append(x[1])

        if str_dataset.lower() == "imagenet":
            imagenet_counter += 1
            
            if imagenet_counter >= 200:
                break
    # clean_val_x = np.stack([x[0].numpy() for x in dataset])
    # clean_val_y = np.array([x[1] for x in dataset])

    clean_val_x = np.stack(clean_val_x)
    clean_val_y = np.array(clean_val_y)

    # if str_dataset.lower() == "imagenet":
    #     clean_val_x = clean_val_x[:5000]
    #     clean_val_y = clean_val_y[:5000]

    # Attacks setup
    attacks = get_attacks(model, str_dataset)

    # Evaluate model on attacks
    adv_accuracies = {}

    for atk_name, attack in attacks.items():
        attack.set_params(verbose=True)
        adv_acc = eval_on_attack(model, attack, clean_val_x, clean_val_y)
        adv_accuracies[atk_name] = adv_acc

        if verbose:
            print(f"Accuracy on adversarial test examples generated by {atk_name}: {adv_acc * 100}%")

    return adv_accuracies


def eval_net_robustness(model_path: str, verbose: bool = True):

    # Model setup
    model_trainer = ClassificationModelTrainer.for_eval(model_path)

    # # Dataset setup
    dataset_name = get_info_from_path(model_path)[1]

    eval_robustness(model_trainer, dataset_name, verbose)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)

    eval_net_robustness(args.m)


if __name__ == '__main__':
    main()
