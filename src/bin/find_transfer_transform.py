import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Union

import warnings
import torch

warnings.filterwarnings('ignore')

import git

if './' not in sys.path:
    sys.path.append('./')

from src.bin.eval_net_robustness import eval_robustness
from src.dataset import get_datasets

from src.train import Trainer, ClassificationModelTrainer, TransferFrankModelTrainer

from src.attacks.pgd import PGD
from src.utils.eval_values import eval_net
from src.utils.robustness import eval_net_transfer_robustness
from src.utils.to_pdf import to_pdf


def _float_parse_type(eps: Union[str, int, float]) -> float:
    try:
        return float(eps)
    except ValueError:
        return eval(eps)


def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('front_model', help='Path first model', type=str)
    parser.add_argument('end_model', help='Path to second model', type=str)
    parser.add_argument('front_layer',
                        help='Last layer of first model',
                        type=str)
    parser.add_argument('end_layer',
                        help='Last layer of second model',
                        type=str)
    parser.add_argument(
        'dataset',
        help='Name of dataset',
        choices=['fashion', 'mnist', 'cifar10', 'cifar100', 'celeba', 'svhn', 'imagenet'])
    parser.add_argument('-o',
                        '--out-dir',
                        help='Dir to save matrix to',
                        default='results/')
    parser.add_argument('--run-name',
                        help='Name of the current run',
                        type=str,
                        default=None)
    parser.add_argument('--optimizer',
                        help='Optimizer',
                        choices=['adam', 'sgd'],
                        default='adam')
    parser.add_argument('--seed',
                        help='Seed of init and grad',
                        type=int,
                        default=0)
    parser.add_argument('-e',
                        '--epochs',
                        help='Number of epochs to train',
                        type=int,
                        default=30)
    parser.add_argument('-s',
                        '--save-frequency',
                        help='Save per iterations',
                        type=int,
                        default=10)
    parser.add_argument('-b',
                        '--batch-size',
                        help='Batch size',
                        type=int,
                        default=128)
    parser.add_argument('-lr',
                        '--lr',
                        help='Learning rate',
                        type=float,
                        default=1e-3)
    parser.add_argument('-wd',
                        '--weight-decay',
                        help='Weight decay',
                        type=float,
                        default=1e-4)
    parser.add_argument('--debug',
                        help='Debug True/False',
                        action='store_true')
    parser.add_argument('--flatten',
                        help='Flatten around transformation',
                        action='store_true')
    parser.add_argument('--l1', help='l1 value', type=float, default=0.)
    parser.add_argument('--cka-reg', help='Regularization on cka in stitching layer', type=float, default=0.)
    parser.add_argument('-i','--init',
                        help='Initial matrix',
                        choices=['random', 'perm', 'eye', 'ps_inv', 'ones-zeros', 'zeros'],
                        default='random')
    parser.add_argument('-m','--mask',
                        help='Mask applied on transformation',
                        choices=['identity', 'semi-match', 'abs-semi-match', 'random-permutation'],
                        default='identity')
    parser.add_argument('-r',
                        '--low-rank',
                        help='Rank',
                        type=int,
                        default=None)
    parser.add_argument('--target-type',
                        help='type of label to use for training',
                        choices=['hard', 'soft_1', 'soft_2', 'soft_12', 'soft_1_plus_2'],
                        default='hard')
    parser.add_argument('--temperature',
                        help='temperature of soft labels',
                        type=float,
                        default=1.)
    parser.add_argument('--epsilon',
                        help='Epsilon for robust training',
                        type=_float_parse_type,
                        default=8/255)
    parser.add_argument('--step-size',
                        help='Epsilon for robust training',
                        type=_float_parse_type,
                        default=2/255)
    parser.add_argument('--perturb-steps',
                        help='Perturbation steps for robust training',
                        type=int,
                        default=10)
    parser.add_argument('--attack-ratio',
                        help='Ratio of adversarial inputs for Madry training',
                        type=float,
                        default=1)
    parser.add_argument('--robust-model',
                        help='Target model to be attacked when creating adversarial inputs',
                        choices=['front', 'end'],
                        nargs='*',
                        default=[])
    return parser.parse_args(args)


def get_hits_overlap(m1_hits, m2_hits):
    total_overlap = (m1_hits == m2_hits).sum()
    m1_right_m2_right = (m1_hits[m2_hits == m1_hits]).sum()
    m1_right_m2_wrong = (m1_hits[m2_hits != m1_hits]).sum()
    m1_wrong_m2_right = (m2_hits[m2_hits != m1_hits]).sum()
    m1_wrong_m2_wrong = total_overlap - m1_right_m2_right
    return {
        'rr': m1_right_m2_right,
        'rw': m1_right_m2_wrong,
        'wr': m1_wrong_m2_right,
        'ww': m1_wrong_m2_wrong
    }


def create_data(conf, trainers, pre_train_matrices):
    out_data = {}

    # Setup transfer base attack
    transfer_adversary = PGD(eps=conf.epsilon, 
                             alpha=conf.step_size, 
                             n_steps=20, 
                             random_start=True, 
                             targeted=False, 
                             loss_fn=torch.nn.CrossEntropyLoss(), 
                             min=0, max=1, attack_ratio=1)

    # Save original models' normal and robust accuracies
    m1_runner = ClassificationModelTrainer.for_eval(conf.front_model)
    m2_runner = ClassificationModelTrainer.for_eval(conf.end_model)
    m1_loss, m1_acc, _ = eval_net(m1_runner, conf.dataset)
    m2_loss, m2_acc, _ = eval_net(m2_runner, conf.dataset)
    m1_rob_acc = eval_robustness(m1_runner, conf.dataset)
    m2_rob_acc = eval_robustness(m2_runner, conf.dataset)
    m1 = m1_runner.model
    m2 = m2_runner.model
    m1_transfer_m1 = eval_net_transfer_robustness(m1, 
                                                  m1, 
                                                  transfer_adversary, 
                                                  conf.dataset, 
                                                  conf.batch_size)
    m1_transfer_m2 = eval_net_transfer_robustness(m1, 
                                                  m2, 
                                                  transfer_adversary, 
                                                  conf.dataset, 
                                                  conf.batch_size)
    m2_transfer_m1 = eval_net_transfer_robustness(m2, 
                                                  m1, 
                                                  transfer_adversary, 
                                                  conf.dataset, 
                                                  conf.batch_size)
    m2_transfer_m2 = eval_net_transfer_robustness(m2, 
                                                  m2, 
                                                  transfer_adversary, 
                                                  conf.dataset, 
                                                  conf.batch_size)

    out_data["model_results"] = {
        'front': {
            'loss': m1_loss,
            'acc': m1_acc,
            **m1_rob_acc,
            'front_transfer': m1_transfer_m1,
            'end_transfer': m1_transfer_m2
        },
        'end': {
            'loss': m2_loss,
            'acc': m2_acc,
            **m2_rob_acc,
            'front_transfer': m2_transfer_m1,
            'end_transfer': m2_transfer_m2
        },
    }

    # Save running parameters
    out_data['params'] = vars(conf)

    # Save data about every stitched network
    stitching_types = ["front_trans", "end_trans", "both_trans", "repr_union"]

    for idx, stitch_type in enumerate(stitching_types):
        trainer = trainers[idx]
        pre_train_mtx = pre_train_matrices[idx]

        frank_model = trainer.model_trainer.model

        frank_loss, frank_acc, _ = eval_net(trainer.model_trainer, 
                                                     conf.dataset)

        frank_adv_acc = eval_robustness(trainer.model_trainer, conf.dataset)
        
        frank_transfer_m1 = eval_net_transfer_robustness(frank_model, 
                                                         m1, 
                                                         transfer_adversary, 
                                                         conf.dataset, 
                                                         conf.batch_size)
        frank_transfer_m2 = eval_net_transfer_robustness(frank_model, 
                                                         m2, 
                                                         transfer_adversary, 
                                                         conf.dataset, 
                                                         conf.batch_size)

        

        out_data[stitch_type] = {
            'loss': frank_loss,
            'acc': frank_acc,
            **frank_adv_acc,
            'front_transfer': frank_transfer_m1,
            'end_transfer': frank_transfer_m2
        }

    # Git info
    repo = git.Repo('./')
    out_data['git'] = {
        'branch': repo.active_branch.name,
        'commit': repo.head.commit.hexsha
    }

    out_data['runner_code'] = ' '.join(['python'] + sys.argv)

    out_data["trans_m"] = []
    out_data["trans_fit"] = []
    out_data["transform_svs"] = []

    return out_data


def _extract_trans_diff(data):
    before = data['before']
    after = data['after']
    w = after['w'] - before['w']
    b = after['b'] - before['b']
    return {'w': w, 'b': b}


def _extract_trans_w_and_b(model):
    w_b_dict = model.transform.get_param_dict()
    return w_b_dict


def save(conf, trainers, pre_train_matrices):
    data = create_data(conf, trainers, pre_train_matrices)

    # Create outgoing directory if not exist
    os.makedirs(os.path.join(conf.out_dir, 'matrix'), exist_ok=True)
    os.makedirs(os.path.join(conf.out_dir, 'pdf'), exist_ok=True)

    # Save pdf
    #filename = str(now)#.strftime("%Y-%m-%d--%H-%M-%S")
    now = datetime.now()
    filename = '{}-{}'.format(conf.front_layer, conf.end_layer) + str(now)
    filename = "_".join(filename.split(":"))

    if conf.low_rank is None:
        pdf_file = os.path.join(conf.out_dir, 'pdf', filename + '.pdf')
        to_pdf(data, pdf_file, now, "templates/transfer.html")

    # Save transform matrix
    matrix_filenames = ["front_trans", "end_trans", "both_trans", "repr_union"]
    for idx, trainer in enumerate(trainers):
        matrix_file = os.path.join(conf.out_dir, 
                                   "matrix", 
                                   f"{matrix_filenames[idx]}_{filename}.pt")
        
        torch.save(trainer.model_trainer.model.transform.state_dict(),
                   matrix_file,
                   _use_new_zipfile_serialization=False)

def set_random_seeds(seed=0):
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)

def run(conf):

    conf.use_clean_examples = False
    conf.target_model = conf.robust_model

    # Retreve save folder
    now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = f"{conf.run_name}" if conf.run_name is not None else f"{now}"
    conf.out_dir = os.path.join(conf.out_dir, run_name)
    
    # Get datasets
    datasets = get_datasets(conf.dataset)

    # Setup save frequency
    save_frequency = conf.save_frequency * conf.epochs
    
    # Training types
    # 1: adv examples for front model
    # 2: adv examples for end model
    # 3: adv examples for both models
    # 4: normal examples and adv examples for the robust model (conf.target_model)

    # Adv examples for front model
    front_trans_trainer = TransferFrankModelTrainer.from_arg_config(conf)
    front_trans_trainer.target_model = ["front"]
    front_trans_trainer.use_clean_examples = False

    # trans_m_before_train = _extract_trans_w_and_b(front_trans_trainer.model)


    # Train #1
    trainer_1 = Trainer(
        datasets,
        front_trans_trainer,
        batch_size=conf.batch_size,
        n_workers=4,
        drop_last=True,
        save_folder=conf.out_dir,
    )
    trainer_1.train(conf.epochs, save_frequency, freeze_bn=True)

    # Adv examples for end model
    end_trans_trainer = TransferFrankModelTrainer.from_arg_config(conf)
    end_trans_trainer.target_model = ["end"]
    end_trans_trainer.use_clean_examples = False

    # Train #2
    trainer_2 = Trainer(
        datasets,
        end_trans_trainer,
        batch_size=conf.batch_size,
        n_workers=4,
        drop_last=True,
        save_folder=conf.out_dir,
    )
    trainer_2.train(conf.epochs, save_frequency, freeze_bn=True)

    # Adv examples for both models
    both_trans_trainer = TransferFrankModelTrainer.from_arg_config(conf)
    both_trans_trainer.target_model = ["front", "end"]
    both_trans_trainer.use_clean_examples = False

    # Train #3
    trainer_3 = Trainer(
        datasets,
        both_trans_trainer,
        batch_size=conf.batch_size,
        n_workers=4,
        drop_last=True,
        save_folder=conf.out_dir,
    )
    trainer_3.train(conf.epochs, save_frequency, freeze_bn=True)

    # Representation union stitching: use adversarial examples of robust models
    # along with clean examples if at least one of the models is non-robust
    repr_union_trainer = TransferFrankModelTrainer.from_arg_config(conf)
    repr_union_trainer.target_model = conf.robust_model
    repr_union_trainer.use_clean_examples = len(conf.robust_model) < 2

    # Train #4
    trainer_4 = Trainer(
        datasets,
        repr_union_trainer,
        batch_size=conf.batch_size,
        n_workers=4,
        drop_last=True,
        save_folder=conf.out_dir,
    )
    trainer_4.train(conf.epochs, save_frequency, freeze_bn=True)

    # Save
    save(conf, [trainer_1, trainer_2, trainer_3, trainer_4], [None, None, None, None])
    print('Done.')


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    set_random_seeds(args.seed)

    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
    logger = logging.getLogger("trans")
    logger.setLevel(logging.DEBUG if args.debug else logging.WARNING)

    run(args)


if __name__ == '__main__':
    main()
