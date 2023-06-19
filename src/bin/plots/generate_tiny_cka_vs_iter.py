import pandas as pd
import pickle
import os
import argparse
import sys


if './' not in sys.path:
    sys.path.append('./')

from src.train import FrankModelTrainer, Trainer
from src.dataset import get_datasets
from src.utils.eval_values import frank_m2_similarity
from src.comparators.activation_comparator import ActivationComparator

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('i', type=int, help='Index of tiny pairs from 0 to 9')
    parser.add_argument('-c','--csv', default="results/official/tiny_bn/summary.csv")
    parser.add_argument('-o','--out-dir', default='results/collection/cka_per_iter_lr')
    parser.add_argument('-cr','--cka-reg', type=float, default=0)
    parser.add_argument('-n', '--n-iter', type=int, default=500)
    parser.add_argument('-l', '--layer', default='bn3')
    parser.add_argument('--init', default='ps_inv')

    return parser.parse_args(args)

def get_df(csv, layer, init='ps_inv'):
    df = pd.read_csv(csv)
    w1 = df.init == init
    w2 = df.front_layer == layer
    df = df[w1&w2].reset_index()
    return df

def load_pickle(df, index):
    with open(df.loc[index, 'pickle'], 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

def get_frank_model_n_history(data_dict, n_cut, cka_reg):
    datasets = get_datasets(data_dict['params']['dataset'])
    data_dict['params']['cka_reg'] = cka_reg
    frank_trainer = FrankModelTrainer.from_data_dict(data_dict, 'before')
    trainer = Trainer(
        datasets,
        frank_trainer,
        batch_size=data_dict['params']['batch_size'],
        n_workers=4,
        drop_last=True,
        save_folder='results/cka_vs_iter',
    )
    freq = 10
    trainer.train(epochs=30, save_frequency=freq, freeze_bn=True)
    n_iters = min(n_cut, len(frank_trainer.transform_history))
    state_dicts = [frank_trainer.transform_history[i*freq] for i in range(n_iters)]
    indices = [i*freq for i in range(n_iters)]
    frank_model = frank_trainer.model
    return frank_model, state_dicts, indices

def run(conf):
    # Get data
    df = get_df(conf.csv, conf.layer, init=conf.init)
    data_dict = load_pickle(df, conf.i)

    # Run n iterations
    frank_model, state_dicts, indices = get_frank_model_n_history(data_dict, conf.n_iter, conf.cka_reg)

    # Init cka calculator
    dataset = data_dict['params']['dataset']

    # Load in each matrix and calculate measures
    results = {'cka' : [], 'crossentropy' : [], 'lr' : []}
    for state_dict in state_dicts:
        frank_model.transform.load_state_dict(state_dict)
        cka_comparator = ActivationComparator.from_frank_model(frank_model, 'frank', 'end')
        measures = cka_comparator(dataset, ['cka_torch', 'lr_torch'], dataset_type='val')
        ce = frank_m2_similarity(frank_model, dataset)['crossentropy']
        results['cka'].append(measures['cka_torch'])
        results['crossentropy'].append(ce)
        results['lr'].append(measures['lr_torch'])

    # Save
    os.makedirs(conf.out_dir, exist_ok=True)
    path = os.path.join(conf.out_dir, f"{conf.i}.csv")
    pd.DataFrame(results, index=indices).to_csv(path)

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    run(args)

if __name__ == '__main__':
    main()
