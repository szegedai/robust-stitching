import sys
import argparse
import uuid
from matplotlib import pyplot as plt
import numpy as np
import torch

from tqdm import tqdm

if './' not in sys.path:
    sys.path.append('./')

from src.train import AutoencoderModelTrainer
from src.dataset.utils import _get_data_loader
from src.dataset import get_datasets, get_n_classes_and_channels
from src.models import get_info_from_path
from src.utils.config import get_cuda_device

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('m', help='Path to model')
    return parser.parse_args(args)


def draw_ae_examples(model_trainer: AutoencoderModelTrainer, 
                     str_dataset: str, 
                     save_image: bool = True, 
                     show_image: bool = False
) -> str:
    model_trainer.model.eval()

    n_classes, n_channels = get_n_classes_and_channels(str_dataset)

    test_dataset = get_datasets("cifar10")["val"]
    targets = np.array(test_dataset.targets)
    t_idx = { i : np.where(targets==i)[0][0] for i in range(n_classes) }

    plt.figure(figsize=(16, 4.5))
    
    for i in range(n_classes):
        ax = plt.subplot(2,n_classes,i+1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(get_cuda_device())
        model_trainer.model.eval()

        with torch.no_grad():
            rec_img  = model_trainer.model(img)

        if n_channels > 1:
            plt.imshow(img.cpu().squeeze().numpy().transpose(1, 2, 0))
        else:  
            plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')  

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        
        if i == n_classes//2:
            ax.set_title('Original images')
        
        ax = plt.subplot(2, n_classes, i + 1 + n_classes)

        if n_channels > 1:
            plt.imshow(rec_img.cpu().squeeze().numpy().transpose(1, 2, 0))
        else:  
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        
        if i == n_classes//2:
            ax.set_title('Reconstructed images')
    
    out_file = None
    
    if save_image:
        out_file = str(uuid.uuid4()) + '.png'
        plt.savefig(out_file, bbox_inches='tight')
    
    if show_image:
        plt.show()

    return out_file


def eval_autoencoder(model_trainer: AutoencoderModelTrainer, 
                     str_dataset: str, 
                     dataset_type: str = "val", 
                     verbose: bool = False
) -> float:

    model_trainer.model.eval()

    batch_size = 32
    val_data_loader = _get_data_loader(str_dataset, 
                                       dataset_type, 
                                       batch_size=batch_size, 
                                       seed=0)
    model_outputs = []
    original_inputs = []

    val_data_loader = tqdm(val_data_loader) if verbose else val_data_loader

    for inputs, _ in val_data_loader:
        outputs = model_trainer.evaluate(inputs)
        model_outputs.append(outputs.detach().cpu())
        original_inputs.append(inputs.detach().cpu())
        torch.cuda.empty_cache()


    model_outputs = torch.cat(model_outputs)
    original_inputs = torch.cat(original_inputs)

    # Calcualte mean values for loss and accuracy
    loss = model_trainer.loss(model_outputs, original_inputs)

    return loss.item()


def run(model_path, verbose=False):

    # Initialize variables
    model_trainer = AutoencoderModelTrainer.for_eval(model_path)
    dataset_name = get_info_from_path(model_path)[1]

    # Calculate loss, accuracy and hits
    mean_loss = eval_autoencoder(model_trainer,
                                 dataset_name,
                                 "val",
                                 verbose=verbose)

    # Print if requested
    if verbose:
        print('Loss: {:.8f}'.format(mean_loss))
        draw_ae_examples(model_trainer, dataset_name, False, True)


    return mean_loss


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    run(args.m, verbose=True)


if __name__ == '__main__':
    main()