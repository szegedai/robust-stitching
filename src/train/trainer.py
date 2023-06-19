import uuid
from torch.utils.data import DataLoader
import torch
import pkbar
from dotmap import DotMap
import pandas as pd
import os
from packaging import version

from src.utils import config
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from typing import Union, List, Tuple
from src.attacks import Attack
from src.attacks.pgd import PGD
from src.utils.stack_images import stack_images
from src.dataset.utils import order_dataset_by_label


def get_layer(model: nn.Module, layer_name: str) -> nn.Module:
    for name, layer in model.named_modules():
        if name == layer_name:
            return layer
    return None


def cosine_similarity(a: np.array, b: np.array, eps: float = 1e-8) -> float:
    # return np.inner(a, b) / max((np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_representation_difference(model: nn.Module, 
                                  layer_name: str, 
                                  loader: torch.utils.data.DataLoader, 
                                  attack: Attack,
                                  return_labels: bool = True
) -> Union[List[np.array], Tuple[List[float], np.array]]:
    # Get model device
    device = next(model.parameters()).device

    # Storing representation differences
    repr_differences = []
    labels = []
    
    # Storage for clean and adversarial representations
    clean_repr = None
    adv_repr = None

    # Clean representation storage hook
    def clean_repr_store(model: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        nonlocal clean_repr
        clean_repr = torch.flatten(output.cpu().detach().clone(), start_dim=1)
    
    # Adversarial representation storage hook
    def adv_repr_store(model: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        nonlocal adv_repr
        adv_repr = torch.flatten(output.cpu().half().detach().clone(), start_dim=1)

    # Get observed layer and register hooks
    layer = get_layer(model, layer_name)
    clean_repr_store_hook = layer.register_forward_hook(clean_repr_store)
    adv_repr_store_hook = layer.register_forward_hook(adv_repr_store)

    # Process data
    model.eval()
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # Remove both hooks to avoid duplicate and false storing
        clean_repr_store_hook.remove()
        adv_repr_store_hook.remove()

        # Clean inputs
        clean_repr_store_hook = layer.register_forward_hook(clean_repr_store)
        model(x)
        clean_repr_store_hook.remove()

        # Adversarial inputs
        adv_x = attack.generate(model, x, y)
        adv_repr_store_hook = layer.register_forward_hook(adv_repr_store)
        model(adv_x)
        adv_repr_store_hook.remove()
        
        # Store representation difference
        repr_diff = clean_repr - adv_repr
        repr_differences.extend(repr_diff)

        if return_labels:
            labels.extend(y.detach().cpu().numpy())
    

    if return_labels:
        return repr_differences, labels
    else:
        return repr_differences


def get_attack_cosine_similarities(model: nn.Module, 
                                   layer_name: str, 
                                   label_of_interest: int,
                                   data_loader: torch.utils.data.DataLoader, 
                                   attack: Attack,
) -> List[float]:
    repr_diffs, labels = get_representation_difference(model, 
                                                       layer_name, 
                                                       data_loader, 
                                                       attack)

    # Compute cosine similarities
    cosine_similarities = []

    for i in range(len(repr_diffs)):
        for j in range(i+1, len(repr_diffs)):
            if labels[i] == labels[j] and labels[i] == label_of_interest:
                cosine_similarities.append(cosine_similarity(repr_diffs[i], 
                                                             repr_diffs[j]))

    return cosine_similarities


def plot_cosine_similarities(cosine_similarities: List[float], 
                             layer_name: str, 
                             label_of_interest: int, 
                             epoch: int, 
                             filename: str
) -> None:
    # Plot histogram of cosine similarities
    plt.figure(figsize=(10, 5))
    n, bins, patches = plt.hist(x=cosine_similarities, 
                                bins='auto', 
                                color='#0504aa',
                                alpha=1)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Cosine similarity')
    # plt.ylabel('Frequency')
    plt.title('Cosine similarity of adversarial representation shift for class ' 
             f'{label_of_interest} at layer {layer_name} at epoch {epoch}')
    maxfreq = n.max() # save this for later
    # Set a clean upper y-axis limit.
    ax = plt.gca()
    ax.relim()
    # plt.ylim(ymax=maxfreq+10)
    plt.xlim(xmin=-1, xmax=1)
    # plt.show()
    plt.savefig(filename)
    pass


def plot_cosine_similarity_trend(cos_sim_mean: List[float], 
                                 cos_sim_std: List[float], 
                                 filename: str
) -> None:
    x = range(len(cos_sim_mean))

    plt.figure(figsize=(10, 5))
    plt.plot(x, cos_sim_mean)
    plt.fill_between(x, cos_sim_mean - cos_sim_std, cos_sim_mean + cos_sim_std, color="blue", alpha=0.2)
    plt.ylim(-1, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Cosine similarity")

    
    plt.savefig(filename)
    

def get_cosine_similarities(vecs: List[np.ndarray]) -> List[float]:
    cos_sims = []

    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            cos_sims.append(cosine_similarity(vecs[i], vecs[j]))
    
    return cos_sims


def get_attacked_representation(model: nn.Module, 
                                layer_name: str, 
                                x: torch.Tensor, 
                                y: torch.Tensor, 
                                attack: Attack,
                                label_of_interest: int = 0
) -> np.ndarray:
    representation = None

    def repr_store_hook(model: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        nonlocal representation
        representation = torch.flatten(output.cpu().detach().clone(), start_dim=1).numpy()
    
    # Put model in eval_mode
    model.eval()

    # Generate adversarial example
    x_adv = attack.generate(model, x, y)

    # Attach forward hook
    layer = get_layer(model, layer_name)
    hook_handler = layer.register_forward_hook(repr_store_hook)

    # Use forward pass
    model(x_adv)

    # Filter only representations of label of interest
    label_indices = [idx for idx, label in enumerate(y) if label == label_of_interest]
    representation = representation[label_indices]

    # Remove forward hook
    hook_handler.remove()

    # Put model back to train mode
    model.train()

    return representation


class Trainer:
    def __init__(self,
                 datasets,
                 model_trainer,
                 gradient_noise=None,
                 batch_size=32,
                 n_workers=4,
                 drop_last=False,
                 save_folder='snapshots',
                 lr_schedule=[(0.333, 0.1), (0.666, 0.1)],
                 lr_schedule_type='step_function',
                 plot_attack_cosine_similarities: bool = False,
                 eval_attack: Attack = None,
                 disable_bn: bool = False
    ) -> None:

        self.datasets = datasets
        self.model_trainer = model_trainer
        self.gradient_noise = gradient_noise
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.drop_last = drop_last
        self.save_folder = save_folder
        self.lr_schedule = lr_schedule
        self.lr_schedule_index = 0
        self.lr_schedule_type = lr_schedule_type
        self.plot_attack_cosine_similarities = plot_attack_cosine_similarities
        self.image_files = {
            "train": [],
            "val": []
        }
        self.cosine_similarity_mean = []
        self.cosine_similarity_std = []

        self.eval_attack = eval_attack

        self.disable_bn = disable_bn

        self.model = model_trainer.model
        self.data_loaders = self._create_data_loaders(self.datasets,
                                                      self.gradient_noise,
                                                      self.n_workers,
                                                      self.drop_last)

    @property
    def _pbar_length(self):
        data_loader = self.data_loaders.train
        return self._get_n_iter(data_loader)

    @property  # Should be cached_property on python 3.8
    def save_subfolder(self):
        data_name = str(self.datasets.train.__class__.__name__)
        in_folder = 'in' + str(self.model.seed)
        gn_folder = 'gn' + str(self.gradient_noise)
        initfolder = in_folder + '-' + gn_folder
        folder = os.path.join(self.save_folder, self.model.name, data_name,
                              initfolder)
        os.makedirs(folder, exist_ok=True)
        return folder

    def _get_n_iter(self, data_loader):
        return len(data_loader.dataset) // data_loader.batch_size

    def _get_pbar(self, n, i, n_epochs):
        return pkbar.Kbar(target=n,
                          epoch=i,
                          num_epochs=n_epochs,
                          width=8,
                          always_stateful=False)

    def _update_running_metrics(self, orig, new):
        for k, v in new.items():
            if k not in orig:
                orig[k] = new[k]
            else:
                orig[k] += new[k]
        return orig

    def save(self, n_iter_ran):
        self.model_trainer.save(n_iter_ran, self.save_subfolder)

    def _append_and_save_log(self, epoch_data):
        self.model_trainer.append_and_save_log(epoch_data, self.save_subfolder)

    def _update_lr(self, progress):
        if self.lr_schedule_index >= len(self.lr_schedule):
            return
        next_schedule = self.lr_schedule[self.lr_schedule_index]
        progress_threshold = next_schedule[0]
        if progress >= progress_threshold:
            multiplier = next_schedule[1]
            for g in self.model_trainer.optimizer.param_groups:
                g['lr'] *= multiplier
            self.lr_schedule_index += 1

    def train(self, epochs=10, save_frequency=None, freeze_bn=False):

        self.model.to(config.device)

        # Turn on running mean and std for batch norm
        if freeze_bn:
            self.model.eval()
        else:
            self.model.train()

        n_iterations_ran = 0

        if save_frequency is not None:
            self.save(0)

        if self.lr_schedule_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.model_trainer.optimizer, factor=0.1)

        for epoch_i in range(epochs):

            # Metrics and progress bar
            epoch_data = {}
            pbar = self._get_pbar(self._pbar_length, epoch_i, epochs)
            if self.lr_schedule_type == 'step_function':
                self._update_lr(epoch_i / epochs)

            for phase in ['train', 'val']:

                # Init metrics and data load dataloader
                running_metrics = {}
                data_loader = self.data_loaders[phase]
                n_iter = self._get_n_iter(data_loader)

                # Store attacked representational differences
                atk_repr_diffs = []

                # Put model in the correct mode
                if phase == 'val':
                    self.model_trainer.model.eval()
                else:
                    if not self.disable_bn: # only in Frank training
                        self.model_trainer.model.train()
                    else:
                        # When disabling BN, Frank training only modifies the 
                        # stitching matrix and the BN modules of the original
                        # networks remain unaffected (their running_mean and
                        # running_var parameters remain unchanged)
                        self.model_trainer.model.transform.train() 
                
                for i, (inputs, labels) in enumerate(data_loader):
                    # If analysing attack angle similarities, get adversarial
                    # representations before backpropagation
                    pre_backprop_repr = None
                    post_backprop_repr = None
                    
                    if self.plot_attack_cosine_similarities and phase == 'train':
                        pre_backprop_repr = get_attacked_representation(self.model, "transform", inputs, labels, self.eval_attack)

                    # Make a train step and retreive metrics
                    new_metrics = self.model_trainer.step(inputs, labels, phase == 'train')
                    running_metrics = self._update_running_metrics(running_metrics, new_metrics)
                    new_values = [(phase + '_' + name, value)
                                  for (name, value) in new_metrics.items()]
                    # Save model checkpoint
                    if phase == 'train':
                        pbar.update(i, values=new_values)
                        n_iterations_ran += 1
                        if save_frequency and n_iterations_ran % save_frequency == 0:
                            self.save(n_iterations_ran)

                    # Save representation difference after backpropagation
                    if self.plot_attack_cosine_similarities and phase == 'train':
                        post_backprop_repr = get_attacked_representation(self.model, "transform", inputs, labels, self.eval_attack)
                        atk_repr_diffs.extend(pre_backprop_repr - post_backprop_repr)

                # Save information about this epoch
                for name, value in running_metrics.items():
                    if name != 'loss':
                        value = value.detach().cpu().numpy()
                    epoch_data[phase + '_' + name] = value / n_iter

                if phase == 'val':
                    val_values = [(x, y) for (x, y) in epoch_data.items()
                                  if x.startswith('val_')]
                    pbar.add(1, values=val_values)
                    if self.lr_schedule_type == 'reduce_on_plateau':
                        scheduler.step(epoch_data['val_loss'])

                if self.plot_attack_cosine_similarities and phase == 'train':
                    # Get all cosine similarities
                    cos_sims = get_cosine_similarities(atk_repr_diffs)

                    filename = f"e_{epoch_i}_{phase}_cos_sim_{uuid.uuid4()}.png"
                    plot_cosine_similarities(cos_sims, 
                                             "transform", 
                                             0, 
                                             epoch_i, 
                                             filename)
                    
                    cos_sims = np.array(cos_sims)
                    self.cosine_similarity_mean.append(np.mean(cos_sims))
                    self.cosine_similarity_std.append(np.std(cos_sims))

                    self.image_files[phase].append(filename)
                
                
                # if self.plot_attack_cosine_similarities and phase == 'val':
                #     # Plot attack cosine similarities
                #     filename = f"e_{epoch_i}_{phase}_cos_sim_{uuid.uuid4()}.png"
                #     cos_sims = get_attack_cosine_similarities(self.model_trainer.model, 
                #                                               "transform", 
                #                                               0, 
                #                                               data_loader, 
                #                                               self.eval_attack)
                #     plot_cosine_similarities(cos_sims, 
                #                              "transform", 
                #                              0, 
                #                              epoch_i, 
                #                              filename)

                #     cos_sims = np.array(cos_sims)
                #     self.cosine_similarity_mean.append(np.mean(cos_sims))
                #     self.cosine_similarity_std.append(np.std(cos_sims))

                #     self.image_files[phase].append(filename)
            
            # Save training log
            self._append_and_save_log(epoch_data)
            atk_repr_diffs = []

        if self.plot_attack_cosine_similarities:
            cosine_sims_filename = f'cosine_sims_{uuid.uuid4()}.png'
            stack_images(self.image_files['train'], cosine_sims_filename)
            self.image_files['train'].append(cosine_sims_filename)

            cosine_sims_trend_filename = f"cosine_sims_trend_{uuid.uuid4()}.png"
            plot_cosine_similarity_trend(np.array(self.cosine_similarity_mean), 
                                         np.array(self.cosine_similarity_std),
                                         cosine_sims_trend_filename)
            self.image_files['train'].append(cosine_sims_trend_filename)

    def _create_data_loaders(self, 
                             datasets, 
                             gradient_noise, 
                             n_workers,
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

        # if self.plot_attack_cosine_similarities:
        #     datasets.train = order_dataset_by_label(datasets.train)

        train = DataLoader(datasets.train, 
                           shuffle=not self.plot_attack_cosine_similarities, 
                           **common_settings)
        val = DataLoader(datasets.val, **common_settings)

        data_loaders = DotMap({'train': train, 'val': val})
        return data_loaders
