import torch
import pandas as pd
import os

from src.train import AutoencoderModelTrainer
from src.attacks import PGD

class RobustAutoencoderModelTrainer(AutoencoderModelTrainer):
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer_name='adam', 
                 lr=0.001, 
                 multilabel=False, 
                 weight_decay=0
    ):
        super().__init__(model, optimizer_name, lr, multilabel, weight_decay)
        
        self.attack = PGD(eps=8/255, 
                          alpha=16/255, 
                          n_steps=20, 
                          random_start=True, 
                          loss_fn=torch.nn.MSELoss())

    def step(self, inputs, _, is_train):

        inputs = inputs.to(self.device)
        # labels = labels.to(self.device)
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            model_outputs = self.model(inputs)
            loss = self.loss(inputs, model_outputs)
            metrics = self._get_metrics(loss)

            if is_train:
                adv_inputs = self.attack.generate(self.model, inputs, inputs)
                adv_outputs = self.model(adv_inputs)
                adv_loss = self.loss(adv_outputs, inputs)
                adv_loss.backward()
                self.optimizer.step()

        return metrics

    def _get_save_subfolder(self, data_name):
        in_folder = 'in' + str(self.model_trainer.model.seed)
        gn_folder = 'gn' + str(self.gradient_noise)
        initfolder = in_folder + '-' + gn_folder + '-rob'
        folder = os.path.join(self.save_folder, self.model.name, data_name,
                              initfolder)
        os.makedirs(folder, exist_ok=True)
        return folder
