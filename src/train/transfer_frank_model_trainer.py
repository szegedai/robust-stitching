import torch

from typing import List, Union
from src.train.madry_frank_model_trainer import MadryFrankModelTrainer


class TransferFrankModelTrainer(MadryFrankModelTrainer):
    def __init__(self, 
                 model, 
                 optimizer_name='sgd', 
                 lr=0.0001, 
                 multilabel=False, 
                 weight_decay=0, 
                 l1=0, 
                 cka_reg=0, 
                 temperature=1, 
                 target_type='hard', 
                 epsilon=8/255, 
                 alpha=2/255, 
                 n_steps=7, 
                 attack_ratio=1,
                 target_model: Union[str, List[str]] = [],
                 use_clean_examples: bool = False
    ) -> None:
        super().__init__(model, 
                         optimizer_name, 
                         lr, 
                         multilabel, 
                         weight_decay, 
                         l1, 
                         cka_reg, 
                         temperature, 
                         target_type, 
                         epsilon, 
                         alpha, 
                         n_steps, 
                         attack_ratio)

        if type(target_model) == str:
            self.target_model = []
        elif type(target_model) == list:
            self.target_model = target_model

        self.use_clean_examples = use_clean_examples
    
    @classmethod
    def from_arg_config(cls, conf):
        # Frankenstein model setup
        from src.models.frank.frankenstein import FrankeinsteinNet
        multilabel = conf.dataset == 'celeba'
        model = FrankeinsteinNet.from_arg_config(conf)

        return cls(model,
                   optimizer_name=conf.optimizer,
                   lr=conf.lr,
                   multilabel=multilabel,
                   weight_decay=conf.weight_decay,
                   l1=conf.l1,
                   cka_reg=conf.cka_reg,
                   temperature=conf.temperature,
                   target_type=conf.target_type,
                   epsilon=conf.epsilon,
                   alpha=conf.step_size,
                   n_steps=conf.perturb_steps,
                   attack_ratio=conf.attack_ratio,
                   target_model=conf.target_model,
                   use_clean_examples=conf.use_clean_examples)

    def step(self, inputs, labels, is_train):

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            
            if is_train:
                if "front" in self.target_model:
                    adv_inputs = self.attack.generate(self.model.front_model, 
                                                      inputs, 
                                                      labels)
                    adv_outputs = self.model(adv_inputs)
                    adv_loss = self.loss(adv_outputs, labels, inputs)
                    adv_loss.backward()
                if "end" in self.target_model:
                    adv_inputs = self.attack.generate(self.model.end_model, 
                                                      inputs, 
                                                      labels)
                    adv_outputs = self.model(adv_inputs)
                    adv_loss = self.loss(adv_outputs, labels, inputs)
                    adv_loss.backward()
                if self.use_clean_examples:
                    outputs = self.model(inputs)
                    loss = self.loss(outputs, labels, inputs)
                    loss.backward()

                self.optimizer.step()

            model_outputs = self.model(inputs)
            rounded_predictions = self._get_predictions(model_outputs)
            loss = self.loss(model_outputs, labels, inputs)
            metrics = self._get_metrics(labels, rounded_predictions, loss)

        return metrics

