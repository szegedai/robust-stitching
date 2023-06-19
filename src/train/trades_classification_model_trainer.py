import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


class TRADESClassificationModelTrainer:
    def __init__(self,
                 model,
                 optimizer_name='sgd',
                 lr=1e-4,
                 multilabel=False,
                 weight_decay=0.,
                 epsilon=8/255,
                 step_size=2/255,
                 perturb_steps=10,
                 beta=1.0
    ) -> None:
        self.model = model
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.multilabel = multilabel
        self.weight_decay = weight_decay

        self.epsilon = epsilon
        self.step_size = step_size
        self.perturb_steps = perturb_steps
        self.beta = beta

        self.loss = self._get_loss()
        self.optimizer = self._get_optimizer()
        self.stats = pd.DataFrame()

        # Put device on cuda if possible
        self.model.set_cuda()

    @classmethod
    def from_arg_config(cls, conf):
        from src.dataset import get_n_classes_and_channels
        from src.models import get_model
        n_classes, n_channels = get_n_classes_and_channels(conf.dataset)
        model = get_model(conf.model, n_classes, n_channels, conf.init_noise)
        multilabel = conf.dataset == 'celeba'

        return cls(model,
                   optimizer_name=conf.optimizer,
                   lr=conf.lr,
                   multilabel=multilabel,
                   weight_decay=conf.weight_decay,
                   epsilon=conf.epsilon,
                   step_size=conf.step_size,
                   beta=conf.beta,
                   perturb_steps=conf.perturb_steps)

    @classmethod
    def for_eval(cls, model_path):
        from src.models import load_from_path, get_info_from_path
        dataset = get_info_from_path(model_path)[1]
        multilabel = dataset == 'celeba'
        model = load_from_path(model_path)
        return cls(model, multilabel=multilabel)

    @property
    def device(self):
        ''' Cuda or CPU '''
        return self.model.device

    def save(self, n_iter_ran, save_folder):
        filename = str(n_iter_ran) + '.pt'
        path = os.path.join(save_folder, filename)
        torch.save(self.model.state_dict(),
                   path,
                   _use_new_zipfile_serialization=False)

    def append_and_save_log(self, epoch_data, save_folder):
        self.stats = self.stats.append(epoch_data, ignore_index=True)
        save_path = os.path.join(save_folder, 'training_log.csv')
        self.stats.to_csv(save_path, index=False)

    def summarize(self, shape):
        try:
            from torchsummary import summary
            summary(self.model, shape)
        except Exception as e:
            print('Could not print out model graph. Skipping.')

    def train_step(self, inputs, labels):
        self.step(inputs, labels, is_train=True)

    def eval_step(self, inputs, labels):
        self.step(inputs, labels, is_train=False)

    def evaluate(self, inputs, labels=None):
        self.model.eval()
        need_loss = not labels is None
        torch.cuda.empty_cache()
        inputs = inputs.to(self.device)
        with torch.set_grad_enabled(False):
            model_outputs = self.model(inputs)
            rounded_predictions = self._get_predictions(model_outputs)
            if need_loss:
                labels = labels.to(self.device)
                loss = self.loss(model_outputs, labels, inputs)

        self.model.train()
        if need_loss:
            return rounded_predictions, loss
        else:
            return rounded_predictions

    def step(self, inputs, labels, is_train):

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            model_outputs = self.model(inputs)
            rounded_predictions = self._get_predictions(model_outputs)
            loss = self.loss(model_outputs, labels, inputs)
            metrics = self._get_metrics(labels, rounded_predictions, loss)

            if is_train:
                train_loss = trades_loss(self.model,
                                         inputs,
                                         labels,
                                         self.optimizer,
                                         step_size=self.step_size,
                                         epsilon=self.epsilon,
                                         perturb_steps=self.perturb_steps,
                                         beta=self.beta)
                train_loss.backward()
                self.optimizer.step()

        return metrics

    def _get_predictions(self, outputs):
        if self.multilabel:
            preds = torch.where(torch.nn.Sigmoid()(outputs) > 0.5,
                                torch.ones_like(outputs),
                                torch.zeros_like(outputs))
        else:
            preds = torch.max(outputs.data, 1)[1]
        return preds

    def _get_loss(self):
        if self.multilabel:
            loss_function = torch.nn.BCEWithLogitsLoss()
        else:
            loss_function = torch.nn.CrossEntropyLoss()

        def mixed_loss(outputs, targets, inputs=None):
            # The input arg might be important for other loss calculation
            # so it's added as an optional argument here
            return loss_function(outputs, targets)

        return mixed_loss

    def _get_optimizer(self):
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(self.model.parameters(),
                                   lr=self.lr,
                                   momentum=0.9,
                                   nesterov=True,
                                   weight_decay=self.weight_decay)
        elif self.optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

    def _get_metrics(self, labels, preds, loss):
        data = {}
        data['loss'] = loss.item()
        data['accuracy'] = self._accuracy(labels, preds)
        return data

    def _accuracy(self, labels, preds):
        equality = torch.sum(labels == preds, dtype=torch.float32)
        accuracy = equality / labels.nelement()
        return accuracy

    def _get_save_subfolder(self, data_name):
        in_folder = 'in' + str(self.model_trainer.model.seed)
        gn_folder = 'gn' + str(self.gradient_noise)
        initfolder = in_folder + '-' + gn_folder
        folder = os.path.join(self.save_folder, self.model.name, data_name,
                              initfolder)
        os.makedirs(folder, exist_ok=True)
        return folder
