import torch
from torch import nn

from src.attacks import Attack
from src.dataset.utils import _get_data_loader


def eval_net_transfer_robustness(eval_model: nn.Module,
                                 attack_model: nn.Module,
                                 attack: Attack,
                                 str_dataset: str,
                                 batch_size: int = 64
) -> float:

    data_loader = _get_data_loader(str_dataset, "val", batch_size)

    corr = 0
    all = 0

    eval_model.eval()
    attack_model.eval()

    device = next(eval_model.parameters()).device

    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        
        # Generate adversarial example for attack_model
        adv_input = attack.generate(attack_model, input, target)

        # Evaluate eval_model on adversarial examples
        with torch.no_grad():
            output = eval_model(adv_input)
            corr += torch.argmax(output, 1).eq(target).sum().item()
            all += len(target)

    return corr / all
