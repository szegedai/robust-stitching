import os

import torch
from src.dataset import get_n_classes_and_channels
from src.models.classifiers.resnet import resnet18, resnet34, resnet50
from src.models.classifiers.custom.EngstromRN50 import EngstromRN50
from src.models.classifiers.custom.ImageNetRN50 import ImageNetRN50
from src.models.classifiers.robust_vision_transformer import PoolingTransformer
from src.models.classifiers.simple_cnn import SimpleCNN
from src.models.classifiers.vgg import VGG11, VGG13, VGG16, VGG19
from src.models.classifiers.vision_transformer import VisionTransformer
from src.models.classifiers.wide_resnet import WideResNet, PreActResNet
from src.utils.config import get_cuda_device

celeba_models = [
    'imagenet', 'adam', 'adam_overfit', 'sgd', 'img-gn1', 'img-gn2', 'img-gn3',
    'in1-gn1', 'in1-gn2', 'in1-gn3', 'in2-gn1', 'in2-gn2', 'in2-gn3',
    'in3-gn1', 'in3-gn2', 'in3-gn3', 'img-gn1-bn', 'in1-gn1-bn', 'in2-gn1-bn',
    'in3-gn1-bn'
] + [f"FRANK_in{i}-gn{i}" for i in range(20)]


def load_from_path(path):
    # Get target device
    # device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = torch.device(device_name)
    device = get_cuda_device()


    # Skeleton model
    model = get_skeleton_model(path)
    model.to(device)

    # Load weights
    if path in celeba_models:
        celeba_name = path
        base_url = 'https://no_url'
        url = base_url + celeba_name + '.pt'
        state_dict = torch.hub.load_state_dict_from_url(url,
                                                        progress=True,
                                                        map_location=device)
    else:
        state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model

def get_info_from_path(path):
    if path in celeba_models:
        model_name = 'inceptionv1'
        n_classes = 40
        n_channels = 3
        data_name = 'celeba'
    else:
        path_split = path.split(os.sep)
        model_name = path_split[-4]
        data_name = path_split[-3]
        n_classes, n_channels = get_n_classes_and_channels(data_name)
    return model_name, data_name, n_classes, n_channels


def get_skeleton_model(path):
    ''' Reads in a model architecture with random init '''
    model_name, _, n_classes, n_channels = get_info_from_path(path)
    model = get_model(model_name, n_classes, n_channels, seed=None)
    return model


def get_model(str_nn,
              n_classes,
              n_channels,
              seed=None,
              model_path=None,
              celeba_name=None):
    str_nn = str_nn.lower()

    if str_nn == 'lenet':
        from src.models import LeNet
        model = LeNet(n_classes, n_channels, seed=seed, model_path=model_path)
    elif str_nn == 'tiny10':
        from src.models import Tiny10
        model = Tiny10(n_classes, n_channels, seed=seed, model_path=model_path)
    elif str_nn in ['nbn_tiny10', 'nbntiny10']:
        from src.models import NbnTiny10
        model = NbnTiny10(n_classes,
                          n_channels,
                          seed=seed,
                          model_path=model_path)
    elif str_nn == 'dense':
        from src.models import Dense
        model = Dense(n_classes, n_channels, seed=seed, model_path=model_path)
    elif str_nn == 'inceptionv1':
        from src.models import InceptionV1
        model = InceptionV1(n_classes, n_channels, celeba_name=celeba_name)
    elif str_nn == "simplecnn":
        model = SimpleCNN(n_classes, n_channels, model_path, seed)
    elif str_nn[:8] == 'resnet20':
        parts = str_nn.split('_')
        width = 1
        if len(parts) > 1:
            width = int(parts[1][1:])
        from src.models import ResNet20
        model = ResNet20(n_classes,
                         n_channels,
                         seed=seed,
                         model_path=model_path,
                         width=width)
    elif str_nn == 'resnet50':
        model = resnet50(n_classes, n_channels, seed=seed, model_path=model_path)
    elif str_nn == 'resnet34':
        model = resnet34(n_classes, n_channels, seed=seed, model_path=model_path)
    elif str_nn == 'resnet18':
        model = resnet18(n_classes, n_channels, seed=seed, model_path=model_path)
    elif str_nn == 'preactresnet18':
        model = PreActResNet(num_classes=n_classes,
                             depth=18,
                             width=0,
                             num_input_channels=n_channels,
                             seed=seed,
                             model_path=model_path)
    elif str_nn.startswith('wide_resnet'):
        tokens = str_nn.split('_')
        depth = int(tokens[2])
        width = int(tokens[3])
        model = WideResNet(n_classes, n_channels, depth, width)
    elif str_nn == 'vgg11':
        model = VGG11(n_classes=n_classes,
                      n_channels_in=n_channels,
                      seed=seed,
                      model_path=model_path)
    elif str_nn == 'vgg13':
        model = VGG13(n_classes=n_classes,
                      n_channels_in=n_channels,
                      seed=seed,
                      model_path=model_path)
    elif str_nn == 'vgg16':
        model = VGG16(n_classes=n_classes,
                      n_channels_in=n_channels,
                      seed=seed,
                      model_path=model_path)
    elif str_nn == 'vgg19':
        model = VGG19(n_classes=n_classes,
                      n_channels_in=n_channels,
                      seed=seed,
                      model_path=model_path)
    elif str_nn == 'engstromrn50':
        model = EngstromRN50(n_classes=n_classes, n_channels=n_channels, seed=seed, model_path=model_path)
    elif str_nn == 'imagenetrn50':
        model = ImageNetRN50(n_classes=n_classes, n_channels=n_channels, seed=seed, model_path=model_path)
    elif str_nn.startswith('vit_'):
        tokens = str_nn.split('_')
        patch_size = int(tokens[1])
        dim = int(tokens[2])
        depth = int(tokens[3])
        heads = int(tokens[4])
        mlp_dim = int(tokens[5])
        model=VisionTransformer(n_classes=n_classes,
                                n_channels_in=n_channels,
                                seed=seed,
                                model_path=model_path,
                                patch_size=patch_size,
                                dim=dim,
                                depth=depth,
                                heads=heads,
                                mlp_dim=mlp_dim)
    elif str_nn == 'rvit':
        model = PoolingTransformer(num_classes=n_classes,
                                   in_chans=n_channels,
                                   seed=seed,
                                   model_path=model_path)
    else:
        raise ValueError('Network {} is not known.'.format(str_nn))

    return model
