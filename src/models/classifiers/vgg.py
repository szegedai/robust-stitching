from typing import OrderedDict
from torch import nn
from src.models.classifiers.general import GeneralNet


'''VGG11/13/16/19 in Pytorch.'''

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(GeneralNet):
    def __init__(self, vgg_name, n_classes=10, n_channels_in = 3, **kwargs):
        super(VGG, self).__init__(n_classes, n_channels_in, **kwargs)

        self.vgg_name = vgg_name
        self.n_classes = n_classes
        self.n_channels_in = n_channels_in

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = self._make_classifier()
        
        self._finish_init()

    def forward(self, x):
        out = self.features(x)
        latent = out.view(out.size(0), -1)
        out = self.classifier(latent)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.n_channels_in
        
        conv_block = 1
        conv_layer_in_block = 1

        for x in cfg:
            if x == 'M':
                layers += [(f'maxpool{conv_block}', nn.MaxPool2d(kernel_size=2, stride=2))]
                conv_block += 1
                conv_layer_in_block = 1
            else:
                layers += [
                    (f'conv{conv_block}_{conv_layer_in_block}', nn.Conv2d(in_channels, x, kernel_size=3, padding=1)),
                    (f'bn{conv_block}_{conv_layer_in_block}', nn.BatchNorm2d(x)),
                    (f'relu{conv_block}_{conv_layer_in_block}', nn.ReLU(inplace=True))
                ]
                in_channels = x
                conv_layer_in_block += 1
        
        layers += [('avgpool', nn.AvgPool2d(kernel_size=1, stride=1))]
        return nn.Sequential(OrderedDict(layers))

    def _make_classifier(self):
        return nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512, 256)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(256, 128)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.5)),
            ('fc3', nn.Linear(128, self.n_classes))
        ]))

    @property
    def name(self):
        return self.vgg_name


def VGG11(**kwargs):
    return VGG('VGG11', **kwargs)

def VGG13(**kwargs):
    return VGG('VGG13', **kwargs)

def VGG16(**kwargs):
    return VGG('VGG16', **kwargs)

def VGG19(**kwargs):
    return VGG('VGG19', **kwargs)
