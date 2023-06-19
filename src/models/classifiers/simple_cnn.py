import torch
from torch import nn

from typing import Optional
from src.models.classifiers import GeneralNet


class SimpleCNN(GeneralNet):
    def __init__(self, 
                 n_classes: int, 
                 n_channels_in: int, 
                 model_path: Optional[str] = None, 
                 seed: Optional[int] = None
    ) -> None:
        super().__init__(n_classes, n_channels_in, model_path, seed)

        # Layer 1
        self.conv1 = nn.Conv2d(n_channels_in, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # Layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # Layer 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        # Layer 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)

        # Layer 5
        self.pool5 = nn.MaxPool2d(4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, self.n_classes, bias=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Layer 1
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu1(out)

        # Layer 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        # Layer 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        # Layer 4
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.pool4(out)

        # Layer 5
        out = self.pool5(out)
        out = self.flatten(out)
        out = self.fc(out)

        return out
    
    @property
    def name(self):
        return "simplecnn"