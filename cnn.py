import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from torch.autograd.variable import Variable
import util


class CNN(nn.Module):
    def __init__(self, input_size, in_channels, num_classes, hidden_size):
        super(CNN, self).__init__()
        self._activations = nn.Sequential(nn.Conv2d(in_channels, 10, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
                                          nn.Conv2d(10, 10, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
                                          nn.Conv2d(10, 32, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2))
        self.asize = util.infer_shape((in_channels, input_size[0], input_size[1]), self._activations)
        self._linear1 = nn.Linear(np.prod(self.asize), hidden_size)
        self._linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self._activations(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self._linear1(x))
        x = self._linear2(x)
        return x

    def activations(self, x):
        return self._activations(x)


class ResNet(nn.Module):
    def __init__(self, input_size, num_classes, in_channels=3, base_model=resnet34(pretrained=True)):
        super(ResNet, self).__init__()
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.avgpool = base_model.avgpool
        self.layer4 = base_model.layer4
        self.asize = util.infer_shape((in_channels, input_size[0], input_size[1]), self.activations)
        self.asizes = self._infer_shapes((in_channels, input_size[0], input_size[1]))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        c4 = self.layer4(x)

        x = self.avgpool(c4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def activations(self, x, return_c3=False):
        # No backprop for Resnet's layers
        x = Variable(x.data, volatile=True)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c4 = Variable(c4.data, volatile=False)  # enable downstream backprop
        if return_c3:
            c3 = Variable(c3.data, volatile=False)
            return c3, c4
        return c4

    def _infer_shapes(self, input_shape):
        x = Variable(util.cuda_as(th.ones(2, *input_shape), self))
        c3, c4 = self.activations(x, return_c3=True)
        return [c.size()[1:] for c in (c3, c4)]
