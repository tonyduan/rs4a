import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli
from torchvision import models as base_models
from src.datasets import *
from src.lib.wide_resnet import WideResNet as WideResNetBase
from src.lib.alexnet import AlexNet as AlexNetBase
from src.lib.lenet import LeNet as LeNetBase
from src.lib.classic_resnet import resnet110 as classic_resnet110


class Forecaster(nn.Module):

    def __init__(self, dataset, device):
        super().__init__()
        self.device = device
        self.norm = NormalizeLayer(get_normalization_shape(dataset), device,
                                   **get_normalization_stats(dataset))

    def forward(self, x):
        raise NotImplementedError

    def forecast(self, theta):
        return Categorical(logits=theta)

    def loss(self, x, y):
        forecast = self.forecast(self.forward(x))
        return -forecast.log_prob(y)


class ResNet(Forecaster):

    def __init__(self, dataset, device):
        super().__init__(dataset, device)
        if dataset == "imagenet":
            self.model = nn.DataParallel(base_models.resnet50(num_classes=1000))
        else:
            self.model = nn.DataParallel(classic_resnet110())
        self.norm = nn.DataParallel(self.norm)
        self.norm = self.norm.to(device)
        self.model = self.model.to(device)

    def forward(self, x):
        x = self.norm(x)
        return self.model(x)


class WideResNet(Forecaster):
    
    def __init__(self, dataset, device):
        super().__init__(dataset, device)
        self.model = nn.DataParallel(WideResNetBase(depth=40, widen_factor=2,
                                                    num_classes=get_num_labels(dataset)))
        self.norm = nn.DataParallel(self.norm)
        self.norm = self.norm.to(device)
        self.model = self.model.to(device)

    def forward(self, x):
        x = self.norm(x)
        return self.model(x)


class LinearModel(Forecaster):

    def __init__(self, dataset, device):
        super().__init__(dataset, device)
        self.model = nn.Linear(get_dim(dataset), get_num_labels(dataset))
        self.model = self.model.to(device)

    def forward(self, x):
        x = self.norm(x).view(x.shape[0], -1)
        return self.model(x)


class AlexNet(Forecaster):

    def __init__(self, dataset, device):
        super().__init__(dataset, device)
        self.model = AlexNetBase(get_num_labels(dataset), drop_rate=0.5)
        self.model = self.model.to(device)

    def forward(self, x):
        x = self.norm(x)
        return self.model(x)


class LeNet(Forecaster):

    def __init__(self, dataset, device):
        super().__init__(dataset, device)
        self.model = LeNetBase(get_normalization_shape(dataset)[0], get_num_labels(dataset))
        self.model = self.model.to(device)

    def forward(self, x):
        x = self.norm(x)
        return self.model(x)


class MLP(Forecaster):

    def __init__(self, dataset, device):
        super().__init__(dataset, device)
        self.model = nn.Sequential(
            nn.Linear(get_dim(dataset), 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, get_num_labels(dataset)))
        self.model = self.model.to(device)

    def forward(self, x):
        x = self.norm(x).view(x.shape[0], -1)
        return self.model(x)


class NormalizeLayer(nn.Module):
    """
    Normalizes across the first non-batch axis.

    Examples:
        (64, 3, 32, 32) [CIFAR] => normalizes across channels
        (64, 8) [UCI]  => normalizes across features
    """
    def __init__(self, dim, device, mu=None, sigma=None):
        super().__init__()
        self.dim = dim
        if mu and sigma:
            self.mu = nn.Parameter(torch.tensor(mu, device=device).reshape(dim), requires_grad=False)
            self.log_sig = nn.Parameter(torch.log(torch.tensor(sigma, device=device)).reshape(dim), requires_grad=False)
            self.initialized = True
        else:
            raise ValueError

    def forward(self, x):
        if not self.initialized:
            self.initialize_parameters(x)
            self.initialized = True
        return (x - self.mu) / torch.exp(self.log_sig)

    def initialize_parameters(self, x):
        with torch.no_grad():
            mu = x.view(x.shape[0], x.shape[1], -1).mean((0, 2))
            std = x.view(x.shape[0], x.shape[1], -1).std((0, 2))
            self.mu.copy_(mu.data.view(self.dim))
            self.log_sig.copy_(torch.log(std).data.view(self.dim))

