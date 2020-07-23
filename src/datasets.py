import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
from src.lib.zipdata import ZipData
from src.lib.ds_imagenet import DSImageNet
from src.lib.cifar10_selftrained import CIFAR10SelfTrained


def get_dim(name):
    if name == "cifar":
        return 3 * 32 * 32
    if name == "svhn":
        return 3 * 32 * 32
    if name == "ds-imagenet":
        return 3 * 32 * 32
    if name == "mnist":
        return 28 * 28
    if name == "imagenet":
        return 3 * 224 * 224
    if name == "fashion":
        return 28 * 28
    if name == "cifar10selftrained":
        return 3 * 32 * 32

def get_num_labels(name):
    return 1000 if "imagenet" in name else 10

def get_normalization_shape(name):
    if name == "cifar":
        return (3, 1, 1)
    if name == "imagenet":
        return (3, 1, 1)
    if name == "ds-imagenet":
        return (3, 1, 1)
    if name == "svhn":
        return (3, 1, 1)
    if name == "mnist":
        return (1, 1, 1)
    if name == "fashion":
        return (1, 1, 1)
    if name == "cifar10selftrained":
        return (3, 1, 1)

def get_normalization_stats(name):
    if name == "cifar" or name == "cifar10selftrained":
        return {"mu": [0.4914, 0.4822, 0.4465], "sigma": [0.2023, 0.1994, 0.2010]}
    if name == "imagenet" or name == "ds-imagenet":
        return {"mu": [0.485, 0.456, 0.406], "sigma": [0.229, 0.224, 0.225]}
    if name == "svhn":
        return {"mu": [0.436, 0.442, 0.471], "sigma": [0.197, 0.200, 0.196]}
    if name == "mnist":
        return {"mu": [0.1307,], "sigma": [0.3081,]}
    if name == "fashion":
        return {"mu": [0.2849,], "sigma": [0.3516,]}

def get_dataset(name, split):

    if name == "cifar" and split == "train":
        return datasets.CIFAR10("./data/cifar_10", train=True, download=True,
                                transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                              transforms.RandomHorizontalFlip(),
                                                              transforms.ToTensor()]))
    if name == "cifar" and split == "test":
        return datasets.CIFAR10("./data/cifar_10", train=False, download=True,
                                transform=transforms.ToTensor())

    if name == "imagenet" and split == "train":
        return ZipData("/mnt/bucket/imagenet/train.zip",
                       "/mnt/bucket/imagenet/train_map.txt",
                       transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor()]))

    if name == "imagenet" and split == "test":
        return ZipData("/mnt/bucket/imagenet/val.zip",
                       "/mnt/bucket/imagenet/val_map.txt",
                       transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor()]))

    if name == "ds-imagenet" and split == "train":
        return DSImageNet("/mnt/bucket/downsampled_imagenet/", split="train",
                          transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor()]))

    if name == "ds-imagenet" and split == "test":
        return DSImageNet("/mnt/bucket/downsampled_imagenet/", split="test",
                          transform=transforms.ToTensor())

    if name == "mnist":
        return datasets.MNIST("./data/mnist", train=(split == "train"), download=True,
                              transform=transforms.ToTensor())
    if name == "fashion":
        return datasets.FashionMNIST("./data/fashion", train=(split == "train"), download=True,
                                     transform=transforms.ToTensor())
    if name == "svhn":
        return datasets.SVHN("./data/svhn", split=split, download=True,
                             transform=transforms.ToTensor())

    if name == "cifar10selftrained" and split == "train":
        return ConcatDataset([CIFAR10SelfTrained("/mnt/bucket/cifar10_selftrained/ti_500K_pseudo_labeled.pickle",
                                                 transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                               transforms.ToTensor()])),
                              datasets.CIFAR10("./data/cifar_10", train=True, download=True,
                                               transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                                            transforms.RandomHorizontalFlip(),
                                                                            transforms.ToTensor()])),
                              ])

    raise ValueError
