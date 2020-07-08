import numpy as np
import pickle
import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class CIFAR10SelfTrained(Dataset):

    def __init__(self, path, transform=None, target_transform=None):
        with open(path, "rb") as fd:
            self.dataset = pickle.load(fd)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        
        img = self.dataset["data"][index]
        target = self.dataset["extrapolated_targets"][index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.dataset["extrapolated_targets"])

