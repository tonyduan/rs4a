import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import os
import itertools
from argparse import ArgumentParser
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from matplotlib import pyplot as plt
from tqdm import tqdm
from src.attacks import *
from src.noises import *
from src.models import *
from src.datasets import get_dataset


def get_final_layer_mlp(model, x):
    out = model.model[0](x.reshape(x.shape[0], -1))
    out = model.model[1](out)
    out = model.model[2](out)
    out = model.model[3](out)
    return out

def get_final_layer(model, x):
    out = model.model.conv1(x)
    out = model.model.block1(out)
    out = model.model.block2(out)
    out = model.model.block3(out)
    out = model.model.relu(model.model.bn1(out))
    out = F.avg_pool2d(out, 8)
    return out.view(-1, model.model.nChannels)

if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda:0", type=str)
    argparser.add_argument("--batch-size", default=4, type=int),
    argparser.add_argument("--num-workers", default=os.cpu_count(), type=int)
    argparser.add_argument("--sample-size", default=64, type=int)
    argparser.add_argument("--dataset", default="cifar", type=str)
    argparser.add_argument("--dataset-skip", default=20, type=int)
    argparser.add_argument("--model", default="ResNet", type=str)
    argparser.add_argument("--dir", type=str, default="cifar_snapshots")
    argparser.add_argument("--load", action="store_true")
    args = argparser.parse_args()

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    noises = ["Uniform", "Gaussian", "Laplace"]
    epochs = np.arange(1, 30, 1)

    test_dataset = get_dataset(args.dataset, "test")
    test_dataset = Subset(test_dataset, list(range(0, len(test_dataset), args.dataset_skip)))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                             num_workers=args.num_workers)

    results = defaultdict(list)

    for noise_str, epoch in itertools.product(noises, epochs):

        if args.load:
            break

        sigma = 1.0

        save_path = f"{args.dir}/cifar_{noise_str}_{sigma}/{epoch-1}/model_ckpt.torch"
        model = eval(args.model)(dataset=args.dataset, device=args.device)
        model.load_state_dict(torch.load(save_path))
        model.eval()

        noise = eval(noise_str)(sigma=sigma, device=args.device, p=2, dim=get_dim(args.dataset))

        for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):

            x, y = x.to(args.device), y.to(args.device)

            v = x.unsqueeze(1).expand((args.batch_size, args.sample_size, 3, 32, 32))
            v = v.reshape((-1, 3, 32, 32))
            noised = noise.sample(v)
            if args.model == "ResNet":
                rep_noisy = get_final_layer(model, noised)
            elif args.model == "MLP":
                rep_noisy = get_final_layer_mlp(model, noised)
            else:
                raise ValueError
            rep_noisy = rep_noisy.reshape(args.batch_size, -1, rep_noisy.shape[-1])

            top_cats = model(noised).reshape(args.batch_size, -1, 10).argmax(dim=2).mode(dim=1)
            top_cats = top_cats.values

            l2 = torch.stack([F.pdist(rep_i, p=2) for rep_i in rep_noisy]).mean(dim=1).data
            l1 = torch.stack([F.pdist(rep_i, p=1) for rep_i in rep_noisy]).mean(dim=1).data
            linf = torch.stack([F.pdist(rep_i, p=float("inf")) for rep_i in rep_noisy]).mean(dim=1).data

            results["acc"] += (y == top_cats).float().cpu().numpy().tolist()
            results["l1"] += l1.cpu().numpy().tolist()
            results["l2"] += l2.cpu().numpy().tolist()
            results["linf"] += linf.cpu().numpy().tolist()
            results["noise"] += args.batch_size * [noise_str]
            results["epoch"] += args.batch_size * [epoch]

    if args.load:
        results = pd.read_csv(f"{args.dir}/snapshots.csv")
    else:
        results = pd.DataFrame(results)
    results.to_csv(f"{args.dir}/snapshots.csv")

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    sns.lineplot(x="epoch", y="l2", hue="noise", data=results)
    plt.xlabel("Epoch")
    plt.ylabel("L2")
    plt.legend()
    plt.subplot(2, 2, 2)
    sns.lineplot(x="epoch", y="l1", hue="noise", data=results)
    plt.xlabel("Epoch")
    plt.ylabel("L1")
    plt.legend()
    plt.subplot(2, 2, 3)
    sns.lineplot(x="epoch", y="linf", hue="noise", data=results)
    plt.xlabel("Epoch")
    plt.ylabel("Linf")
    plt.legend()
    plt.subplot(2, 2, 4)
    sns.lineplot(x="epoch", y="acc", hue="noise", data=results)
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.ylim((0, 1))
    plt.legend()
    plt.show()

