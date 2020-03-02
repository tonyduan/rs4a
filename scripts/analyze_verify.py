import numpy as np
import pandas as pd
import os
import pickle
import matplotlib as mpl
import seaborn as sns
from argparse import ArgumentParser
from collections import defaultdict
from dfply import *
from matplotlib import pyplot as plt


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--experiment-name", default="cifar_uniform_05", type=str)
    argparser.add_argument("--dir", default="./ckpts", type=str)
    args = argparser.parse_args()

    sns.set_style("white")
    sns.set_palette("husl")

    df = defaultdict(list)
    eps_range = (3.0, 2.0, 1.0, 0.5, 0.25)

    save_path = f"{args.dir}/{args.experiment_name}"
    experiment_args = pickle.load(open(f"{args.dir}/{args.experiment_name}/args.pkl", "rb"))
    results = {}

    for k in ["preds_smooth", "radius_smooth", "labels"] + \
             [f"preds_adv_{eps}" for eps in eps_range]:
        results[k] = np.load(f"{save_path}/{k}.npy")

    top_1_preds_smooth = np.argmax(results["preds_smooth"], axis=1)

    for eps in eps_range:

        top_1_preds_adv = np.argmax(results[f"preds_adv_{eps}"], axis=1)
        top_1_acc_cert = ((results["radius_smooth"] >= eps) & \
                          (top_1_preds_smooth == results["labels"])).mean()
        top_1_acc_adv = (top_1_preds_adv == results["labels"]).mean()
        df["eps"].append(eps)
        df["top_1_acc_cert"].append(top_1_acc_cert)
        df["top_1_acc_adv"].append(top_1_acc_adv)

    breakpoint()
    df = pd.DataFrame(df) >> gather("type", "top_1_acc", ["top_1_acc_cert", "top_1_acc_adv"])

    plt.figure(figsize=(5, 3))
    sns.lineplot(x="eps", y="top_1_acc", hue="type", data=df)
    plt.title(args.experiment_name)
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.show()

