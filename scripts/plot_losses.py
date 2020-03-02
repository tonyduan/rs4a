#
# Plot the training loss for each model trained.
#
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib as mpl
import seaborn as sns
from argparse import ArgumentParser
from dfply import *
from matplotlib import pyplot as plt
from src.noises import *
from src.datasets import get_dim


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dir", default="./ckpts", type=str)
    args = argparser.parse_args()

    dataset = args.dir.split("_")[0]
    experiment_names = list(filter(lambda s: os.path.isdir(args.dir + "/" + s),
                                   os.listdir(args.dir)))

    sns.set_style("white")
    sns.set_palette("husl")

    losses_df = pd.DataFrame({"noise": [], "sigma": [], "losses_train": [], "iter": []})

    for experiment_name in experiment_names:

        save_path = f"{args.dir}/{experiment_name}"
        experiment_args = pickle.load(open(f"{args.dir}/{experiment_name}/args.pkl", "rb"))
        results = {}

        for k in ("losses_train",):
            results[k] = np.load(f"{save_path}/{k}.npy")

        noise = parse_noise_from_args(experiment_args, device="cpu",
                                      dim=get_dim(experiment_args.dataset))

        losses_df >>= bind_rows(pd.DataFrame({
            "experiment_name": experiment_name,
            "noise": noise.plotstr(),
            "sigma": experiment_args.sigma,
            "losses": results["losses_train"],
            "iter": np.arange(len(results["losses_train"]))}))

    # show training curves
    sns.relplot(x="iter", y="losses", hue="noise", data=losses_df, col="sigma",
                col_wrap=2, kind="line", height=1.5, aspect=3.5, alpha=0.5)
    plt.tight_layout()
    plt.show()

