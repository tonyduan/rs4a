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
from src.noises import *
from src.datasets import get_dim


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dir", default="./ckpts", type=str)
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--show", action="store_true")
    argparser.add_argument("--adv", default=1, type=float)
    argparser.add_argument("--eps-max", default=5.0, type=float)
    argparser.add_argument("--fancy-markers", action="store_true")
    args = argparser.parse_args()
    args.adv = round(args.adv)

    markers = ["o", "D", "s"] if args.fancy_markers else True

    sns.set_context("notebook", rc={"lines.linewidth": 2})
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    dataset = args.dir.split("_")[0]
    experiment_names = list(filter(lambda s: os.path.isdir(args.dir + "/" + s), 
                                   os.listdir(args.dir)))

    df = defaultdict(list)
    eps_range = np.linspace(0, args.eps_max, 81)

    for experiment_name in experiment_names:

        save_path = f"{args.dir}/{experiment_name}"
        results = {}
        experiment_args = pickle.load(open(f"{args.dir}/{experiment_name}/args.pkl", "rb"))

        for k in ("preds", "labels", f"radius_l{str(args.adv)}", "acc_train"):
            results[k] = np.load(f"{save_path}/{k}.npy")

        noise = parse_noise_from_args(experiment_args, device="cpu", 
                                      dim=get_dim(experiment_args.dataset))

        top_1_preds = np.argmax(results["preds"], axis=1)
        top_1_acc_pred = (top_1_preds == results["labels"]).mean()

        if experiment_args.adversarial:
            noise_str = noise.plotstr() + f",$\\epsilon={experiment_args.eps}$"
        else:
            noise_str = noise.plotstr()

        for eps in eps_range:

            top_1_acc_cert = ((results[f"radius_l{str(args.adv)}"] >= eps) & \
                              (top_1_preds == results["labels"])).mean()
            df["experiment_name"].append(experiment_name)
            df["sigma"].append(noise.sigma)
            df["noise"].append(noise_str)
            df["eps"].append(eps)
            df["top_1_acc_train"].append(results["acc_train"][0])
            df["top_1_acc_cert"].append(top_1_acc_cert)
            df["top_1_acc_pred"].append(top_1_acc_pred)

    # save the experiment results
    df = pd.DataFrame(df) >> arrange(X.noise)
    df.to_csv(f"{args.dir}/results_{dataset}_l{str(args.adv)}.csv", index=False)

    if args.debug:
        breakpoint()

    # print top-1 certified accuracies for table in paper
    print(df >> mask(X.eps.isin((0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0))) \
             >> group_by(X.eps, X.noise) >> arrange(X.top_1_acc_cert, ascending=False) >> head(1))

    # plot clean training accuracy against certified accuracy at eps
#    tmp = df >> mask(X.eps == 0.25) >> arrange(X.noise)
#    plt.figure(figsize=(3, 2.8))
#    ax = sns.scatterplot(x="top_1_acc_train", y="top_1_acc_cert", hue="noise", style="noise",
#                         markers=markers,  size="sigma", data=tmp, legend="full")
#    handles, labels = ax.get_legend_handles_labels()
#    i = [i for i, t in enumerate(ax.legend_.texts) if t.get_text() == "sigma"][0]
#    ax.legend(handles[:i], labels[:i])
#    plt.plot(np.linspace(0.0, 1.0), np.linspace(0.0, 1.0), "--", color="gray")
#    plt.ylim((0.2, 1.0))
#    plt.xlim((0.2, 1.0))
#    plt.xlabel("Top-1 training accuracy")
#    plt.ylabel("Top-1 certified accuracy, $\epsilon$ = 0.25")
#    plt.tight_layout()
#    plt.savefig(f"{args.dir}/train_vs_certified.pdf")
#
#    tmp = df >> mask(X.eps.isin((0.25, 0.5, 0.75, 1.0))) >> \
#                mutate(tr=X.top_1_acc_train, cert=X.top_1_acc_cert)
#    fig = sns.relplot(data=tmp, kind="scatter", x="tr", y="cert",
#                      hue="noise", col="eps", col_wrap=2, aspect=1, height=3, size="sigma")
#    fig.map_dataframe(plt.plot, (plt.xlim()[0], plt.xlim()[1]), (plt.xlim()[0], plt.xlim()[1]), 'k--').set_axis_labels("tr", "cert").add_legend()
#
    # plot clean training and testing accuracy
    grouped = df >> group_by(X.experiment_name) \
                 >> mask(X.sigma <= 1.25) \
                 >> summarize(experiment_name=first(X.experiment_name),
                              noise=first(X.noise),
                              sigma=first(X.sigma),
                              top_1_acc_train=first(X.top_1_acc_train),
                              top_1_acc_pred=first(X.top_1_acc_pred))

    plt.figure(figsize=(6.5, 2.5))
    plt.subplot(1, 2, 1)
    sns.lineplot(x="sigma", y="top_1_acc_train", hue="noise", markers=markers, 
                 style="noise", data=grouped, alpha=1)
    plt.xlabel("$\sigma$")
    plt.ylabel("Top-1 training accuracy")
    plt.ylim((0, 1))
    plt.subplot(1, 2, 2)
    sns.lineplot(x="sigma", y="top_1_acc_pred", hue="noise", markers=markers, 
                 style="noise", data=grouped, alpha=1, legend=False)
    plt.xlabel("$\sigma$")
    plt.ylabel("Top-1 testing accuracy")
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.savefig(f"{args.dir}/train_test_accuracies.pdf")

    # plot certified accuracies
    selected = df >> mutate(certacc=X.top_1_acc_cert) 
    sns.relplot(x="eps", y="certacc", hue="noise", kind="line", col="sigma", 
                data=selected, height=2, aspect=1.5, col_wrap=2)
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.savefig(f"{args.dir}/per_sigma_l{str(args.adv)}.pdf")

    # plot top certified accuracy per epsilon, per type of noise
    grouped = df >> mask(X.noise != "Clean") \
                 >> group_by(X.eps, X.noise) \
                 >> arrange(X.top_1_acc_cert, ascending=False) \
                 >> summarize(top_1_acc_cert=first(X.top_1_acc_cert),
                              noise=first(X.noise))

    plt.figure(figsize=(3.0, 2.8))
    sns.lineplot(x="eps", y="top_1_acc_cert", data=grouped, hue="noise", style="noise")
    plt.ylim((0, 1))
    plt.xlabel(f"$\\ell_{str(args.adv)}$ radius")
    plt.ylabel("Top-1 certified accuracy")
    plt.tight_layout()
    plt.savefig(f"{args.dir}/certified_accuracies_l{str(args.adv)}.pdf", bbox_inches="tight")

    if args.show:
        plt.show()

