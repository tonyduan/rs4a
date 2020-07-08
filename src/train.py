import logging
import pathlib
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
from torchnet import meter
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models import *
from src.noises import *
from src.smooth import *
from src.attacks import pgd_attack_smooth
from src.datasets import get_dataset, get_dim


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda", type=str)
    argparser.add_argument("--lr", default=0.1, type=float)
    argparser.add_argument("--batch-size", default=64, type=int)
    argparser.add_argument("--num-workers", default=min(os.cpu_count(), 8), type=int)
    argparser.add_argument("--num-epochs", default=120, type=int)
    argparser.add_argument("--print-every", default=20, type=int)
    argparser.add_argument("--save-every", default=50, type=int)
    argparser.add_argument("--experiment-name", default="cifar", type=str)
    argparser.add_argument("--noise", default="Clean", type=str)
    argparser.add_argument("--sigma", default=None, type=float)
    argparser.add_argument("--adv", default=2, type=int)
    argparser.add_argument("--eps", default=0.0, type=float)
    argparser.add_argument("--k", default=None, type=int)
    argparser.add_argument("--j", default=None, type=int)
    argparser.add_argument("--a", default=None, type=int)
    argparser.add_argument("--lambd", default=None, type=float)
    argparser.add_argument("--model", default="WideResNet", type=str)
    argparser.add_argument("--dataset", default="cifar", type=str)
    argparser.add_argument("--adversarial", action="store_true")
    argparser.add_argument("--stability", action="store_true")
    argparser.add_argument("--direct", action="store_true")
    argparser.add_argument("--save-path", type=str, default=None)
    argparser.add_argument('--output-dir', type=str, default=os.getenv("PT_OUTPUT_DIR"))
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = eval(args.model)(dataset=args.dataset, device=args.device)

    # for fine-tuning a pre-trained model, we strip out the last fc layer
    if args.save_path:
        saved_dict = torch.load(args.save_path)
        del saved_dict["model.module.fc.weight"]
        del saved_dict["model.module.fc.bias"]
        model.load_state_dict(saved_dict, strict=False)

    model.train()

    train_loader = DataLoader(get_dataset(args.dataset, "train"),
                              shuffle=True,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=False)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=1e-4,
                          nesterov=True)
    annealer = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    loss_meter = meter.AverageValueMeter()
    time_meter = meter.TimeMeter(unit=False)

    noise = parse_noise_from_args(args, device=args.device, dim=get_dim(args.dataset))

    train_losses = []

    for epoch in range(args.num_epochs):

        for i, (x, y) in enumerate(train_loader):

            x, y = x.to(args.device), y.to(args.device)

            if args.adversarial:
                eps = min(args.eps, epoch * args.eps / (args.num_epochs // 2))
                x, loss = pgd_attack_smooth(model, x, y, eps, noise, sample_size=4, adv=args.adv)
            elif args.stability:
                x_tilde = noise.sample(x.view(len(x), -1)).view(x.shape)
                x = noise.sample(x.view(len(x), -1)).view(x.shape)
            elif not args.direct:
                x = noise.sample(x.view(len(x), -1)).view(x.shape)

            if args.direct:
                loss = -direct_train_log_lik(model, x, y, noise, sample_size=16).mean()
            elif args.stability:
                pred_x = model.forecast(model.forward(x))
                pred_x_tilde = model.forecast(model.forward(x_tilde))
                loss = -pred_x.log_prob(y) + 6.0 * torch.distributions.kl_divergence(pred_x, pred_x_tilde)
                loss = loss.mean()
            elif not args.adversarial:
                loss = model.loss(x, y).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.cpu().data.numpy(), n=1)

            if i % args.print_every == 0:
                logger.info(f"Epoch: {epoch}\t"
                            f"Itr: {i} / {len(train_loader)}\t"
                            f"Loss: {loss_meter.value()[0]:.2f}\t"
                            f"Mins: {(time_meter.value() / 60):.2f}\t"
                            f"Experiment: {args.experiment_name}")
                train_losses.append(loss_meter.value()[0])
                loss_meter.reset()

        if (epoch + 1) % args.save_every == 0:
            save_path = f"{args.output_dir}/{args.experiment_name}/{epoch}/"
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/model_ckpt.torch")

        annealer.step()

    pathlib.Path(f"{args.output_dir}/{args.experiment_name}").mkdir(parents=True, exist_ok=True)
    save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    torch.save(model.state_dict(), save_path)
    args_path = f"{args.output_dir}/{args.experiment_name}/args.pkl"
    pickle.dump(args, open(args_path, "wb"))
    save_path = f"{args.output_dir}/{args.experiment_name}/losses_train.npy"
    np.save(save_path, np.array(train_losses))

