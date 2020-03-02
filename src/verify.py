import torch
import os
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.attacks import *
from src.noises import *
from src.models import *
from src.datasets import get_dataset, get_num_labels


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda", type=str)
    argparser.add_argument("--batch-size", default=4, type=int),
    argparser.add_argument("--num-workers", default=os.cpu_count(), type=int)
    argparser.add_argument("--sample-size-pred", default=64, type=int)
    argparser.add_argument("--noise-batch-size", default=512, type=int)
    argparser.add_argument("--sigma", default=0.0, type=float)
    argparser.add_argument("--noise", default="Clean", type=str)
    argparser.add_argument("--k", default=None, type=int)
    argparser.add_argument("--j", default=None, type=int)
    argparser.add_argument("--a", default=None, type=int)
    argparser.add_argument("--lambd", default=None, type=float)
    argparser.add_argument("--adv", default=2, type=int)
    argparser.add_argument("--experiment-name", default="cifar", type=str)
    argparser.add_argument("--dataset", default="cifar", type=str)
    argparser.add_argument("--model", default="WideResNet", type=str)
    argparser.add_argument("--output-dir", type=str, default=os.getenv("PT_OUTPUT_DIR"))
    args = argparser.parse_args()

    test_dataset = get_dataset(args.dataset, "test")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, # todo: fix
                             num_workers=args.num_workers)

    save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    model = eval(args.model)(dataset=args.dataset, device=args.device)
    model.load_state_dict(torch.load(save_path))
    model.eval()

    noise = parse_noise_from_args(args, device=args.device, dim=get_dim(args.dataset))

    eps_range = (3.0, 2.0, 1.0, 0.5, 0.25)

    results = {f"preds_adv_{eps}": np.zeros((len(test_dataset), 10)) for eps in eps_range}

    for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):

        x, y = x.to(args.device), y.to(args.device)
        lower, upper = i * args.batch_size, (i + 1) * args.batch_size

        for eps in eps_range:
            x_adv, _ = pgd_attack_smooth(model, x, y, eps=eps, noise=noise, sample_size=128,
                                         steps=20, p=args.adv, clamp=(0, 1))
            preds_adv = smooth_predict_hard(model, x_adv, noise, args.sample_size_pred,
                                            args.noise_batch_size)
            results[f"preds_adv_{eps}"][lower:upper, :] = preds_adv.probs.data.cpu().numpy()
            assert ((x - x_adv).reshape(x.shape[0], -1).norm(dim=1, p=args.adv) <= eps + 1e-2).all()

    save_path = f"{args.output_dir}/{args.experiment_name}"
    for k, v in results.items():
        np.save(f"{save_path}/{k}.npy", v)

