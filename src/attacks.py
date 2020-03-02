import torch
import torch.nn as nn
from torch.autograd import grad
from src.smooth import *


def project_onto_ball(x, eps, p="inf"):
    """
    Note that projection onto inf-norm and 2-norm take O(d) time, and projection onto 1-norm
    takes O(dlogd) using the sorting-based algorithm given in [Duchi et al. 2008].
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    assert not torch.isnan(x).any()
    if p == "inf":
        x = x.clamp(-eps, eps)
    elif p == 2:
        x = x.renorm(p=2, dim=0, maxnorm=eps)
    elif p == 1:
        mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
        mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
        cumsum = torch.cumsum(mu, dim=1)
        arange = torch.arange(1, x.shape[1] + 1, device=x.device)
        rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
        theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
        proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
        x = mask * x + (1 - mask) * proj * torch.sign(x)
    else:
        raise ValueError("Can only project onto 1,2,inf norm balls.")
    return x.view(original_shape)

def pgd_attack(model, x, y, eps, steps=20, adv="inf", clamp=(0, 1)):
    """
    Attack a model with PGD.
    """
    step_size = 2 * eps / steps
    x.requires_grad = True
    x_orig = x.clone().detach()

    for _ in range(steps):
        loss = model.loss(x, y).mean()
        grads = grad(loss, x)[0].reshape(x.shape[0], -1)
        if adv == 1:
            keep_vals = torch.kthvalue(grads.abs(), k=grads.shape[1] * 15 // 16, dim=1).values
            grads[torch.abs(grads) < keep_vals.unsqueeze(1)] = 0
            grads = torch.sign(grads)
            grads_norm = torch.norm(grads, dim=1, p=1)
            grads = grads / (grads_norm.unsqueeze(1) + 1e-8)
        elif adv == 2:
            grads_norm = torch.norm(grads, dim=1, p=2)
            grads = grads / (grads_norm.unsqueeze(1) + 1e-8)
        elif adv == "inf":
            grads = torch.sign(grads)
        else:
            raise ValueError
        diff = x + step_size * grads.reshape(x.shape) - x_orig
        diff = project_onto_ball(diff, eps, adv)
        x = (x_orig + diff).clamp(*clamp)

    loss = model.loss(x, y).mean()
    x = x.detach()
    x.requires_grad = False
    return x, loss

def pgd_attack_smooth(model, x, y, eps, noise, sample_size, steps=20, adv="inf", clamp=(0, 1)):
    """
    Attack a smoothed model with PGD.
    """
    step_size = 2 * eps / steps
    x.requires_grad = True
    x_orig = x.clone().detach()
    rng = torch.cuda.get_rng_state_all()

    for _ in range(steps):
        torch.cuda.set_rng_state_all(rng)
        forecast = smooth_predict_soft(model, x, noise, sample_size)
        loss = -forecast.log_prob(y).mean()
        grads = grad(loss, x)[0].reshape(x.shape[0], -1)
        if adv == 1:
            keep_vals = torch.kthvalue(grads.abs(), k=grads.shape[1] * 15 // 16, dim=1).values
            grads[torch.abs(grads) < keep_vals.unsqueeze(1)] = 0
            grads = torch.sign(grads)
            grads_norm = torch.norm(grads, dim=1, p=1)
            grads = grads / (grads_norm.unsqueeze(1) + 1e-8)
        elif adv == 2:
            grads_norm = torch.norm(grads, dim=1, p=2)
            grads = grads / (grads_norm.unsqueeze(1) + 1e-8)
        elif adv == "inf":
            grads = torch.sign(grads)
        else:
            raise ValueError
        diff = x + step_size * grads.reshape(x.shape) - x_orig
        diff = project_onto_ball(diff, eps, adv)
        x = (x_orig + diff).clamp(*clamp)
#        forecast = smooth_predict_hard(model, x, noise, sample_size).probs
#        print(_, (torch.argmax(forecast, dim=1) == y).sum() / float(x.shape[0]),
#              diff.reshape(x.shape[0], -1).norm(dim=1, p=1).mean(),
#              diff.reshape(x.shape[0], -1).norm(dim=1, p=2).mean())

    torch.cuda.set_rng_state_all(rng)
    forecast = smooth_predict_soft(model, x, noise, sample_size)
    loss = -forecast.log_prob(y).mean()

    x = x.detach()
    x.requires_grad = False
    return x, loss

