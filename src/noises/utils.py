import math
from tqdm import tqdm

import numpy as np
import scipy as sp
import scipy.special
import scipy.stats
import torch
from scipy.stats import beta, binom, gamma, norm, laplace
from torch.distributions import (Beta, Dirichlet, Gamma, Laplace, Normal,
                                 Pareto, Uniform)


def atanh(x):
    return 0.5 * np.log((1 + x) / (1 - x))

def relu(x):
    if isinstance(x, np.ndarray):
        return np.maximum(x, 0, x)
    else:
        return max(x, 0)

def wfun(r, s, e, d):
    '''W function in the paper.
    Calculates the probability a point sampled from the surface of a ball
    of radius `r` centered at the origin is outside a ball of radius `s`
    with center `e` away from the origin.
    '''
    t = ((r+e)**2 - s**2)/(4*e*r)
    return beta((d-1)/2, (d-1)/2).cdf(t)

def plexp(z, mode='lowerbound'):
    '''Computes LambertW(e^z) numerically safely.
    For small value of z, we use `scipy.special.lambertw`.
    For large value of z, we apply the approximation

        z - log(z) < W(e^z) < z - log(z) - log(1 - log(z)/z).
    '''
    if np.isscalar(z):
        if z > 500:
            if mode == 'lowerbound':
                return z - np.log(z)
            elif mode == 'upperbound':
                return z - np.log(z) - np.log(1 - np.log(z) / z)
            else:
                raise ValueError(f'Unknown mode: {mode}')
        else:
            return sp.special.lambertw(np.exp(z))
    else:
        if mode == 'lowerbound':
            # print(z)
            u = z - np.log(z)
        elif mode == 'upperbound':
            u = z - np.log(z) - np.log(1 - np.log(z) / z)
        else:
            raise ValueError(f'Unknown mode: {mode}')
        w = sp.special.lambertw(np.exp(z))
        w[z > 500] = u[z > 500]
        return w


def sample_linf_sphere(device, shape):
    noise = (2 * torch.rand(shape, device=device) - 1
            ).reshape((shape[0], -1))
    sel_dims = torch.randint(noise.shape[1], size=(noise.shape[0],))
    idxs = torch.arange(0, noise.shape[0], dtype=torch.long)
    noise[idxs, sel_dims] = torch.sign(
        torch.rand(shape[0], device=device) - 0.5)
    return noise

def sample_l2_sphere(device, shape):
    '''Sample uniformly from the unit l2 sphere.
    Inputs:
        device: 'cpu' | 'cuda' | other torch devices
        shape: a pair (batchsize, dim)
    Outputs:
        matrix of shape `shape` such that each row is a sample.
    '''
    noises = torch.randn(shape)
    noises /= noises.norm(dim=1, keepdim=True)
    return noises

def sample_l1_sphere(device, shape):
    '''Sample uniformly from the unit l1 sphere, i.e. the cross polytope.
    Inputs:
        device: 'cpu' | 'cuda' | other torch devices
        shape: a pair (batchsize, dim)
    Outputs:
        matrix of shape `shape` such that each row is a sample.
    '''
    batchsize, dim = shape
    dirdist = Dirichlet(concentration=torch.ones(dim, device=device))
    noises = dirdist.sample([batchsize])
    signs = torch.sign(torch.rand_like(noises) - 0.5)
    return noises * signs


def get_radii_from_table(table_rho, table_radii, prob_lb):
    prob_lb = prob_lb.numpy()
    idxs = np.searchsorted(table_rho, prob_lb, 'right') - 1
    return torch.tensor(table_radii[idxs], dtype=torch.float)


def get_radii_from_convex_table(table_rho, table_radii, prob_lb):
    '''
    Assuming 1) radii is a convex function of rho and
    2) table_rho[0] = 1/2, table_radii[0] = 0.
    Uses the basic fact that if f is convex and a < b, then

        f'(b) >= (f(b) - f(a)) / (b - a).
    '''
    prob_lb = prob_lb.numpy()
    idxs = np.searchsorted(table_rho, prob_lb, 'right') - 1
    slope = (table_radii[idxs] - table_radii[idxs-1]) / (
        table_rho[idxs] - table_rho[idxs-1]
    )
    rad = table_radii[idxs] + slope * (prob_lb - table_rho[idxs])
    rad[idxs == 0] = 0
    return torch.tensor(rad, dtype=torch.float)


def diffmethod_table(Phi, inc, grid_type, upper, f):
    r'''Calculates a table of robust radii using the differential method.
    Given function Phi and the probability rho of correctly classifying an input
    perturbed by smoothing noise, the differential method gives

        \int_{1 - \rho}^{1/2} dp/\Phi(p)

    for the robust radius.
    Inputs:
        Phi: Phi function
        inc: grid increment (default: 0.001)
        grid_type: 'radius' | 'prob' (default: 'radius')
            In a `radius` grid, the probabilities rho are calculated as

                f([0, inc, 2 * inc, ..., upper - inc, upper]),

            where `f` and `upper` are additional inputs to this function.
            In a `prob` grid, the probabilities rho are spaced out evenly
            in increments of `inc`

                [1/2, 1/2 + inc, 1/2 + 2 * inc, ..., 1 - inc]
                
        upper: if `grid_type == 'radius'`, then the upper limit to the
            radius grid.
        f: the function used to determine the grid if `grid_type == 'radius'`
    Outputs:
        A Python dictionary `table` with

            table[rho] = radius

        for a grid of rho.
    '''
    table = {1/2: 0}
    lastrho = 1/2
    if grid_type == 'radius':
        rgrid = np.arange(inc, upper+inc, inc)
        grid = f(rgrid)
    elif grid_type == 'prob':
        grid = np.arange(1/2+inc, 1, inc)
    else:
        raise ValueError(f'Unknown grid_type {grid_type}')
    for rho in tqdm(grid):
        delta = sp.integrate.quad(lambda p: 1/Phi(p), 1 - rho, 1 - lastrho)[0]
        table[rho] = table[lastrho] + delta
        lastrho = rho
    return np.array(list(table.keys())), np.array(list(table.values()))


def lvsetmethod_table(get_pbig, get_psmall, sigma, inc=0.01, upper=3):
    '''Calculates a table of robust radii using the level set method.
    Inputs:
        get_pbig: function for computing the big measure of a Neyman-Pearson set.
        get_psmall: same, for the small measure of a Neyman-Pearson set.
        sigma: sqrt(E[\|noise\|^2_2/d])
        inc: radius increment of the table
        upper: upper limit of radius for the table.
    Outputs:
        table_rho, table_radii
    '''
    def find_NP_log_ratio(u, x0=0, bracket=(-100, 100)):
        return sp.optimize.root_scalar(
            lambda t: get_pbig(t, u) - 0.5, x0=x0, bracket=bracket)
    table = {0: {'radius': 0, 'rho': 1/2}}
    prv_root = 0
    for eps in tqdm(np.arange(inc, upper + inc, inc)):
        e = eps * sigma
        t = find_NP_log_ratio(e, prv_root)
        table[eps] = {
            't': t.root,
            'radius': e,
            'normalized_radius': eps,
            'converged': t.converged,
            'info': t
        }
        if t.converged:
            table[eps]['rho'] = 1 - get_psmall(t.root, e)
            prv_root = t.root
    return np.array([x['rho'] for x in table.values()]), \
            np.array([x['radius'] for x in table.values()])

def make_or_load(basename, make, inc=0.001, grid_type='radius', upper=3,
                save=True, loc='tables'):
    '''Calculate or load a table of robust radii.
    First try to load a table under `./tables/` with the corresponding
    parameters. If this fails, calculate the table.
    Inputs:
        Phi: Phi function
        inc: grid increment (default: 0.001)
        grid_type: 'radius' | 'prob' (default: 'radius')
            In a `radius` grid, the probabilities rho are calculated as

                f([0, inc, 2 * inc, ..., upper - inc, upper]),

            where `f` and `upper` are additional inputs to this function.
            In a `prob` grid, the probabilities rho are spaced out evenly
            in increments of `inc`

                [1/2, 1/2 + inc, 1/2 + 2 * inc, ..., 1 - inc]
                
        upper: if `grid_type == 'radius'`, then the upper limit to the
            radius grid.
        f: the function used to determine the grid if `grid_type == 'radius'`
    Outputs:
        table_rho, table_radii
    '''
    from os.path import join
    if grid_type == 'radius':
        basename += f'_inc{inc}_grid{grid_type}_upper{upper}'
    else:
        basename += f'_inc{inc}_grid{grid_type}'
    rho_fname = join(loc, basename + '_rho.npy')
    radii_fname = join(loc, basename + '_radii.npy')
    try:
        table_rho = np.load(rho_fname)
        table_radii = np.load(radii_fname)
        print('Found and loaded saved table: ' + basename)
    except FileNotFoundError:
        print('Making robust radii table: ' + basename)
        table_rho, table_radii = make(inc=inc, grid_type=grid_type, upper=upper)
        if save:
            import os
            print('Saving robust radii table')
            os.makedirs(loc, exist_ok=True)
            np.save(rho_fname, table_rho)
            np.save(radii_fname, table_radii)
    return table_rho, table_radii

