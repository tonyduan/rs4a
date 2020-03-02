import math
import warnings

import numpy as np
import scipy as sp
import scipy.special
import scipy.stats
import torch
from scipy.stats import beta, binom, gamma, laplace, norm, betaprime
import torch.distributions as D

from .utils import (atanh, diffmethod_table, get_radii_from_convex_table,
                    get_radii_from_table, lvsetmethod_table, make_or_load,
                    plexp, relu, sample_l1_sphere, sample_l2_sphere,
                    sample_linf_sphere, wfun)


class Noise(object):
    '''Parent class of all noise objects.
    Methods:
        _sigma: returns the standard deviation sqrt(E ||noise||^2_2) for scale
                factor lambda=1, used to calculate the appropriate sigma given
                lambda or vice versa
        plotstr: return string for label in plots
        tabstr: return string for use in name of serialized table of robust radii
        sample: return sample of noise added to input x (of the same shape)
        certify: given a 1D array of probability lower bounds, returns the
                 corresponding robust radii against a particular adversary
    Attributes:
        dim: dimension of noise
        sigma: standard deviation sqrt(E ||noise||^2_2)
        lambd: scale parameter
        device: cpu | cuda, or other torch devices
        __adv__: a list of all adversaries for which we have direct
                robust radii
    Note: 
        (1) In the certification process, if the adversary model is in __adv__ 
        (i.e. the list of threats for which we have direct robust radii), 
        we return the corresponding calculated radius. Otherwise, we calculate
        the robust radius for all threats in __adv__ and convert into the 
        given threat model at a factor d^(1/p-1/adv).
        (2) A noise may store numpy files in the tables/ directory.
    '''
    __adv__ = []

    def __init__(self, dim, sigma=None, lambd=None, device='cpu'):
        self.dim = dim
        self.device = device
        if lambd is None and sigma is not None:
            self.sigma = sigma
            self.lambd = self.get_lambd(sigma)
        elif sigma is None and lambd is not None:
            self.lambd = lambd
            self.sigma = self.get_sigma(lambd)
        else:
            raise ValueError('Please give exactly one of sigma or lambd')

    def _sigma(self):
        '''Calculates the sigma if lambd = 1
        '''
        raise NotImplementedError()
    def get_sigma(self, lambd=None):
        '''Calculates the sigma given lambd
        '''
        if lambd is None:
            lambd = self.lambd
        return lambd * self._sigma()

    def get_lambd(self, sigma=None):
        '''Calculates the lambd given sigma
        '''
        if sigma is None:
            sigma = self.sigma
        return sigma / self._sigma()

    def sample(self, x):
        '''Apply noise to x'''
        raise NotImplementedError()

    def certify(self, prob_lb, adv):
        raise NotImplementedError()

    def _certify_lp_convert(self, prob_lb, adv, warn=True):
        if adv in self.__adv__:
            cert = getattr(self, f'certify_l{adv}')
            return cert(prob_lb)
        else:
            r = {}
            for a in self.__adv__:
                cert = getattr(self, f'certify_l{a}')
                ppen = self.dim ** (1/a - 1/adv) if adv > a else 1
                r[a] = cert(prob_lb) / ppen
            if warn:                
                lpstr = ', '.join([f'l{p}' for p in self.__adv__])
                warnings.warn(f'No direct robustness guarantee for l{adv}; '
                              f'converting {lpstr} radii to l{adv}.')
            if len(r) == 1:
                return list(r.values())[0]
            else:
                radii = list(r.values())
                out = torch.max(radii.pop(), radii.pop())
                while len(radii) > 0:
                    c = radii.pop()
                    out = torch.max(out, c)
                return out

    def certify_l1(self, prob_lb):
        return self.certify(prob_lb, adv=1)

    def certify_l2(self, prob_lb):
        return self.certify(prob_lb, adv=2)

    def certify_linf(self, prob_lb):
        return self.certify(prob_lb, adv=np.inf)


class Clean(Noise):

    def __init__(self, dim, device='cpu'):
        super().__init__(device, None, sigma=1)

    def __str__(self):
        return "Clean"

    def sample(self, x):
        return x

    def _sigma(self):
        return 1


class Uniform(Noise):
    '''Uniform noise on [-lambda, lambda]^dim
    '''

    __adv__ = [1, np.inf]

    def __init__(self, dim, sigma=None, lambd=None, device='cpu'):
        super().__init__(dim, sigma, lambd, device)

    def __str__(self):
        return f"Uniform(dim={self.dim}, lambd={self.lambd}, sigma={self.sigma})"

    def plotstr(self):
        return 'Uniform'

    def tabstr(self, adv):
        return f'unif_{adv}_d{self.dim}'

    def _sigma(self):
        return 3 ** -0.5

    def sample(self, x):
        return (torch.rand_like(x, device=self.device) - 0.5) * 2 * self.lambd + x

    def certify(self, prob_lb, adv):
        return self._certify_lp_convert(prob_lb, adv, warn=True)

    def certify_l1(self, prob_lb):
        return 2 * self.lambd * (prob_lb - 0.5)

    def certify_linf(self, prob_lb):
        return 2 * self.lambd * (1 - (1.5 - prob_lb) ** (1 / self.dim))


class Gaussian(Noise):
    '''Isotropic Gaussian noise
    '''

    __adv__ = [2]

    def __init__(self, dim, sigma=None, lambd=None, device='cpu'):
        super().__init__(dim, sigma, lambd, device)
        self.norm_dist = D.Normal(loc=torch.tensor(0., device=device),
                                scale=torch.tensor(self.lambd, device=device))

    def __str__(self):
        return f"Gaussian(dim={self.dim}, lambd={self.lambd}, sigma={self.sigma})"
    
    def plotstr(self):
        return "Gaussian"

    def tabstr(self, adv):
        return f'gaussian_{adv}_d{self.dim}'

    def _sigma(self):
        return 1

    def sample(self, x):
        return torch.randn_like(x) * self.lambd + x

    def certify(self, prob_lb, adv):
        return self._certify_lp_convert(prob_lb, adv, warn=False)

    def certify_l2(self, prob_lb):
        return self.norm_dist.icdf(prob_lb)


class Laplace(Noise):
    '''Isotropic Laplace noise
    '''

    __adv__ = [1, np.inf]

    def __init__(self, dim, sigma=None, lambd=None, device='cpu'):
        super().__init__(dim, sigma, lambd, device)
        self.laplace_dist = D.Laplace(loc=torch.tensor(0.0, device=device),
                                    scale=torch.tensor(self.lambd, device=device))
        self.linf_radii = self.linf_rho = self._linf_table_info = None

    def __str__(self):
        return f"Laplace(dim={self.dim}, lambd={self.lambd}, sigma={self.sigma})"

    def plotstr(self):
        return "Laplace"

    def tabstr(self, adv):
        return f'laplace_{adv}_d{self.dim}'

    def _sigma(self):
        return 2 ** 0.5

    def sample(self, x):
        return self.laplace_dist.sample(x.shape) + x

    def certify(self, prob_lb, adv):
        return self._certify_lp_convert(prob_lb, adv)

    def certify_l1(self, prob_lb):
        return -self.lambd * (torch.log(2 * (1 - prob_lb)))

    def certify_linf(self, prob_lb, mode='approx',
                    inc=0.001, grid_type='radius', upper=3, save=True):
        '''Certify Laplace smoothing against linf adversary.
        There are two modes of certification: "approx" or "integrate".
        The latter computes a table of robust radii from the differential
        method and performs lookup during certification, and is guaranteed
        to be correct. But this table calculation takes a bit of overhead
        (though it's only done once, and the table will be saved for loading
        in the future).
        The former uses the following approximation which is highly accurate
        in high dimension:

            lambda * GaussianCDF(prob_lb) / d**0.5

        We verify the quality of this approximation in `test_noises.py`.
        By default, "approx" mode is used.
        '''
        if mode == 'approx':
            return self.lambd * D.Normal(0, 1).icdf(prob_lb) / self.dim ** 0.5
        elif mode == 'integrate':
            table_info = dict(inc=inc, grid_type=grid_type, upper=upper)
            if self.linf_rho is None or self._linf_table_info != table_info:
                self.make_linf_table(inc, grid_type, upper, save)
                self._table_info = table_info
            return self.lambd * get_radii_from_convex_table(
                            self.linf_rho, self.linf_radii, prob_lb
            )
        else:
            raise ValueError(f'Unrecognized mode "{mode}"')

    def Phi_linf(self, prob):
        def phi(c, d):
            return binom(d, 0.5).sf((c+d)/2)
        def phiinv(p, d):
            return 2 * binom(d, 0.5).isf(p) - d
        d = self.dim
        c = phiinv(prob, d)
        pp = phi(c, d)
        return c * (prob - pp) + d * phi(c - 1/2, d-1) - d * phi(c, d)

    def _certify_linf_integrate(self, rho):
        return sp.integrate.quad(lambda p: 1/self.Phi_linf(p),
                                1 - rho, 1/2)[0]
    def make_linf_table(self, inc=0.001, grid_type='radius', upper=3, save=True,
                                loc='tables'):
        '''Calculate or load a table of robust radii for linf adversary.
        First try to load a table under `loc` with the corresponding
        parameters. If this fails, calculate the table.
        Inputs:
            inc: grid increment (default: 0.001)
            grid_type: 'radius' | 'prob' (default: 'radius')
                In a `radius` grid, the probabilities rho are calculated as
                GaussianCDF([0, inc, 2 * inc, ..., upper - inc, upper]).
                In a `prob` grid, the probabilities rho are spaced out evenly
                in increments of `inc`
                    [1/2, 1/2 + inc, 1/2 + 2 * inc, ..., 1 - inc]
            upper: if `grid_type == 'radius'`, then the upper limit to the
                radius grid. (default: 3)
            save: whether to save the table computed. (Default: True)
            loc: the folder containing the table. (Default: ./tables)
        Outputs:
            None, but `self.table`, `self.table_rho`, `self.table_radii`
            are now defined.
        '''
        self.linf_rho, self.linf_radii = make_or_load(
                        self.tabstr('linf'), self._make_linf_table, inc=inc,
                        grid_type=grid_type, upper=upper, save=save, loc=loc)
        return self.linf_rho, self.linf_radii

    def _make_linf_table(self, inc=0.001, grid_type='radius', upper=3):
        return diffmethod_table(self.Phi_linf, f=norm.cdf,
                    inc=inc, grid_type=grid_type, upper=upper)


class Pareto(Noise):
    '''Pareto (i.e. power law) noise in each coordinate, iid.
    '''

    __adv__ = [1]

    def __init__(self, dim, sigma=None, lambd=None, a=3, device='cpu'):
        self.a = a
        super().__init__(dim, sigma, lambd, device)
        self.pareto_dist = D.Pareto(
            scale=torch.tensor(self.lambd, device=device, dtype=torch.float),
            alpha=torch.tensor(self.a, device=device, dtype=torch.float))

    def plotstr(self):
        return f"Pareto,a={self.a}"

    def tabstr(self, adv):
        return f'pareto_{adv}_d{self.dim}_a{self.a}'

    def __str__(self):
        return (f'Pareto(dim={self.dim}, a={self.a}, '
                f'lambd={self.lambd}, sigma={self.sigma})')

    def _sigma(self):
        a = self.a
        if a > 2:
            return (0.5 * (a - 1) * (a - 2)) ** -0.5
        else:
            return np.np.inf

    def sample(self, x):
        samples = self.pareto_dist.sample(x.shape) - self.lambd
        signs = torch.sign(torch.rand_like(x) - 0.5)
        return samples * signs + x

    def certify_l1(self, prob_lb):
        prob_lb = prob_lb.numpy()
        a = self.a
        radius = sp.special.hyp2f1(
                    1, a / (a + 1), a / (a + 1) + 1,
                    (2 * prob_lb - 1) ** (1 + 1 / a)
                ) * self.lambd * (2 * prob_lb - 1) / a
        return torch.tensor(radius, dtype=torch.float)

    def certify(self, prob_lb, adv):
        return self._certify_lp_convert(prob_lb, adv)
        

class UniformBall(Noise):
    '''Uniform distribution over the l2 ball'''

    __adv__ = [2]

    def __init__(self, dim, sigma=None, lambd=None, device='cpu'):
        super().__init__(dim, sigma, lambd, device)
        self.beta_dist = sp.stats.beta(0.5 * (self.dim + 1), 0.5 * (self.dim + 1))

    def plotstr(self):
        return "UniformBall"

    def tabstr(self, adv):
        return f'unifball_{adv}_d{self.dim}'

    def __str__(self):
        return f"UniformBall(dim={self.dim}, lambd={self.lambd}, sigma={self.sigma})"

    def _sigma(self):
        return (self.dim + 2) ** -0.5

    def sample(self, x):
        radius = torch.rand((len(x), 1), device=self.device) ** (1 / self.dim)
        radius *= self.lambd
        noise = torch.randn(x.shape, device=self.device).reshape(len(x), -1)
        noise = noise / torch.norm(noise, dim=1, p=2).unsqueeze(1) * radius
        return noise + x

    def certify_l2(self, prob_lb):
        radius = self.lambd * (
            2 - 4 * self.beta_dist.ppf(0.75 - 0.5 * prob_lb.numpy()))
        return torch.tensor(radius, dtype=torch.float)

    def certify(self, prob_lb, adv):
        return self._certify_lp_convert(prob_lb, adv=adv, warn=False)


class CubicalDist(Noise):

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.linf_rho = self.linf_radii = self._linf_table_info = None

    def certify_linf(self, prob_lb):
        return self.certify_linf_table(prob_lb)

    def certify_linf_table(self, prob_lb, inc=0.001, grid_type='radius',
                    upper=3, save=True):
        table_info = dict(inc=inc, grid_type=grid_type, upper=upper)
        if self.linf_rho is None or self._linf_table_info != table_info:
            # this defines self.l1_rho and self.l1_radii
            self.make_linf_table(inc, grid_type, upper, save)
            self._linf_table_info = table_info
        return self.lambd * get_radii_from_convex_table(
                        self.linf_rho, self.linf_radii, prob_lb)
    
    def Phi_linf(self, prob):
        raise NotImplementedError()

    def _certify_linf_integrate(self, rho):
        return sp.integrate.quad(lambda p: 1/self.Phi_linf(p), 1 - rho, 1/2)[0]

    def make_linf_table(self, inc=0.001, grid_type='radius', upper=3, save=True,
                            loc='tables'):
        '''Calculate or load a table of robust radii for linf adversary.
        First try to load a table under `loc` with the corresponding
        parameters. If this fails, calculate the table.
        Inputs:
            inc: grid increment (default: 0.001)
            grid_type: 'radius' | 'prob' (default: 'radius')
                In a `radius` grid, the probabilities rho are calculated as
                LaplaceCDF([0, inc, 2 * inc, ..., upper - inc, upper] * sqrt(2)).
                In a `prob` grid, the probabilities rho are spaced out evenly
                in increments of `inc`
                    [1/2, 1/2 + inc, 1/2 + 2 * inc, ..., 1 - inc]
            upper: if `grid_type == 'radius'`, then the upper limit to the
                radius grid. (default: 3)
            save: whether to save the table computed. (Default: True)
            loc: the folder containing the table. (Default: ./tables)
        Outputs:
            Defines and outputs self.linf_rho, self.linf_radii
        '''
        self.linf_rho, self.linf_radii = make_or_load(
                        self.tabstr('linf'), self._make_linf_table, inc=inc,
                        grid_type=grid_type, upper=upper, save=save, loc=loc)
        return self.linf_rho, self.linf_radii

    def _make_linf_table(self, inc=0.001, grid_type='radius', upper=3):
        return diffmethod_table(
            self.Phi_linf, f=lambda r: laplace.cdf(np.sqrt(2) * r),
            inc=inc, grid_type=grid_type, upper=upper)

    def certify(self, prob_lb, adv):
        return self._certify_lp_convert(prob_lb, adv)


class ExpInf(CubicalDist):
    r'''Noise of the form \|x\|_\infty^{-j} e^{-\|x/\lambda\|_\infty^k}
    '''

    __adv__ = [1, np.inf]

    def __init__(self, dim, sigma=None, lambd=None, k=1, j=0, device='cpu'):
        self.k = k
        self.j = j
        super().__init__(dim, sigma, lambd, device)
        if dim > 1:
            self.gamma_factor = dim / (dim - 1) * math.exp(
                math.lgamma((dim - j) / k) - math.lgamma((dim - j - 1) / k))
        elif j == 0:
            self.gamma_factor = math.exp(
                math.lgamma((dim + k) / k) - math.lgamma((dim + k - 1) / k))
        else:
            raise ValueError(
                f'ExpInf(dim={dim}, k={k}, j={j}) is not a distribution.')
        self.gamma_dist = D.Gamma(
            concentration=torch.tensor((dim - j) / k, device=device),
            rate=1)

    def plotstr(self):
        return f"ExpInf,k={self.k},j={self.j}"

    def tabstr(self, adv):
        return f'expinf_{adv}_d{self.dim}_k{self.k}_j{self.j}'

    def __str__(self):
        return (f"ExpInf(dim={self.dim}, k={self.k}, j={self.j}, "
                f"lambd={self.lambd}, sigma={self.sigma})")
    
    def _sigma(self):
        k = self.k
        j = self.j
        d = self.dim
        r2 = (d - 1) / 3 + 1
        return np.sqrt(r2 / d * (
            math.exp(math.lgamma((d + 2 - j) / k)
            - math.lgamma((d - j) / k))))

    def sample(self, x):
        radius = (self.gamma_dist.sample((len(x), 1))) ** (1 / self.k)
        noise = sample_linf_sphere(self.device, x.shape)
        return self.lambd * (noise * radius).view(x.shape) + x

    def certify_l1(self, prob_lb):
        '''
        Note that if `prob_lb > 1 - 1/self.dim`, then better radii
        are available (see paper), but when `self.dim` is large, like in CIFAR10
        or ImageNet, this almost never happens.
        '''
        return 2 * self.lambd * self.gamma_factor * (prob_lb - 0.5)

    def certify_linf(self, prob_lb):
        if self.j != 0:
            raise NotImplementedError()
        if self.k == 1:
            return self.lambd * torch.log(0.5 / (1 - prob_lb))
        else:
            return self.certify_linf_table(prob_lb)

    def certify_linf_table(self, *args, **kw):
        if self.j != 0:
            raise NotImplementedError()
        return super().certify_linf_table(*args, **kw)

    def Phi_linf(self, prob):
        k = self.k
        d = self.dim
        g = gamma((d + k - 1)/k).cdf
        ig = gamma(d/k).ppf
        return (1 - g(ig(1 - 2 * prob))) * k / 2 * \
            np.exp(math.lgamma((d+k-1)/k) -
                    math.lgamma(d/k))

    def certify(self, prob_lb, adv):
        return self._certify_lp_convert(prob_lb, adv)


class PowInf(CubicalDist):
    r'''Linf-based power law, with density of the form (1 + \|x\|_\infty)^{-a}'''

    __adv__ = [1, np.inf]

    def __init__(self, dim, sigma=None, lambd=None, a=None, device='cpu'):
        self.a = a
        if a is None:
            raise ValueError('Parameter `a` is required.')
        super().__init__(dim, sigma, lambd, device)
        self.beta_dist = betaprime(dim, a - self.dim)

    def plotstr(self):
        return f"PowInf,a={self.a}"

    def tabstr(self, adv):
        return f'powinf_{adv}_d{self.dim}_a{self.a}'

    def __str__(self):
        return (f"PowInf(dim={self.dim}, a={self.a}, "
                f"lambd={self.lambd}, sigma={self.sigma})")

    def _sigma(self):
        d = self.dim
        a = self.a
        r2 = (d - 1) / 3 + 1
        return np.sqrt(r2 * (d + 1) / (a - d - 1) / (a - d - 2))

    def sample(self, x):
        samples = self.beta_dist.rvs((len(x), 1))
        radius = torch.tensor(samples, dtype=torch.float, device=self.device)
        noise = sample_linf_sphere(self.device, x.shape)
        return (noise * radius * self.lambd).view(x.shape) + x

    def certify_l1(self, prob_lb):
        return self.lambd * 2 * self.dim / (self.a - self.dim) * (prob_lb - 0.5)

    def Phi_linf(self, prob):
        d = self.dim
        a = self.a
        g = betaprime(d, a + 1 - d).cdf
        ig = betaprime(d, a - d).ppf
        return g(ig(2 * prob)) * (a - d) / 2


class SphericalDist(Noise):
    
    __adv__ = [2]

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.l2_radii = self.l2_rho = self._l2_table_info = None

    def tabstr(self, adv):
        raise NotImplementedError()

    def certify(self, prob_lb, adv):
        return self._certify_lp_convert(prob_lb, adv, warn=False)

    def certify_l2(self, prob_lb, inc=0.01, upper=3, save=True):
        return self.certify_l2_table(prob_lb, inc, upper, save)

    def certify_l2_table(self, prob_lb, inc=0.01, upper=3, save=True):
        table_info = dict(inc=inc, upper=upper)
        if self.l2_rho is None or self._l2_table_info != table_info:
            self.make_l2_table(inc, upper, save)
            self._table_info = table_info
        return self.lambd * get_radii_from_convex_table(
                        self.l2_rho, self.l2_radii, prob_lb)

    def _pbig(self, *args, **kw):
        '''Compute the big measure of a Neyman-Pearson set with ratio e^t.
        This function assumes `self.lambd == 1`.
        Inputs:
            t: log(kappa)
            e: the l2 norm of perturbation
            mode: integrate | mc
            nsamples: number of samples when `mode == 'mc'`
        Outputs:
        '''
        raise NotImplementedError()

    def _psmall(self, t, e, mode='integrate', nsamples=1000):
        '''Compute the big measure of a Neyman-Pearson set with ratio e^t.
        This function assumes `self.lambd == 1`.
        Inputs:
            t: log(kappa)
            e: the l2 norm of perturbation
            mode: integrate | mc
            nsamples: number of samples when `mode == 'mc'`
        Outputs:
        '''
        return 1 - self._pbig(-t, e, mode=mode, nsamples=nsamples)

    def _make_l2_table(self, inc=0.01, upper=3, grid_type='radius'):
        return lvsetmethod_table(self._pbig, self._psmall, self._sigma(),
                                                    inc=inc, upper=upper)

    def make_l2_table(self, inc=0.01, upper=3, save=True, loc='tables'):
        '''Calculate or load a table of robust radii for l2 adversary.
        First try to load a table under `loc` with the corresponding
        parameters. If this fails, calculate the table using level set method.
        Inputs:
            inc: grid increment (default: 0.01)
            upper: if `grid_type == 'radius'`, then the upper limit to the
                radius grid. (default: 3)
            save: whether to save the table computed. (Default: True)
            loc: location of saved tables
        Outputs:
        '''
        self.l2_rho, self.l2_radii = make_or_load(
            self.tabstr('l2'), self._make_l2_table, inc=inc,
            upper=upper, save=save, loc=loc
        )


class Exp2(SphericalDist):
    r'''L2-based distribution of the form \|x\|_2^{-j} e^{\|x/\lambda\|_2^k}'''

    def __init__(self, dim, sigma=None, lambd=None, k=1, j=0, device='cpu'):
        self.k = k
        self.j = j
        super().__init__(dim, sigma, lambd, device)
        self.gamma_dist = D.Gamma(
            concentration=torch.tensor((dim - j) / k, device=device),
            rate=1)
    def plotstr(self):
        return f"Exp2,k={self.k},j={self.j}"

    def tabstr(self, adv):
        return f'exp2_{adv}_d{self.dim}_k{self.k}_j{self.j}'

    def __str__(self):
        return (f"Exp2(dim={self.dim}, k={self.k}, j={self.j}, "
                f"lambd={self.lambd}, sigma={self.sigma})")

    def _sigma(self):
        k = self.k
        j = self.j
        d = self.dim
        return np.sqrt(1 / d *
                    math.exp(math.lgamma((d + 2 - j) / k)
                            - math.lgamma((d - j) / k)
                        )
                    )

    def sample(self, x):
        radius = (self.gamma_dist.sample((len(x), 1))) ** (1 / self.k)
        noise = sample_l2_sphere(self.device, x.shape)
        return self.lambd * (noise * radius).view(x.shape) + x

    def certify_l2(self, prob_lb, mode='levelset',
                inc=0.01, upper=3, save=True):
        if self.k == 1 and self.j == 0:
            beta_dist = sp.stats.beta(0.5 * (self.dim - 1),
                                               0.5 * (self.dim - 1))
            radius = self.lambd * (self.dim - 1) * \
                atanh(1 - 2 * beta_dist.ppf(1 - prob_lb.numpy()))
            return torch.tensor(radius, dtype=torch.float)
        elif self.k == 2 and self.j == 0:
            norm_dist = D.Normal(loc=torch.tensor(0., device=self.device),
                                scale=torch.tensor(self.lambd / np.sqrt(2),
                                                    device=self.device))
            return norm_dist.icdf(prob_lb)
        elif mode == 'levelset':
            return self.certify_l2_table(prob_lb, inc, upper, save)

    def _pbig(self, t, e, mode='integrate', nsamples=1000):
        '''Compute the big measure of a Neyman-Pearson set with ratio e^t.
        This function assumes `self.lambd == 1`.
        Inputs:
            t: log(kappa)
            e: the l2 norm of perturbation
            mode: integrate | mc
            nsamples: number of samples when `mode == 'mc'`
        Outputs:
        '''
        d = self.dim
        k = self.k
        j = self.j
        ul = 10 * (d / k)
        if self.j == 0:
            def s(rpow):
                return relu(rpow - t)**(1/k)
            def integrand(rpow):
                return wfun(rpow**(1/k), s(rpow), e, d)
            if mode == 'integrate':
                # need to manually split integral because scipy.integrate
                # doesn't allow infinite upper limit when specifying
                # singularities through `points`
                return gamma(d/k).expect(integrand, points=[d/k-1], lb=0, ub=ul
                        )  + gamma(d/k).expect(integrand, lb=ul, ub=np.inf)
            elif mode == 'mc':
                rpow = gamma(d/k).rvs(size=nsamples)
                return np.mean(integrand(rpow))
            else:
                raise ValueError(f'Unrecognized mode: {mode}')
        else:
            def s(rpow):
                q = k/j * (rpow - t) + np.log(rpow) + np.log(k/j)
                p = relu(plexp(q, mode='lowerbound').real)
                s = (j/k * p)**(1/k)
                return s.real
            def integrand(rpow):
                return wfun(rpow**(1/k), s(rpow), e, d)
            if mode == 'integrate':
                # need to manually split integral because scipy.integrate
                # doesn't allow infinite upper limit when specifying
                # singularities through `points`
                return gamma(d/k - j/k).expect(
                        integrand, points=[d/k-j/k-1], lb=0, ub=ul) + \
                    gamma(d/k).expect(integrand, lb=ul, ub=np.inf)
            elif mode == 'mc':
                rpow = gamma(d/k - j/k).rvs(size=nsamples)
                return np.mean(integrand(rpow))
            else:
                raise ValueError(f'Unrecognized mode: {mode}')


class Pow2(SphericalDist):
    r'''L2-based distribution of the form (1 + \|x\|_2^k)^{-a}'''

    def __init__(self, dim, sigma=None, lambd=None, k=1, a=None, device='cpu'):
        self.k = k
        if a is None:
            self.a = dim + 10
        else:
            self.a = a
        super().__init__(dim, sigma, lambd, device)
        self.beta_dist = sp.stats.betaprime(dim / k, self.a - dim / k)
        self.beta_mode = (dim/k - 1) / (self.a - dim/k + 1)

    def plotstr(self):
        return f"Pow2,k={self.k},a={self.a}"

    def tabstr(self, adv):
        return f'pow2_{adv}_d{self.dim}_k{self.k}_a{self.a}'

    def __str__(self):
        return (f"Pow2(dim={self.dim}, k={self.k}, a={self.a}, "
                f"lambd={self.lambd}, sigma={self.sigma})")

    def _sigma(self):
        k = self.k
        a = self.a
        d = self.dim
        g = math.lgamma
        return np.exp(0.5 * (
                g((d+2)/k) + g(a - (d+2)/k) - g(d/k) - g(a - d/k) - np.log(d)
            ))

    def sample(self, x):
        samples = self.beta_dist.rvs((len(x), 1))
        radius = torch.tensor(samples**(1/self.k),
                    dtype=torch.float, device=self.device)
        noise = sample_l2_sphere(self.device, x.shape)
        return (self.lambd * radius * noise).view(x.shape) + x

    def _pbig(self, t, e, mode='integrate', nsamples=1000):
        d = self.dim
        k = self.k
        a = self.a
        def s(rpow):
            return relu((1 + rpow) * np.exp(-t/a) - 1)**(1/k)
        def integrand(rpow):
            return wfun(rpow**(1/k), s(rpow), e, d)
        if mode == 'integrate':
            return self.beta_dist.expect(integrand)
        elif mode == 'mc':
            rpow = self.beta_dist.rvs(size=nsamples)
            return np.mean(integrand(rpow))
        else:
            raise ValueError(f'Unrecognized mode: {mode}')


class Exp1(Noise):
    r'''L1-based distribution of the form e^{\|x/\lambda\|_1^k}'''

    __adv__ = [1]

    def __init__(self, dim, sigma=None, lambd=None, k=1, device='cpu'):
        self.k = k
        if k <= 0:
            raise ValueError(f'k must be positive: {k} received')
        super().__init__(dim, sigma, lambd, device)
        self.l1_radii = self.l1_rho = self._l1_table_info = None
        self.gamma_dist = D.Gamma(
            concentration=torch.tensor(dim / k, device=device),
            rate=1)

    def plotstr(self):
        return f"Exp1,k={self.k}"

    def __str__(self):
        return (f"Exp1(dim={self.dim}, k={self.k}, "
                f"lambd={self.lambd}, sigma={self.sigma})")

    def tabstr(self, adv):
        return f'exp1_{adv}_d{self.dim}_k{self.k}'

    def _sigma(self):
        k = self.k
        d = self.dim
        return np.sqrt(2 / d / (d+1) *
                    math.exp(math.lgamma((d + 2) / k)
                            - math.lgamma(d / k)
                        )
                    )

    def sample(self, x):
        radius = (self.gamma_dist.sample((len(x), 1))) ** (1 / self.k)
        radius *= self.lambd
        noises = sample_l1_sphere(self.device, x.shape)
        return noises * radius + x

    def certify(self, prob_lb, adv):
        return self._certify_lp_convert(prob_lb, adv, warn=True)

    def certify_l1(self, prob_lb):
        '''
        If `self.k==1`, then use the Laplace robust radii.
        Otherwise, construct/load table computed from differential method.
        '''
        if self.k == 1:
            # use the Laplace radii
            return -self.lambd * (torch.log(2 * (1 - prob_lb)))
        return self.certify_l1_table(prob_lb)

    def certify_l1_table(self, prob_lb, inc=0.001, grid_type='radius',
                    upper=3, save=True):
        table_info = dict(inc=inc, grid_type=grid_type, upper=upper)
        if self.l1_rho is None or self._l1_table_info != table_info:
            # this defines self.l1_rho and self.l1_radii
            self.make_l1_table(inc, grid_type, upper, save)
            self._l1_table_info = table_info
        return self.lambd * get_radii_from_convex_table(
                        self.l1_rho, self.l1_radii, prob_lb)

    def Phi_l1(self, prob):
        k = self.k
        d = self.dim
        g = gamma((d+k-1)/k).cdf
        ig = gamma(d/k).ppf
        if k >= 1:
            Psi = 1 - g(ig(1 - 2 * prob))
        elif k > 0:
            Psi = g(ig(2 * prob))
        else:
            raise ValueError(f'invalid k: {self.k}')
        return Psi * k / 2 * np.exp(math.lgamma((d+k-1)/k) - math.lgamma(d/k))

    def _certify_l1_integrate(self, rho):
        return sp.integrate.quad(lambda p: 1/self.Phi_l1(p), 1 - rho, 1/2)[0]

    def make_l1_table(self, inc=0.001, grid_type='radius', upper=3, save=True,
                            loc='tables'):
        '''Calculate or load a table of robust radii for l1 adversary.
        First try to load a table under `loc` with the corresponding
        parameters. If this fails, calculate the table.
        Inputs:
            inc: grid increment (default: 0.001)
            grid_type: 'radius' | 'prob' (default: 'radius')
                In a `radius` grid, the probabilities rho are calculated as
                LaplaceCDF([0, inc, 2 * inc, ..., upper - inc, upper] * sqrt(2)).
                In a `prob` grid, the probabilities rho are spaced out evenly
                in increments of `inc`
                    [1/2, 1/2 + inc, 1/2 + 2 * inc, ..., 1 - inc]
            upper: if `grid_type == 'radius'`, then the upper limit to the
                radius grid. (default: 3)
            save: whether to save the table computed. (Default: True)
            loc: the folder containing the table. (Default: ./tables)
        Outputs:
            Defines and outputs self.l1_rho, self.l1_radii
        '''
        self.l1_rho, self.l1_radii = make_or_load(
                        self.tabstr('l1'), self._make_l1_table, inc=inc,
                        grid_type=grid_type, upper=upper, save=save, loc=loc)
        return self.l1_rho, self.l1_radii

    def _make_l1_table(self, inc=0.001, grid_type='radius', upper=3):
        return diffmethod_table(
            self.Phi_l1, f=lambda r: laplace.cdf(np.sqrt(2) * r),
            inc=inc, grid_type=grid_type, upper=upper)

class Expp(Noise):
    r'''Noise of the form e^{\|x\|_p^p} = e^{\sum_i |x_i|^p}.'''

    __adv__ = [1]

    def __init__(self, dim, sigma=None, lambd=None, p=1, device='cpu'):
        self.p = p
        if p <= 0:
            raise ValueError(f'p must be positive: {p} received')
        super().__init__(dim, sigma, lambd, device)
        if p <= 1:
            self.l1_radii = self.l1_rho = self._l1_table_info = None
        self.gamma_dist = D.Gamma(
            concentration=torch.tensor(1 / p, device=device),
            rate=1)

    def _sigma(self):
        p = self.p
        return np.exp(0.5 * (math.lgamma(3/p) - math.lgamma(1/p)))

    def sample(self, x):
        p = self.p
        samples = self.gamma_dist.sample(x.shape)**(1/p)
        signs = torch.sign(torch.rand_like(x) - 0.5)
        return self.lambd * samples * signs + x

    def plotstr(self):
        return f"Exp{self.p},k={self.p}"

    def __str__(self):
        return (f"Expp(dim={self.dim}, p={self.p}, "
                f"lambd={self.lambd}, sigma={self.sigma})")

    def tabstr(self, adv):
        return f'exp{self.p}_{adv}_d{self.dim}_k{self.p}'

    def certify(self, prob_lb, adv):
        return self._certify_lp_convert(prob_lb, adv, warn=True)

    def certify_l1(self, prob_lb):
        p = self.p
        if p >= 1:
            return torch.from_numpy(
                    self.lambd * gamma(1/p).ppf(2 * prob_lb - 1)**(1/p))
        elif p == 0.5:
            c = gamma(1/p).ppf(2 * (1 - prob_lb))
            diln = lambda x: sp.special.spence(1-x)
            ec = np.exp(-c)
            # if c == np.inf, then prob_lb = 1/2 and the resulting radius is 0.
            c[c == np.inf] = 0
            _ec = 1 - ec
            # if c == 0, then c * log(1 - e^c) = 0
            _ec[c == 0] = 1
            clog = -c * np.log(_ec)
            return torch.from_numpy(2 * self.lambd * (clog + diln(ec)))
        else:
            return self.certify_l1_smallp_table(prob_lb)

    def _integrand_l1(self, c):
        '''This is only for the case self.p < 1.'''
        return 1/(np.exp(c ** self.p) - 1)

    def invphi(self, p0):
        return gamma(1/self.p).ppf(2 * p0)**(1/self.p)

    def _certify_l1_integrate2(self, rho):
        return sp.integrate.quad(self._integrand_l1, self.invphi(1-rho), np.inf)

    def _certify_l1_integrate(self, rho):
        return sp.integrate.quad(lambda p: 1/self.Phi_l1(p), 1 - rho, 1/2)

    def reduced_Phi_l1(self, prob):
        return 1 - np.exp(-abs(gamma(1/self.p).ppf(2 * prob)))

    def Phi_l1(self, prob):
        return self.reduced_Phi_l1(prob) / (2 * math.gamma(1 + 1/self.p))

    def certify_l1_smallp_table(self, prob_lb, rhomax=0.999, gridsize=3000,
                                save=True):
        table_info = dict(rhomax=rhomax, gridsize=gridsize)
        if self.l1_rho is None or self._l1_table_info != table_info:
            # this defines self.l1_rho and self.l1_radii
            self.make_l1_table_smallp(rhomax, gridsize, save)
            self._l1_table_info = table_info
        return self.lambd * get_radii_from_convex_table(
                        self.l1_rho, self.l1_radii, prob_lb)

    def make_l1_table_smallp(self, rhomax=0.999, gridsize=3000, save=True,
                            loc='tables'):
        '''Calculate or load a table of robust radii for l1 adversary.
        First try to load a table under `loc` with the corresponding
        parameters. If this fails, calculate the table.
        Inputs:
            rhomax: maximum probability lower bound to consider. (Default: 0.999)
            gridsize: approximate number of points in the robust radii table.
                (Default: 3000)
            save: whether to save the table computed. (Default: True)
            loc: the folder containing the table. (Default: ./tables)
        Outputs:
            Defines and outputs self.l1_rho, self.l1_radii
        '''
        from os.path import join
        basename = self.tabstr('l1') + f'_rhomax{rhomax}_gridsize{gridsize}'
        rho_fname = join(loc, basename + '_rho.npy')
        radii_fname = join(loc, basename + '_radii.npy')
        try:
            table_rho = np.load(rho_fname)
            table_radii = np.load(radii_fname)
            print('Found and loaded saved table: ' + basename)
        except FileNotFoundError:
            print('Making robust radii table: ' + basename)
            table_rho, table_radii = self._make_l1_table_smallp(rhomax, gridsize)
            if save:
                import os
                print('Saving robust radii table')
                os.makedirs(loc, exist_ok=True)
                np.save(rho_fname, table_rho)
                np.save(radii_fname, table_radii)
        self.l1_rho = table_rho
        self.l1_radii = table_radii
        return table_rho, table_radii

    def _make_l1_table_smallp(self, rhomax=0.999, gridsize=3000):
        if self.p > 1:
            raise ValueError(
            'certify_l1_table_smallp is only applicable when p <= 1, '
            f'but self.p = {self.p}')
        rmax = self._certify_l1_integrate(rhomax)[0]
        inc = rmax / gridsize
        table = {rhomax: rmax}
        lastrho = rhomax
        rho = rhomax - inc * self.Phi_l1(1 - rhomax)
        from tqdm import tqdm
        fmtstr = ('{l_bar}{bar}| {n:.3f}/{total:.3f} [{elapsed}<{remaining}, '
                    '{rate_fmt}{postfix}]')
        with tqdm(total=rho-1/2, bar_format=fmtstr) as pbar:
            while rho > 1/2:
                delta = sp.integrate.quad(lambda p: 1/self.Phi_l1(p),
                                        1 - lastrho, 1 - rho)[0]
                table[rho] = table[lastrho] - delta
                lastrho = rho
                drho = inc * self.Phi_l1(1 - rho)
                rho -= drho
                pbar.update(drho)
            table[1/2] = 0
        return (np.array(list(table.keys())[::-1]),
                np.array(list(table.values())[::-1]))