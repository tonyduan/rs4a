import unittest
import numpy as np
import tqdm
import torch
import src.noises.noises as noises

class TestSigma(unittest.TestCase):

    def test_sigma(self):
        rel_tol = 1e-2
        nsamples = int(1e4)
        dim = 3 * 32 * 32
        dev = 'cpu'
        configs = [
            dict(noise=noises.Uniform),
            dict(noise=noises.Gaussian),
            dict(noise=noises.Laplace),
            dict(noise=noises.UniformBall),
        ]
        for a in [3, 10, 100, 1000]:
            configs.append(
                dict(noise=noises.Pareto, a=a)
            )
        for k in [1, 2, 10, 50]:
            for j in [0, 1, 10, 100, 1000]:
                configs.append(
                    dict(noise=noises.ExpInf, k=k, j=j)
                )
        for k in [1, 2, 10, 20]:
            configs.append(
                dict(noise=noises.Exp1, k=k)
            )
        for a in [10, 100, 1000]:
            configs.append(
                dict(noise=noises.PowInf, a=a+dim)
            )
        for k in [1, 2, 10, 100]:
            for j in [0, 1, 10, 100, 1000]:
                configs.append(
                    dict(noise=noises.Exp2, k=k, j=j)
                )
        for k in [1, 2, 10, 100]:
            for a in [10, 100, 1000]:
                a = (dim + a) / k
                configs.append(
                    dict(noise=noises.Pow2, k=k, a=a)
                )
        for p in [0.2, 0.5, 1, 2, 4, 8]:
            configs.append(dict(noise=noises.Expp, p=p))
        for c in tqdm.tqdm(configs):
            c['device'] = dev
            c['dim'] = dim
            c['sigma'] = 1
            with self.subTest(config=dict(c)):
                noisecls = c.pop('noise')
                noise = noisecls(**c)
                samples = noise.sample(torch.zeros(nsamples, dim))
                self.assertEqual(samples.shape, torch.Size((nsamples, dim)))
                emp_sigma = samples.std()
                self.assertAlmostEqual(emp_sigma, noise.sigma,
                                       delta=rel_tol * emp_sigma)

class TestRadii(unittest.TestCase):

    def test_laplace_linf_radii(self):
        '''Test that the "approx" and "integrate" modes of linf certification
        for Laplace agree with each other.'''
        noise = noises.Laplace(3*32*32, sigma=1)
        cert1 = noise.certify_linf(torch.arange(0.5, 1, 0.01))
        cert2 = noise.certify_linf(torch.arange(0.5, 1, 0.01), 'integrate')
        self.assertTrue(np.allclose(cert1, cert2, rtol=1e-3))

    def test_exp2_l2_radii(self):
        r'''Test that for exp(-\|x\|_2), the differential and level set methods
        obtain similar robust radii.'''
        rs = torch.arange(0.5, 1, 0.01)
        with self.subTest(name='Exp2 test, k=1, j=0'):
            noise = noises.Exp2(3*32*32, sigma=1)
            cert1 = noise.certify_l2(rs)
            cert2 = noise.certify_l2_table(rs)
            self.assertTrue(np.allclose(cert1, cert2, rtol=1e-2))
        with self.subTest(name='Exp2 test, k=2, j=0'):
            noise = noises.Exp2(3*32*32, sigma=1, k=2)
            cert1 = noise.certify_l2(rs)
            cert2 = noise.certify_l2_table(rs)
            self.assertTrue(np.allclose(cert1, cert2, rtol=1e-3))

    def test_exp1_l1_radii(self):
        r'''Test that for exp(-\|x\|_1), the laplace and differential method
        table certification match.'''
        rs = torch.arange(0.5, 1, 0.01)
        noise1 = noises.Laplace(3*32*32, sigma=1)
        noise2 = noises.Exp1(3*32*32, sigma=1, k=1)
        cert1 = noise1.certify_l1(rs)
        cert2 = noise2.certify_l1_table(rs)
        self.assertTrue(np.allclose(cert1, cert2, rtol=1e-3))

    def test_expinf_linf_radii(self):
        r'''Test that ExpInf linf radii for k=1 matches known symbolic radii.'''
        rs = torch.arange(0.5, 1, 0.01)
        noise = noises.ExpInf(3*32*32, sigma=1, k=1)
        cert1 = noise.certify_linf(rs)
        cert2 = noise.certify_linf_table(rs)
        self.assertTrue(np.allclose(cert1, cert2, rtol=1e-3))

    def test_expp_largep_radii(self):
        r'''Test that Expp with p=1 and p=2 recovers Laplace and Gaussian.'''
        rs = torch.arange(0.5, 1, 0.01)
        with self.subTest(name='large p: Expp(p=1) vs Laplace'):
            noisep = noises.Expp(3*32*32, sigma=1, p=1)
            noise1 = noises.Laplace(3*32*32, sigma=1)
            cert1 = noisep.certify_l1(rs)
            cert2 = noise1.certify_l1(rs)
            self.assertTrue(np.allclose(cert1, cert2, rtol=1e-3))

        with self.subTest(name='large p: Expp(p=2) vs Gaussian'):
            noisep = noises.Expp(3*32*32, sigma=1, p=2)
            noise2 = noises.Gaussian(3*32*32, sigma=1)
            cert1 = noisep.certify_l1(rs)
            cert2 = noise2.certify_l1(rs)
            self.assertTrue(np.allclose(cert1, cert2, rtol=1e-3))

    def test_expp_smallp_radii(self):
        r'''Test the log convex* radii for Expp, p = 1, recovers Laplace,
        and for p = 0.5, recovers analytic expression.'''
        rs = torch.arange(0.5, 1, 0.01)
        with self.subTest(name='small p: Expp(p=1) vs Laplace'):
            noisep = noises.Expp(3*32*32, sigma=1, p=1)
            noise1 = noises.Laplace(3*32*32, sigma=1)
            cert1 = noisep.certify_l1_smallp_table(rs)
            cert2 = noise1.certify_l1(rs)
            self.assertTrue(np.allclose(cert1, cert2, rtol=1e-3))

        with self.subTest(name='small p: Expp(p=1/2), table vs closed form'):
            noise = noises.Expp(3*32*32, sigma=1, p=0.5)
            # analytic expression
            cert1 = noise.certify_l1(rs)
            # table
            cert2 = noise.certify_l1_smallp_table(rs)
            self.assertTrue(np.allclose(cert1, cert2, rtol=1e-3))

    def test_table_load(self):
        dim = 3 * 32 * 32
        dev = 'cpu'
        configs = []
        for k in [2, 3, 4]:
            configs.append(
                dict(noise=noises.Exp1, k=k, adv=1)
            )
        for k in [2]:
            for j in [2048, 3064, 3068, 3071]:
                configs.append(
                    dict(noise=noises.Exp2, k=k, j=j, adv=2)
                )
        for k in [2]:
            for a in [1538, 1540, 1544]:
                configs.append(
                    dict(noise=noises.Pow2, k=k, a=a, adv=2)
                )
        for k in [1, 2, 4, 8]:
            configs.append(
                dict(noise=noises.ExpInf, k=k, j=0, adv=np.inf)
            )
        for a in [4, 16, 32, 128]:
            configs.append(
                dict(noise=noises.PowInf, a=dim+a, adv=np.inf)
            )
        for p in [0.2, 0.5]:
            configs.append(dict(noise=noises.Expp, p=p, adv=1))
        rhos = torch.arange(0.5, 1, 0.01)
        for c in tqdm.tqdm(configs):
            c['device'] = dev
            c['dim'] = dim
            c['sigma'] = 1
            with self.subTest(config=dict(c)):
                noisecls = c.pop('noise')
                adv = c.pop('adv')
                noise = noisecls(**c)
                noise.certify(rhos, adv=adv)

if __name__ == '__main__':
    unittest.main()