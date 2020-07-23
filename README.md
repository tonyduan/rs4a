# Randomized Smoothing of All Shapes and Sizes

Last update: July 2020.

---

Code to accompany our paper:

**Randomized Smoothing of All Shapes and Sizes**  
*Greg Yang\*, Tony Duan\*, J. Edward Hu, Hadi Salman, Ilya Razenshteyn, Jerry Li.*  
International Conference on Machine Learning (ICML), 2020 [[Paper]](https://arxiv.org/abs/2002.08118) [[Blog Post]](http://decentdescent.org/rs4a1.html)

Notably, we outperform existing provably <img alt="$\ell_1$" src="svgs/839a0dc412c4f8670dd1064e0d6d412f.svg" align="middle" width="13.40191379999999pt" height="22.831056599999986pt"/>-robust classifiers on ImageNet and CIFAR-10.

![Table of SOTA results.](svgs/table.png)

![Figure of SOTA results.](svgs/envelopes.png)

This library implements the algorithms in our paper for computing robust radii for different smoothing distributions against different adversaries; for example, distributions of the form <img alt="$e^{-\|x\|_\infty^k}$" src="svgs/e703845884313f30712bfc7262a5e65b.svg" align="middle" width="50.04031229999999pt" height="33.26775210000002pt"/> against <img alt="$\ell_1$" src="svgs/839a0dc412c4f8670dd1064e0d6d412f.svg" align="middle" width="13.40191379999999pt" height="22.831056599999986pt"/> adversary.

The following summarizes the (distribution, adversary) pairs covered here.

![Venn Diagram of Distributions and Adversaries.](svgs/DistributionVenn.png)

We can compare the certified robust radius each of these distributions implies at a fixed level of <img alt="$\hat\rho_\mathrm{lower}$" src="svgs/b52b48d8661f69776e1b6650998d5067.svg" align="middle" width="38.43053114999999pt" height="22.831056599999986pt"/>, the lower bound on the probability that the classifier returns the top class under noise. Here all noises are instantiated for CIFAR-10 dimensionality (<img alt="$d=3072$" src="svgs/bd5b313d1d74ae2fc57ddb870603d84b.svg" align="middle" width="63.35043164999999pt" height="22.831056599999986pt"/>) and normalized to variance <img alt="$\sigma^2 \triangleq \mathbb{E}[\|x\|_2^2]=1$" src="svgs/bb9e6385ceb6d4a2d83a5b51a3c870c9.svg" align="middle" width="122.71106759999998pt" height="30.137058600000014pt"/>. Note that the first two rows below certify for the <img alt="$\ell_1$" src="svgs/839a0dc412c4f8670dd1064e0d6d412f.svg" align="middle" width="13.40191379999999pt" height="22.831056599999986pt"/> adversary while the last row certifies for the <img alt="$\ell_2$" src="svgs/336fefe2418749fabf50594e52f7b776.svg" align="middle" width="13.40191379999999pt" height="22.831056599999986pt"/> adversary and the <img alt="$\ell_\infty$" src="svgs/44c65658d6cd134b1599c29b31949f77.svg" align="middle" width="19.95444989999999pt" height="22.831056599999986pt"/> adversary. For more details see our `tutorial.ipynb` notebook.

![Certified Robust Radii of Distributions](svgs/robust-radii.png)

## Getting Started

Clone our repository and install dependencies:

```shell
git clone https://github.com/tonyduan/rs4a.git
conda create --name rs4a python=3.6
conda activate rs4a
conda install numpy matplotlib pandas seaborn 
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install torchnet tqdm statsmodels dfply
```

## Experiments

To reproduce our SOTA <img alt="$\ell_1$" src="svgs/839a0dc412c4f8670dd1064e0d6d412f.svg" align="middle" width="13.40191379999999pt" height="22.831056599999986pt"/> results on CIFAR-10, we need to train models over 
<p align="center"><img alt="$$&#10;\sigma \in \{0.15, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,2.0,2.25, 2.5,2.75, 3.0,3.25,3.5\},&#10;$$" src="svgs/8244067f9118b85361c6645cc9f1c526.svg" align="middle" width="534.1843451999999pt" height="16.438356pt"/></p>
For each value, run the following:

```shell
python3 -m src.train
--model=WideResNet
--noise=Uniform
--sigma={sigma}
--experiment-name=cifar_uniform_{sigma}

python3 -m src.test
--model=WideResNet
--noise=Uniform
--sigma={sigma}
--experiment-name=cifar_uniform_{sigma}
--sample-size-cert=100000
--sample-size-pred=64
--noise-batch-size=512
```

The training script will train the model via data augmentation for the specified noise and level of sigma, and save the model checkpoint to a directory `ckpts/experiment_name`.

The testing script will load the model checkpoint from the `ckpts/experiment_name` directory, make predictions over the entire test set using the smoothed classifier, and certify the <img alt="$\ell_1, \ell_2,$" src="svgs/5dc1880e644c7b3a0e9fa954759762ea.svg" align="middle" width="40.319761349999986pt" height="22.831056599999986pt"/> and <img alt="$\ell_\infty$" src="svgs/44c65658d6cd134b1599c29b31949f77.svg" align="middle" width="19.95444989999999pt" height="22.831056599999986pt"/> robust radii of these predictions. Note that by default we make predictions with <img alt="$64$" src="svgs/ec90b4fe342a37de851db6db2b08d4f4.svg" align="middle" width="16.438418699999993pt" height="21.18721440000001pt"/> samples, certify with <img alt="$100,000$" src="svgs/e1085464f81e12de4a74d54d14eb5dc5.svg" align="middle" width="56.62113929999999pt" height="21.18721440000001pt"/> samples, and at a failure probability of <img alt="$\alpha=0.001$" src="svgs/1fa8048512f84790ef174f591d0cb851.svg" align="middle" width="69.93719039999998pt" height="21.18721440000001pt"/>.

To draw a comparison to the benchmark noises, re-run the above replacing `Uniform` with `Gaussian` and `Laplace`. Then to plot the figures and print the table of results (for <img alt="$\ell_1$" src="svgs/839a0dc412c4f8670dd1064e0d6d412f.svg" align="middle" width="13.40191379999999pt" height="22.831056599999986pt"/> adversary), run our analysis script:

```shell
python3 -m scripts.analyze --dir=ckpts --show --adv=1
```

Note that other noises will need to be instantiated with the appropriate arguments when the appropriate training/testing code is invoked. For example, if we want to sample noise <img alt="$\propto \|x\|_\infty^{-100}e^{-\|x\|_\infty^{10}}$" src="svgs/1418cce7d60743be1c545cd950367159.svg" align="middle" width="123.97880054999999pt" height="32.44583099999998pt"/>, we would run:

```shell
 python3 -m src.train
--noise=ExpInf
--k=10
--j=100
--sigma=0.5
--experiment-name=cifar_expinf_0.5
```

## Trained Models

Our pre-trained models are available. 

The following commands will download all models into the `pretrain/` directory. 

```shell
mkdir -p pretrain
wget --directory-prefix=pretrain http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_all.zip
unzip -d pretrain pretrain/cifar_all.zip
wget --directory-prefix=pretrain http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_all.zip
unzip -d pretrain pretrain/imagenet_all.zip
```

ImageNet (ResNet-50): [[All Models, 2.3 GB]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_all.zip)

- Sigma=0.25: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_025.pt) [[Gaussian]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_gaussian_025.pt) [[Laplace]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_laplace_025.pt)
- Sigma=0.5: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_050.pt) [[Gaussian]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_gaussian_050.pt) [[Laplace]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_laplace_050.pt)
- Sigma=0.75: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_075.pt) [[Gaussian]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_gaussian_075.pt) [[Laplace]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_laplace_075.pt)
- Sigma=1.0: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_100.pt) [[Gaussian]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_gaussian_100.pt) [[Laplace]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_laplace_100.pt)
- Sigma=1.25: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_125.pt) [[Gaussian]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_gaussian_125.pt) [[Laplace]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_laplace_125.pt)
- Sigma=1.5: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_150.pt)
- Sigma=1.75: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_175.pt)
- Sigma=2.0: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_200.pt)
- Sigma=2.25: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_225.pt)
- Sigma=2.5: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_250.pt)
- Sigma=2.75: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_275.pt)
- Sigma=3.0: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_300.pt)
- Sigma=3.25: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_325.pt)
- Sigma=3.5: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_350.pt)

CIFAR-10 (Wide ResNet 40-2): [[All Models,  226 MB]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_all.zip)

- Sigma=0.15: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_uniform_015.pt) [[Gaussian]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_gaussian_015.pt) [[Laplace]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_laplace_015.pt)
- Sigma=0.25: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_uniform_025.pt) [[Gaussian]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_gaussian_025.pt) [[Laplace]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_laplace_025.pt)
- Sigma=0.5: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_uniform_050.pt) [[Gaussian]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_gaussian_050.pt) [[Laplace]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_laplace_050.pt)
- Sigma=0.75: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_uniform_075.pt) [[Gaussian]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_gaussian_075.pt) [[Laplace]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_laplace_075.pt)
- Sigma=1.0: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_uniform_100.pt) [[Gaussian]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_gaussian_100.pt) [[Laplace]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_laplace_100.pt)
- Sigma=1.25: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_uniform_125.pt) [[Gaussian]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_gaussian_125.pt) [[Laplace]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_laplace_125.pt)
- Sigma=1.5: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_uniform_150.pt)
- Sigma=1.75: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_uniform_175.pt)
- Sigma=2.0: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_uniform_200.pt)
- Sigma=2.25: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_uniform_225.pt)
- Sigma=2.5: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_uniform_250.pt)
- Sigma=2.75: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_uniform_275.pt)
- Sigma=3.0: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_uniform_300.pt)
- Sigma=3.25: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_uniform_325.pt)
- Sigma=3.5: [[Uniform]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_uniform_350.pt)

By default the models above were trained with noise augmentation. We further improve upon our state-of-the-art certified accuracies using recent advances in training smoothed classifiers: (1) by using stability training (Li et al. NeurIPS 2019), and (2)  by leveraging additional data using (a) pre-training on downsampled ImageNet (Hendrycks et al. NeurIPS 2019) and (b) semi-supervised self-training with data from 80 Million Tiny Images (Carmon et al. 2019). Our improved models trained with these methods are released below.

ImageNet (ResNet 50):

- Stability training: [[All Models, 2.3 GB]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_stability.zip)

CIFAR-10 (Wide ResNet 40-2):

- Stability training: [[All Models, 234 MB]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_stability.zip)
- Stability training + pre-training: [[All Models, 236 MB]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_pretrained.zip)
- Stability training + semi-supervised learning: [[All Models, 235 MB]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/cifar_semisup.zip)

An example of pre-trained model usage is below. For more in depth example see our `tutorial.ipynb` notebook.

```python
from src.models import WideResNet
from src.noises import Uniform
from src.smooth import *

# load the model
model = WideResNet(dataset="cifar", device="cuda")
saved_dict = torch.load("pretrain/cifar_uniform_050.pt")
model.load_state_dict(saved_dict)
model.eval()

# instantiation of noise
noise = Uniform(device="cpu", dim=3072, sigma=0.5)

# training code, to generate samples
noisy_x = noise.sample(x)

# testing code, certify for L1 adversary
preds = smooth_predict_hard(model, x, noise, 64)
top_cats = preds.probs.argmax(dim=1)
prob_lb = certify_prob_lb(model, x, top_cats, 0.001, noise, 100000)
radius = noise.certify(prob_lb, adv=1)
```

## Repository

1. `ckpts/` is used to store experiment checkpoints and results.
2. `data/` is used to store image datasets.
4. `tables/` contains caches of pre-calculated tables of certified radii.
5. `src/` contains the main souce code.
6. `scripts/` contains the analysis and plotting code.

Within the `src/` directory, the most salient files are:

1. `train.py` is used to train models and save to `ckpts/`.
2. `test.py` is used to test and compute robust certificates for <img alt="$\ell_1,\ell_2,\ell_\infty$" src="svgs/8d2d1eabb21bb41807292151fe468472.svg" align="middle" width="63.01387124999998pt" height="22.831056599999986pt"/> adversaries.
3. `noises/test_noises.py` is  a unit test for the noises we include. Run the test with
  
	```python -m unittest src/noises/test_noises.py```
	
    Note that some tests are probabilistic and can fail occasionally.
    If so, rerun a few more times to make sure the failure is not persistent.

4. `noises/noises.py` is a library of noises derived for randomized smoothing.