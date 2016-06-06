# Variational Auto-Encoder in MATLAB 

This is a re-implementation of
[*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114)
in MATLAB.

## Installation

### Data 

I use the MNIST from:
https://github.com/y0ast/VAE-Torch/tree/master/datasets.

### Toolbox

Please install my fork of
[*MatConvNet*](https://github.com/peiyunh/matconvnet), where I
implemented some new layers, including:

- `KLD.m`: handles forward and backward propagation of KL Divergence 
- `NLL.m`: handles forward and backward propagation of Negative
  Log-Likelihood (works for multi-variate Bernoulli distribution)
- `LB.m`: combine KLD and NLL into a lower bound
- `Sampler.m`: sampling operation
- `Tanh.m`: tanh non-linearity 
- `Split.m`: split one variable into multiple while keeping the same
  spatial size

## Usage

### Training

For *training*, please see `train_script.m` on how I trained models. I
implemented four stochastic gradient descent algorithms:

- SGD with momentum 
- ADAM
- ADAGRAD 
- RMSPROP

### Demo

For *demo*, I have four demo scripts for visualization under `demo/`,
which are: 

- `manifold_demo.m`: visualize the manifold of a 2d latent space in
  image space.
- `sample_demo.m`: sample from latent space and visualize in image
  space.
- `reconstruct_demo.m`: visualize a reconstructed version of an input
  image.
- `walk_demo.m`: randomly sample a list of images, and compare the
  morphing process done in both image space and latent space.

### More

To learn about how VAE works under the hood, refer to
[the original paper](https://arxiv.org/pdf/1312.6114v10.pdf) or my
[writeup](https://github.com/peiyunh/mat-vae/blob/master/writeup.pdf).
