# Variational Auto-Encoder in MATLAB 

This is a re-implementation of
[*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114)
in MATLAB.

## Installation

*Data*: 

I use the MNIST from:
https://github.com/y0ast/VAE-Torch/tree/master/datasets.

*Toolbox*: 

Please install [*MatConvNet*](https://github.com/vlfeat/matconvnet). I
used 1.0-beta20.

## Usage

For *training*, please see `train_script.m` on how I trained models. I
implemented four stochastic gradient descent algorithms:

- SGD with momentum 

- ADAM

- ADAGRAD 

- RMSPROP

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
