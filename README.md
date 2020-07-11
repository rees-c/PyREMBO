# PyREMBO

Python implementation of REMBO [1] built on BoTorch/GPyTorch/PyTorch.

Some code adapted from
https://github.com/jmetzen/bayesian_optimization/blob/master/bayesian_optimization/bayesian_optimization.py

An example use case can be found in 'example_main.py'.

References \
[1] Ziyu Wang and Masrour Zoghi and Frank Hutter and David Matheson and 
Nando de Freitas Bayesian Optimization in High Dimensions via Random 
Embeddings. Proceedings of the 23rd international joint conference
on Artificial Intelligence (IJCAI)

Work in progress:
-
- Allow for initial data
- Use 'gpytorch.likelihoods.FixedNoiseGaussianLikelihood' to add a fixed,
known observation noise to the GP predictions
