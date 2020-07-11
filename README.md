# PyREMBO

Python implementation of REMBO [1] built on BoTorch/GPyTorch/PyTorch, allowing
for batched sampling.

Some code adapted from
https://github.com/jmetzen/bayesian_optimization/blob/master/bayesian_optimization/bayesian_optimization.py

REMBO demonstration can be found in 'REMBO_demonstration.py'.
Vanilla BO demonstration in 'bayes_opt.py'.
utils.dict_to_tensor_IO demonstration in 'dict_to_tensor_IO_demonstration.py'.

References \
[1] Ziyu Wang and Masrour Zoghi and Frank Hutter and David Matheson and 
Nando de Freitas Bayesian Optimization in High Dimensions via Random 
Embeddings. Proceedings of the 23rd international joint conference
on Artificial Intelligence (IJCAI)

Work in progress:
-
- Implement interleaved REMBO
- Allow for initial data
- Use 'gpytorch.likelihoods.FixedNoiseGaussianLikelihood' to add a fixed,
known observation noise to the GP predictions
