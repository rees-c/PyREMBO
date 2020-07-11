import numpy as np
import gpytorch
import torch
from sklearn.utils import check_random_state
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model


def global_optimization(objective_function, boundaries, batch_size):
    """

    :param objective_function:
    :param boundaries (torch.Tensor): A `2 x d` tensor of lower and upper bounds
     for each column of `X`.
    :param batch_size (int): Number of candidates to return.
    :param optimizer:
    :param maxf:
    :param random:
    Returns:
        `(num_restarts) x q x d`-dim tensor of generated candidates
    """

    # # USING BOTORCH
    from botorch.optim import optimize_acqf

    # Boundaries must be (2 x d) for optimize_acqf to work
    if boundaries.shape[0] != 2:
        boundaries = boundaries.T

    # optimize
    candidates, _ = optimize_acqf(
        acq_function=objective_function,
        bounds=boundaries,
        q=batch_size,
        num_restarts=10,  # number of initial points for optimization
        raw_samples=512,  # used for initialization heuristic
        return_best_only=True  # only returns the best of the n_restarts random restarts
    )
    # what exactly is raw_samples?

    # Removes the 'candidates' variable from the computational graph
    new_x = candidates.detach()
    return new_x


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_fitted_model(train_x, train_obj, state_dict=None):
    # initialize and fit model
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj)

    # # initialize likelihood and model
    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # model = ExactGPModel(train_x, train_obj, likelihood)
    # model.train()
    # likelihood.train()

    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(train_x)
    fit_gpytorch_model(mll)

    return model
