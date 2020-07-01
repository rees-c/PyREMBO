import numpy as np
import gpytorch
import torch
from sklearn.utils import check_random_state
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model

def synth_obj_func(x):
    # Example synthetic objective function
    return x ** 4 + 2 * x ** 3 - 12 * x ** 2 - 2 * x + 6


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
        raw_samples=512,  # used for intialization heuristic
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


    # # ---------------- manual optimization ----------------
    # training_iter = 50
    # # Use the adam optimizer
    # optimizer = torch.optim.Adam([
    #     {'params': model.parameters()},
    #     # Includes GaussianLikelihood parameters
    # ], lr=0.1)
    # for i in range(training_iter):
    #     # Zero gradients from previous iteration
    #     optimizer.zero_grad()
    #     # Output from model
    #     output = model(train_x)
    #     # Calc loss and backprop gradients
    #     loss = -mll(output, train_obj)
    #     loss.backward()
    #     # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
    #     #     i + 1, training_iter, loss.item(),
    #     #     model.covar_module.base_kernel.lengthscale.item(),
    #     #     model.likelihood.noise.item()
    #     # ))
    #     optimizer.step()


    return model





    # # MORE MANUAL METHOD...
    # if optimizer == "random+GD":
    #     for i in range(maxf):
    #         x_trial = random.uniform(size=boundaries.shape[0]) \
    #             * (boundaries[:, 1] - boundaries[:, 0]) \
    #             + boundaries[:, 0]
    #         f_trial = objective_function(x_trial)
    #
    #         # TODO: NEED OBJECTIVE FUNCTION GRADIENT
    #
    #         if f_trial < f_min:
    #             x_opt = x_trial
    #             f_min = f_trial