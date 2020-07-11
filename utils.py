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


class dict_to_tensor_IO():
    """
    Map back and forth between dictionary data and tensor data.

    Args:
        dict_data (dict)
            State point described with a key-value pairs.
    Returns:
        tensor_state_point ((1, d) torch.Tensor)
            A tensor version of dict_data, where 'd' is the number of
            keys/parameters/features/dimensions.

    """

    def __init__(self, dict_data=None):
        self.column_indexes_to_keys = {}

        # Create the mapping as a dictionary of column_index:key pairs
        if dict_data is not None:
            for index, (key, value) in enumerate(dict_data.items()):
                self.column_indexes_to_keys[index] = key
        else:
            raise AttributeError('A state point dictionary was not passed.')

    def map_dict_state_point_to_tensor(self, dict_data=None):
        """
        Convert state point 'dict_data' into a (1, d) tensor using the map,
        'column_indexes_to_keys', and then return the tensor.
        """
        if dict_data is not None:
            tensor_state_point = torch.Tensor()  # will make this (1, d)
            for index, key in self.column_indexes_to_keys.items():
                value = torch.Tensor([[dict_data[key]]])
                tensor_state_point = torch.cat([tensor_state_point, value],
                                               dim=1)
            return tensor_state_point
        else:
            raise AttributeError('A state point dictionary was not passed.')

    def map_tensor_state_point_to_dict(self, tensor_state_point=None):
        """
        Convert (1, d) state point 'tensor_data' into a dict using the map,
        'column_indexes_to_keys', and then return the dict.
        """
        dict_state_point = {}
        if (tensor_state_point is not None) or \
                tensor_state_point.shape[0] != 1 or \
                tensor_state_point.ndim != 2:

            for index, key in self.column_indexes_to_keys.items():
                value = tensor_state_point[0][index]
                dict_state_point[key] = float(value)

            return dict_state_point
        else:
            raise AttributeError('A (1,d) state point tensor was not passed.')
