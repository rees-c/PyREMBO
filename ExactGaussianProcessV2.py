"""
ISSUE WITH THIS VERSION IS THAT ADDING DATA DOES NOT REFIT HYPERPARAMETERS OF
THE GPYTORCH MODEL. ALSO, THE GPYTORCH FUNCTION, 'set_train_data' SEEMS TO
TURN 'self.train_inputs' INTO A TUPLE, WHICH IS NOT DESIRED (it should be a
tensor).
"""
from abc import ABC

import torch
import gpytorch

from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_model

from botorch.test_functions import Hartmann

# # use a GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype = torch.float


#  GPyTorch
class ExactGaussianProcess(SingleTaskGP):
    def __init__(self, train_x, train_y,
                 likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                 covar_module=gpytorch.kernels.ScaleKernel(
                     gpytorch.kernels.MaternKernel(nu=2.5))
                 ):
        """

        :param train_x: (n x d) torch.Tensor
        :param train_y: (n) torch.Tensor
        :param likelihood:
        :param covar_module: Default assumes that all dimensions of x are of the
            same scale. This assumption requires data preprocessing.
        """
        print("train_x.type(): {}".format(train_x.type()))
        print("train_x: {}".format(train_x))
        print("train_x.float(): {}".format(train_x.float()))
        print("train_y.type(): {}".format(train_y.type()))
        print("train_y: {}".format(train_y))
        print("train_y.float(): {}".format(train_y.float()))
        super().__init__(train_X=train_x.float(),
                         train_Y=train_y.float(),
                         likelihood=likelihood,
                         covar_module=covar_module)

    @property
    def train_x(self):
        return self.train_inputs

    @property
    def train_y(self):
        return self.train_targets

    # def forward(self, x):
    #     """
    #
    #     :param x: (n x d)
    #     :return fpreds (object): fpreds has the following attributes.
    #         f_mean = f_preds.mean (prediction)
    #         f_var = f_preds.variance (prediction uncertainty)
    #         f_covar = f_preds.covariance_matrix
    #         f_samples = f_preds.sample(sample_shape=torch.Size(1000,))
    #     """
    #     # gets called through subsubsubclass' __call__ method
    #
    #     mean_x = self.mean_module(x)
    #     covar_x = self.covar_module(x)
    #     fpreds = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    #     return fpreds

    def fit(self, train_x_, train_y_):
        """
        Fit the Gaussian Process to training data using the Adam optimizer on
        the marginal log likelihood.

        Code based on the following GPyTorch tutorial:
        https://gpytorch.readthedocs.io/en/latest/examples/01_Exact_GPs/Simple_GP_Regression.html#Training-the-model

        :param train_x_: torch.Tensor (n, d)
        :param train_y_: torch.Tensor (n, 1)
        :return:
        """
        print("train_x_: {}".format(train_x_))
        # Update self.train_x and self.train_y
        self.set_train_data(inputs=train_x_, targets=train_y_, strict=False)

        # Fit the model with optimal model hyperparameters
        training_iter = 50

        # Put the model into training mode
        self.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.parameters()},
            # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Output from model
            print("self.train_x: {}".format(self.train_x))

            output = self(self.train_x)

            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()

            print(
                'Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    self.covar_module.base_kernel.lengthscale.item(),
                    self.likelihood.noise.item()
                ))
            optimizer.step()
# # BoTorch
# def _get_and_fit_model(Xs, Ys, **kwargs):
#     train_X, train_Y = Xs[0], Ys[0]
#     model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
#     mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
#     model.train()
#
#     optimizer = SGD([{'params': model.parameters()}], lr=kwargs.get("lr"))
#     for epoch in range(kwargs.get("epochs")):
#         optimizer.zero_grad()
#         output = model(train_X)
#         loss = -mll(output, model.train_targets)
#         loss.backward()
#         optimizer.step()
#
#     return model