from abc import ABC

import torch
import gpytorch

from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_model

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

        :param train_x: (n, d) torch.Tensor
        :param train_y: (n, 1) torch.Tensor
        :param likelihood:
        :param covar_module: Default assumes that all dimensions of x are of the
            same scale. This assumption requires data preprocessing.
        """
        train_X = train_x.float()
        train_Y = train_y.float()

        super().__init__(train_X=train_X,
                         train_Y=train_Y,
                         likelihood=likelihood,
                         covar_module=covar_module)

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
        the marginal log likelihood. (refits the model hyperparameters)

        Code based on the following GPyTorch tutorial:
        https://gpytorch.readthedocs.io/en/latest/examples/01_Exact_GPs/Simple_GP_Regression.html#Training-the-model

        :param train_x_: torch.Tensor (n, d)
        :param train_y_: torch.Tensor (n, 1)
        :return:
        """
        train_X = train_x_.float()
        train_Y = train_y_.float()

        # Update self.train_x and self.train_y
        self.set_train_data(inputs=train_X, targets=train_Y)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        mll = mll.to(train_X)

        fit_gpytorch_model(mll)
        # self.train()
        #
        # optimizer = torch.optim.Adam([{'params': self.parameters()}], lr=0.1)
        # for epoch in range(150):
        #     optimizer.zero_grad()
        #     output = self(train_X)
        #     loss = -mll(output, self.train_targets)
        #     print("loss: {}".format(loss))
        #     loss.backward()
        #     optimizer.step()

    def set_train_data(self, inputs=None, targets=None, strict=True):
        """
        ** Adapted from gpytorch.models.exactgp **
        Set training data (does not re-fit model hyper-parameters).

        :param torch.Tensor inputs: The new training inputs.
        :param torch.Tensor targets: The new training targets.
        :param bool strict: (default True) If `True`, the new inputs and
            targets must have the same dtype/device as the current inputs and
            targets. Otherwise, any dtype/device are allowed.
        """
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            inputs = tuple(input_.unsqueeze(-1) if input_.ndimension() == 1 else input_ for input_ in inputs)
            if strict:
                for input_, t_input in zip(inputs, self.train_inputs or (None,)):
                    for attr in {"dtype", "device"}:
                        expected_attr = getattr(t_input, attr, None)
                        found_attr = getattr(input_, attr, None)
                        if expected_attr != found_attr:
                            msg = "Cannot modify {attr} of inputs (expected {e_attr}, found {f_attr})."
                            msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                            raise RuntimeError(msg)
            self.train_inputs = inputs[0]
        if targets is not None:
            if strict:
                for attr in {"dtype", "device"}:
                    expected_attr = getattr(self.train_targets, attr, None)
                    found_attr = getattr(targets, attr, None)
                    if expected_attr != found_attr:
                        msg = "Cannot modify {attr} of targets (expected {e_attr}, found {f_attr})."
                        msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                        raise RuntimeError(msg)
            self.train_targets = targets
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
