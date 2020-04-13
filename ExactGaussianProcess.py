from abc import ABC

import torch
import gpytorch

from botorch.models import FixedNoiseGP
from botorch import fit_gpytorch_model

# # use a GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype = torch.float

#  GPyTorch
class ExactGaussianProcess(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y,
                 likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                 mean_module=gpytorch.means.ConstantMean(),
                 covar_module=gpytorch.kernels.ScaleKernel(
                     gpytorch.kernels.MaternKernel(nu=2.5))
                 ):
        """

        :param train_x:
        :param train_y:
        :param likelihood:
        :param mean_module:
        :param covar_module: Default assumes that all dimensions of x are of the
            same scale.
        """

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    @property
    def train_x(self):
        return self.train_inputs

    @property
    def train_y(self):
        return self.train_targets

    def forward(self, x):
        """

        :param x: (n x d)
        :return fpreds (object): fpreds has the following attributes.
            f_mean = f_preds.mean (prediction)
            f_var = f_preds.variance (prediction uncertainty)
            f_covar = f_preds.covariance_matrix
            f_samples = f_preds.sample(sample_shape=torch.Size(1000,))
        """
        # gets called through subsubsubclass' __call__ method

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        fpreds = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return fpreds

    def fit(self, train_x_, train_y_):
        if not isinstance(train_x_, torch.Tensor):


        # Update self.train_x and self.train_y
        self.set_train_data(inputs=train_x_, targets=train_y_)

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

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).

        Returns:
            A `GPyTorchPosterior` object, representing `batch_shape` joint
            distributions over `q` points and the outputs selected by
            `output_indices` each. Includes observation noise if specified.
        """
        self.eval()  # make sure model is in eval mode
        with gpt_posterior_settings():
            # insert a dimension for the output dimension
            if self._num_outputs > 1:
                X, output_dim_idx = add_output_dim(
                    X=X, original_batch_shape=self._input_batch_shape
                )
            mvn = self(X)
            if observation_noise is not False:
                if torch.is_tensor(observation_noise):
                    # TODO: Validate noise shape
                    # make observation_noise `batch_shape x q x n`
                    obs_noise = observation_noise.transpose(-1, -2)
                    mvn = self.likelihood(mvn, X, noise=obs_noise)
                elif isinstance(self.likelihood, FixedNoiseGaussianLikelihood):
                    # Use the mean of the previous noise values (TODO: be smarter here).
                    noise = self.likelihood.noise.mean().expand(X.shape[:-1])
                    mvn = self.likelihood(mvn, X, noise=noise)
                else:
                    mvn = self.likelihood(mvn, X)
            if self._num_outputs > 1:
                mean_x = mvn.mean
                covar_x = mvn.covariance_matrix
                output_indices = output_indices or range(self._num_outputs)
                mvns = [
                    MultivariateNormal(
                        mean_x.select(dim=output_dim_idx, index=t),
                        lazify(covar_x.select(dim=output_dim_idx, index=t)),
                    )
                    for t in output_indices
                ]
                mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)

        posterior = GPyTorchPosterior(mvn=mvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        return posterior