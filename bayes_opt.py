import logging

import torch
import numpy as np
from ExactGaussianProcessV3 import ExactGaussianProcess
from sklearn.utils import check_random_state
from botorch.acquisition.monte_carlo import qExpectedImprovement, \
    qUpperConfidenceBound
from botorch import fit_gpytorch_model
from scipy.optimize import fmin_l_bfgs_b
from botorch.acquisition.objective import ConstrainedMCObjective

from utils import global_optimization, get_fitted_model

class bayes_opt():
    """
    Maximize a black-box objective function.
    """
    # TODO: implement functionality for discrete variables,
    #  implement acquisition functions

    def __init__(self, boundaries, initial_x=None,
                 initial_y=None,
                 acquisition_func=qExpectedImprovement,
                 maxf=1000, optimizer="random+GD",
                 initial_random_samples=5,
                 opt=None, fopt=None, random_embedding_seed=0,
                 types=None, do_scaling=True):
        """
        Vanilla Bayesian optimization.

        Parameters
        ----------
        original_boundaries ((D, 2) np.array): Boundaries of the original search
            space (of dimension D). The first column is the minimum value for
            the corresponding dimension/row, and the second column is the
            maximum value.
        initial_x: np.array
            Initial data points (in original data space) TODO: Implement
        initial_y: np.array
            Initial function evaluations TODO: Implement
        acquisition_func (str): Acquisition function to use. # TODO: Implement
        maxf (int): Maximum number of acquisition function evaluations that the
            optimizer can make.
        optimizer (str): Method name to use for optimizing the acquisition
            function.
        opt: (N, D) numpy array
            The global optima of the objective function (if known).
            Allows to compute and plot the distance of the incumbent
            to the global optimum.
        fopt: (N, 1) numpy array
            Function value of the N global optima (if known). Useful
            to compute the immediate or cumulative regret.
        """
        self.initial_random_samples = initial_random_samples
        self.acquisition_func = acquisition_func
        self.optimizer = optimizer
        self.maxf = maxf
        self.rng = check_random_state(random_embedding_seed)
        self.opt = opt  # optimal point
        self.fopt = fopt  # optimal function value
        self.boundaries = np.asarray(boundaries)

        # Dimensions of the original space
        self.d_orig = self.boundaries.shape[0]

        self.X = torch.Tensor()  # running list of data
        self.y = torch.Tensor()  # running list of function evaluations

        self.model = None
        self.boundaries_cache = {}

    def select_query_point(self, batch_size=1):
        """

        :param
            batch_size (int): number of query points to return
        :return:
            (batch_size x d_orig) numpy array
        """

        # TODO: Make the random initialization its own function so it can be done separately from the acquisition argmin
        # Initialize with random points
        if len(self.X) < self.initial_random_samples:

            # Select query point randomly from embedding_boundaries
            X_query = \
                self.rng.uniform(size=self.boundaries.shape[0]) \
                * (self.boundaries[:, 1] - self.boundaries[:, 0]) \
                + self.boundaries[:, 0]
            X_query = torch.from_numpy(X_query).unsqueeze(0)

        # Query by maximizing the acquisition function
        else:
            print("---------------------")
            print('querying')

            print("self.X.shape: {}".format(self.X.shape))
            print("self.y.shape: {}".format(self.y.shape))
            # Initialize model
            if len(self.X) == self.initial_random_samples:
                self.model = ExactGaussianProcess(
                    train_x=self.X.float(),
                    train_y=self.y.float(),
                )

            # Acquisition function
            qEI = qExpectedImprovement(
                model=self.model,
                best_f=torch.max(self.y).item(),
            )
            # qUCB = qUpperConfidenceBound(
            #     model=self.model,
            #     beta=2.0,
            # )

            print("batch_size: {}".format(batch_size))

            # Optimize for a (batch_size x d_embedding) tensor query point
            X_query = global_optimization(
                objective_function=qEI,
                boundaries=torch.from_numpy(self.boundaries).float(),
                batch_size=batch_size,  # number of query points to suggest
                )

            print("batched X_query: {}".format(X_query))
            print("batched X_query.shape: {}".format(X_query.shape))

        print("X concatenated: {}".format(self.X.shape))

        return X_query

    def update(self, X_query, y_query):
        """ Update internal model for observed (X, y) from true function.
        The function is meant to be used as follows.
            1. Call 'select_query_point' to update self.X_embedded with a new
                embedded query point, and to return a query point X_query in the
                original (unscaled) search space
            2. Evaluate X_query to get y_query
            3. Call this function ('update') to update the surrogate model (e.g.
                Gaussian Process)

        Args:
            X_query ((1,d_orig) np.array):
                Point in original input space to query
            y_query (float):
                Value of black-box function evaluated at X_query
            X_query_embedded ((1, d_embedding) np.array):
                Point in embedding space which maps 1:1 with X_query
        """
        print("X_query.shape: {}".format(X_query.shape))
        print("y_query.shape: {}".format(y_query.shape))

        # add new rows of data
        self.X = torch.cat([self.X.float(),
                            X_query.float()],
                           dim=0)
        self.y = torch.cat([self.y, torch.Tensor([[y_query]])], axis=0)

        print("self.X_embedded.shape: {}".format(self.X.shape))
        print("self.y.shape: {}".format(self.y.shape))
        self.model = get_fitted_model(self.X.float(),
                                      self.y.float())

    def best_params(self):
        """ Returns the best parameters found so far."""
        return self.X[np.argmax(self.y.numpy())]

    def best_value(self):
        """ Returns the optimal value found so far."""
        return np.max(self.y.numpy())

    def evaluate_f(self, x_query, black_box_function=None):
        """
        Evaluates input point in embedded space by first projecting back to
        original space and then scaling it to its original boundaries.

        Args:
        :return:
        """
        # BoTorch assumes a maximization problem
        if black_box_function is not None:
            return -black_box_function(x_query)
