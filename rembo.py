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

from utils import global_optimization, synth_obj_func, get_fitted_model

class REMBO():
    """
    Maximize a black-box objective function.
    """
    #TODO: implement functionality for discrete variables,
    # implement acquisition functions

    def __init__(self, original_boundaries, d_embedding, initial_x=None,
                 initial_y=None,
                 acquisition_func=qExpectedImprovement,
                 n_keep_dims=0, maxf=1000, optimizer="random+GD",
                 initial_random_samples=5,
                 opt=None, fopt=None, random_embedding_seed=0,
                 types=None, do_scaling=True):
        """
        Random EMbedding Bayesian Optimization [1] maps the original space
        in a lower dimensional subspace via a random embedding matrix and
        performs Bayesian Optimization only in this lower dimensional
        subspace.

        [1] Ziyu Wang and Masrour Zoghi and Frank Hutter and David Matheson
            and Nando de Freitas
            Bayesian Optimization in High Dimensions via Random Embeddings
            In: Proceedings of the 23rd international joint conference
            on Artificial Intelligence (IJCAI)

        Parameters
        ----------
        original_boundaries ((D, 2) np.array): Boundaries of the original search
            space (of dimension D). This is used for rescaling. The first column
            is the minimum value for the corresponding dimension/row, and the
            second column is the maximum value.
        n_keep_dims (int): Number of dimensions in the original space that are
            preserved in the embedding. This is used if certain dimensions are
            expected to be independently relevant. Assume that these dimensions
            are the first parameters of the input space.
        d_embedding: int
            Number of dimensions for the lower dimensional subspace
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
        do_scaling: boolean
            If set to true the input space is scaled to [-1, 1]. Useful
            to specify priors for the kernel lengthscale.
        """
        self.initial_random_samples = initial_random_samples
        self.acquisition_func = acquisition_func
        self.optimizer = optimizer
        self.maxf = maxf
        self.rng = check_random_state(random_embedding_seed)
        self.opt = opt  # optimal point
        self.fopt = fopt  # optimal function value
        self.n_keep_dims = n_keep_dims
        self.original_boundaries = np.asarray(original_boundaries)

        # Dimensions of the original space
        self.d_orig = self.original_boundaries.shape[0]

        # Dimension of the embedded space
        self.d_embedding = d_embedding

        # Draw random matrix from a standard normal distribution
        # (x = A @ x_embedding)
        self.A = self.rng.normal(loc=0.0,
                                 scale=1.0,
                                 size=(self.d_orig,
                                       self.d_embedding - self.n_keep_dims))

        self.X = []  # running list of data
        self.X_embedded = torch.Tensor()  # running list of embedded data
        self.y = torch.Tensor()  # running list of function evaluations

        self.model = None
        self.boundaries_cache = {}

    def select_query_point(self, batch_size=1):
        """

        :param batch_size: number of query points to return
        :return: (batch_size x d_orig) numpy array
        """

        # Compute boundaries on embedded space
        embedding_boundaries = self._compute_boundaries_embedding(
            self.original_boundaries)
        # embedding_boundaries = np.array(
        #     [[-np.sqrt(self.d_embedding),
        #       np.sqrt(self.d_embedding)]] * self.d_embedding).T
        print(embedding_boundaries)

        # TODO: Make the random initialization its own function so it can be done separately from the acquisition argmin
        # Initialize with random points
        if len(self.X) < self.initial_random_samples:
            # Select query point randomly from [-sqrt(d_embedding), sqrt(d_embedding)]^d
            # X_query_embedded = \
            #     self.rng.uniform(low=0.0, high=1.0, size=self.d_embedding) \
            #     * 2 * np.sqrt(self.d_embedding) \
            #     - np.sqrt(self.d_embedding)
            X_query_embedded = \
                self.rng.uniform(size=embedding_boundaries.shape[0]) \
                * (embedding_boundaries[:, 1] - embedding_boundaries[:, 0]) \
                + embedding_boundaries[:, 0]
            X_query_embedded = torch.from_numpy(X_query_embedded).unsqueeze(0)

            print("X_query_embedded.shape: {}".format(X_query_embedded.shape))

        # Query by maximizing the acquisition function
        else:
            print('querying')

            print("self.X_embedded.shape: {}".format(self.X_embedded.shape))
            print("self.y.shape: {}".format(self.y.shape))
            # Initialize model
            if len(self.X) == self.initial_random_samples:
                self.model = ExactGaussianProcess(
                    train_x=self.X_embedded.float(),
                    train_y=self.y.float(),
                )

            # Acquisition function
            qEI = qExpectedImprovement(
                model=self.model,
                best_f=torch.max(self.y).item(),
            )
            qUCB = qUpperConfidenceBound(
                model=self.model,
                beta=2.0,
            )

            print("batch_size: {}".format(batch_size))
            # Optimize for a (batch_size x d_embedding) tensor query point
            X_query_embedded = global_optimization(
                objective_function=qEI,
                boundaries=torch.from_numpy(embedding_boundaries).float(),
                batch_size=batch_size,  # number of query points to suggest
                )

        # Append to (n x d_embedding) Tensor
        self.X_embedded = torch.cat([self.X_embedded.float(),
                                     X_query_embedded.float()], dim=0)
        # self.X_embedded.append(X_query_embedded)

        # Map to higher dimensional space and clip to hard boundaries [-1, 1]
        X_query = np.clip(a=self._manifold_to_dataspace(X_query_embedded.numpy()),
                          a_min=self.original_boundaries[:, 0],
                          a_max=self.original_boundaries[:, 1])

        return X_query

    def _manifold_to_dataspace(self, X_embedded):
        """
        Map data from manifold to original data space.

        :param X_embedded: (1 x d_embedding) numpy.array
        :return: (1 x d_orig) numpy.array
        """
        X_query_kd = self.A @ X_embedded[self.n_keep_dims:].squeeze()
        X_query_kd = X_query_kd.T

        # concatenate column-wise
        if self.n_keep_dims > 0:
            X_query = np.hstack((X_embedded[:self.n_keep_dims], X_query_kd))
        else:
            X_query = X_query_kd

        # print("X_query ~[-1, 1]: {}".format(X_query))
        # scale 'X_query' to the original dataspace (from [-1, 1]^D)
        X_query = self._unscale(X_query)
        # print("X_query ~[-5, 15]: {}".format(X_query))

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

        :param X_query: np.array (q x d)
        :param y_query: float
        """

        # add new rows of data
        self.X.append(X_query)
        self.y = torch.cat([self.y, torch.Tensor([[y_query]])], axis=0)

        self.model = get_fitted_model(self.X_embedded.float(),
                                      self.y.float())


        # if len(self.X) > self.initial_random_samples:
        #     self.model.fit(self.X_embedded.float(), self.y.float())

        # # # BoTorch:
        # # Initialize model
        # mll_ei, model_ei = initialize_model(
        #     train_x_ei,
        #     train_obj_ei,
        #     train_con_ei,
        #     model_ei.state_dict(),
        # )
        # # Fit model
        # fit_gpytorch_model(mll_ei)

    def best_params(self):
        """ Returns the best parameters found so far."""
        return self.X[np.argmax(self.y.numpy())]

    def best_value(self):
        """ Returns the optimal value found so far."""
        return np.max(self.y.numpy())


########################
    def _rescale(self, x,
                 scaled_lower_bound=-1, scaled_upper_bound=1):
        """
        TODO: This function is unnecessary
        Transforms original input space to the space
            [new_lower_bound, new_upper_bound].

        Args:
            x (np.array (N, D)): Input points in original space
            scaled_lower_bound (int): New lower bound to linearly map x to.
            scaled_upper_bound (int): New upper bound to linearly map x to.
        Returns:
            np.array (N, D): Input points in [new_lower_bound, new_upper_bound]
                space
        """
        x_scaled = np.empty(x.shape)
        orig_ranges = self.original_boundaries[:][1] \
                      - self.original_boundaries[:][0]  # (D x 1) max-min
        new_range = scaled_upper_bound - scaled_lower_bound

        for dim in range(len(x)):
            x_scaled[:][dim] = (x[:][dim] - self.original_boundaries[dim][0]) \
                               * (new_range / orig_ranges[dim]) \
                               + scaled_lower_bound
        return x_scaled

    def _unscale(self, x_scaled,
                 scaled_lower_bound=-1, scaled_upper_bound=1):
        """
        Use this function to scale 'X_query' from 'select_query_point' to the
        original input space boundaries so that it can be evaluated by the
        function, 'evaluate_f'.

        :param x_scaled: (1, D) numpy.array
        :param scaled_lower_bound: int
        :param scaled_upper_bound: int
        :return:
        """
        x_scaled_ = np.copy(x_scaled)
        if not x_scaled_.ndim == 2:
            # Add a dimension if x_scaled is 1-D
            x_scaled_ = np.expand_dims(x_scaled_, axis=0)

        x_unscaled = np.empty(x_scaled_.shape)
        unscaled_ranges = self.original_boundaries[:, 1] \
            - self.original_boundaries[:, 0]  # (D,) max-min
        scaled_range = scaled_upper_bound - scaled_lower_bound

        # print("original_boundaries: {}".format(self.original_boundaries))
        # print("unscaled_ranges.shape: {}".format(unscaled_ranges.shape))
        # print("unscaled_ranges: {}".format(unscaled_ranges))
        for dim in range(len(x_scaled_)):
            x_unscaled[:][dim] = (x_scaled_[:][dim] - scaled_lower_bound) \
                               * (unscaled_ranges[dim] / scaled_range) \
                               + self.original_boundaries[dim][0]
        return x_unscaled

    def evaluate_f(self, x_query, black_box_function=synth_obj_func):
        """
        Evaluates input point in embedded space by first projecting back to
        original space and then scaling it to its original boundaries.

        Args:
        :return:
        """
        # BoTorch assumes a maximization problem
        return -black_box_function(x_query)

    def _compute_boundaries_embedding(self, boundaries):
        """ Approximate box constraint boundaries on low-dimensional manifold

            Args:
                boundaries ((d_orig, 2) np.array):
            Returns:
                boundaries_embedded ((d_embedding, 2) np.narray):

        """
        # Check if boundaries have been determined before
        boundaries_hash = hash(boundaries[self.n_keep_dims:].tostring())
        if boundaries_hash in self.boundaries_cache:
            boundaries_embedded = \
                np.array(self.boundaries_cache[boundaries_hash])
            boundaries_embedded[:self.n_keep_dims] = \
                boundaries[:self.n_keep_dims]  # Overwrite keep-dim's boundaries
            return boundaries_embedded

        # Determine boundaries on embedded space
        boundaries_embedded = \
            np.empty((self.n_keep_dims + self.d_embedding, 2))
        boundaries_embedded[:self.n_keep_dims] = boundaries[:self.n_keep_dims]
        for dim in range(self.n_keep_dims,
                         self.n_keep_dims + self.d_embedding):
            x_embedded = np.zeros(self.n_keep_dims + self.d_embedding)
            while True:
                x = self._manifold_to_dataspace(x_embedded)
                if np.sum(np.logical_or(
                        x[self.n_keep_dims:] < boundaries[self.n_keep_dims:, 0],
                        x[self.n_keep_dims:] > boundaries[self.n_keep_dims:,
                                               1])) \
                        > (self.d_orig - self.n_keep_dims) / 2:
                    break
                x_embedded[dim] -= 0.01
            boundaries_embedded[dim, 0] = x_embedded[dim]

            x_embedded = np.zeros(self.n_keep_dims + self.d_embedding)
            while True:
                x = self._manifold_to_dataspace(x_embedded)
                if np.sum(np.logical_or(
                        x[self.n_keep_dims:] < boundaries[self.n_keep_dims:, 0],
                        x[self.n_keep_dims:] > boundaries[self.n_keep_dims:, 1])) \
                        > (self.d_orig - self.n_keep_dims) / 2:
                    break
                x_embedded[dim] += 0.01
            boundaries_embedded[dim, 1] = x_embedded[dim]

        self.boundaries_cache[boundaries_hash] = boundaries_embedded

        return boundaries_embedded

