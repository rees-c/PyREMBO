import numpy as np

from rembo import REMBO


def main():
    n_dims = 20
    d_embedding = 4
    n_trials = 30

    ind = np.random.RandomState(seed=0).choice(n_dims, 2, replace=False)
    def f(X):  # black-box objective function to minimize
        """
        minimum value of 0
        """
        x1 = X[0][ind[0]]
        x2 = X[0][ind[1]]
        return x1**2 + x2**2

    def ensure_not_1D(x):
        """
        Ensure x is not 1D (i.e. size (D,))
        :param x: torch.Tensor
        :return:
        """
        import torch

        if x.ndim == 1:
            if isinstance(x, np.ndarray):
                x = np.expand_dims(x, axis=0)
            elif isinstance(x, torch.Tensor):
                x = x.unsqueeze(0)
        return x

    original_boundaries = np.array([[-1, 1]] * n_dims)
    print("original_boundaries.shape: {}".format(original_boundaries.shape))
    opt = REMBO(original_boundaries, d_embedding)

    # Perform optimization
    for i in range(n_trials):
        X_queries, X_queries_embedded = opt.select_query_point(batch_size=3)

        # Ensure not 1D (i.e. size (D,))
        X_queries = ensure_not_1D(X_queries)

        # Evaluate the batch of query points 1-by-1
        for row_idx in range(len(X_queries)):
            X_query = X_queries[row_idx]
            X_query_embedded = X_queries_embedded[row_idx]

            # Ensure no 1D tensors (i.e. expand tensors of size (D,))
            X_query = ensure_not_1D(X_query)
            X_query_embedded = ensure_not_1D(X_query_embedded)

            y_query = -f(X_query)
            opt.update(X_query, y_query, X_query_embedded)

        print("best y value: {}".format(opt.best_value()))
        print("best actual x: {}".format(opt.best_params()[0][ind[:2]]))
        print("best actual x values distance from 0: {}".format(
            np.linalg.norm(opt.best_params()[0][ind[:2]])))
        print("---------------------")


if __name__ == "__main__":
    main()
