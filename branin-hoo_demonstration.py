import numpy as np

from rembo import REMBO

def main():
    n_dims = 20
    d_embedding = 3
    n_trials = 30
    ind = np.random.RandomState(seed=0).choice(n_dims, 2, replace=False)
    def f(X):  # target function
        """
        minimum value of 0
        """
        x1 = X[0][ind[0]]
        x2 = X[0][ind[1]]
        return x1**2 + x2**2

    original_boundaries = np.array([[-1, 1]] * n_dims)
    print("original_boundaries.shape: {}".format(original_boundaries.shape))
    opt = REMBO(original_boundaries, d_embedding)

    # Perform optimization
    for i in range(n_trials):
        X_query = opt.select_query_point()
        y_query = -f(X_query)
        opt.update(X_query, y_query)
        print("best y value: {}".format(opt.best_value()))
        print("best actual x values: {}".format(opt.best_params()[0][ind[:2]]))
        print("best actual x values distance from 0: {}".format(
            np.linalg.norm(opt.best_params()[0][ind[:2]])))
        print("---------------------")

    # print("X_embedded: {}".format(opt.X_embedded))
    # print("y: {}".format(opt.y))
    import torch
    # xtry = torch.Tensor([-1.4142,  1.4142])
    # xtry = torch.Tensor([0,  0])
    # opt.model.eval()
    # fpreds = opt.model.forward(xtry)
    # print(fpreds.mean)
    # print(fpreds.covariance_matrix)


if __name__ == "__main__":
    main()
