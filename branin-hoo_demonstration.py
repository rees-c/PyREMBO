import numpy as np

from rembo import REMBO

def main():
    n_dims = 20
    d_embedding = 2
    n_trials = 100
    ind = np.random.RandomState(seed=0).choice(n_dims, 2, replace=False)
    def f(X):  # target function
        """
        minimum value of 0
        """
        x1 = X[0][ind[0]]
        x2 = X[0][ind[1]]
        return x1**2 + x2**2

    original_boundaries = np.array([[-1, 1]] * n_dims)
    opt = REMBO(original_boundaries, d_embedding)

    # Perform optimization
    for i in range(20):
        X_query = opt.select_query_point()
        y_query = -f(X_query)
        opt.update(X_query, y_query)
        print("best value: {}".format(opt.best_value()))

    print("X_embedded: {}".format(opt.X_embedded))
    print("y: {}".format(opt.y))
    import torch
    xtry = torch.Tensor([-1.4142,  1.4142])
    # xtry = torch.Tensor([0,  0])
    opt.model.eval()
    fpreds = opt.model.forward(xtry)
    print(fpreds.mean)
    print(fpreds.covariance_matrix)


if __name__ == "__main__":
    main()