import numpy as np

# compute the Gaussian RBF kernel matrix for a vector of data points (in TensorFlow)
#
#   ISSUE WITH RBF: Hyperparameter scaling is bad (i.e. if parameters we are
#       trying to optimize are on different scales, then RBF metric of distance
#       will emphasize parameters with larger scales)
#
# Xs        points at which to compute the kernel (size: d x m)
# Zs        other points at which to compute the kernel (size: d x n)
# gamma     gamma parameter for the RBF kernel
#
# returns   an (m x n) matrix Sigma where Sigma[i,j] = K(Xs[:,i], Zs[:,j])
def rbf_kernel_matrix(Xs, Zs, gamma):
    # TODO students should implement this

    # NUMPY VERSION--------------------------------------------------------------------
    # credit to: https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/Broadcasting.html
    xsq_summed = np.sum(Xs ** 2, axis=0)  # each column is squared and summed (M,)
    zsq_summed = np.sum(Zs ** 2, axis=0)  # each column is squared and summed (N,)

    # Insert a size-1 dimension in xsq (M,) so that we can add
    # all pairs of numbers between the resulting (M,1) and (N,) arrays
    sq_sum = xsq_summed[:, np.newaxis] + zsq_summed

    # Compute (M x N) cross term
    xz_prod = -2 * (Xs.T @ Zs)

    # Compute l2 dist squared
    l2_sqdist = sq_sum + xz_prod

    #Compute kernel
    kern = np.exp(-gamma * l2_sqdist)

    # # TENSORFLOW VERSION---------------------------------------------------------------
    # #tf.multiply(X,Y) does elementwise multiplication
    # #tf.matmul(X,Y) does matrix multiplication
    # #tf.expand_dims(X,i) adds a 1D dimension to tensor X's shape at index i
    #
    # xsq_summed = tf.reduce_sum(tf.math.square(Xs), axis=0)  # each column is squared and summed (M,)
    # zsq_summed = tf.reduce_sum(tf.math.square(Zs), axis=0)  # each column is squared and summed (N,)
    #
    # # print("passed sq Xs tf: " + str(xsq_summed))
    # # print("passed sq summed Xs tf: " + str(xsq_summed))
    # # print("passed sq summed test tf shape: " + str(xsq_summed.get_shape()))
    #
    # # Insert a size-1 dimension in xsq (M,) so that we can add
    # # all pairs of numbers between the resulting (M,1) and (N,) arrays
    #
    # #print("test shape tf: " + str(xsq_summed.get_shape())) #Tensor("Shape:0", shape=(0,), dtype=int32)
    # sq_sum = tf.expand_dims(xsq_summed, 1) + zsq_summed
    #
    # # Compute (M x N) cross term
    # xz_prod = -2.0 * tf.matmul(tf.transpose(Xs), Zs)
    #
    # # Compute l2 dist squared
    # l2_sqdist = sq_sum + xz_prod
    #
    # # Compute kernel
    # kern = tf.math.exp(-gamma * l2_sqdist)

    return kern

def low_d_rbf_kernel(y1, y2, sigma2=0.1):
    return np.exp(-(y1 - y2)**2 / 2*sigma2)

