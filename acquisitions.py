import numpy as np
import math

# compute the cumulative distribution function of a standard Gaussian random variable
def gaussian_cdf(u):
    # u is a scalar
    return 0.5*(1.0 + math.erf(u/np.sqrt(2.0)))

# compute the probability mass function of a standard Gaussian random variable
def gaussian_pmf(u):
    # u is a scalar
    return np.exp(-u**2/2.0)/np.math.sqrt(2.0*np.pi)

# compute the probability of improvement (PI) acquisition function
#
# Ybest     points at which to compute the kernel (size: d x n)
# mean      mean of prediction
# stdev     standard deviation of prediction (the square root of the variance)
#
# returns   PI acquisition function
def pi_acquisition(Ybest, mean, stdev):
    # TODO students should implement this

    pi_acq_func = -gaussian_cdf((Ybest - mean)/stdev)
    return pi_acq_func


# compute the expected improvement (EI) acquisition function
#
# Ybest     points at which to compute the kernel (size: d x n)
# mean      mean of prediction
# stdev     standard deviation of prediction
#
# returns   EI acquisition function
def ei_acquisition(Ybest, mean, stdev):
    # mean is the mean of the query point vector x*
    # stdev is the stdev of the query point vector x*
    # TODO students should implement this

    u = (Ybest - mean) / stdev
    ei_acq_func = -stdev*(gaussian_pmf(u) + u*gaussian_cdf(u))
    return ei_acq_func

# return a function that computes the lower confidence bound (LCB) acquisition function
#
# kappa     parameter for LCB
#
# returns   function that computes the LCB acquisition function
def lcb_acquisition(kappa):
    def A_lcb(Ybest, mean, stdev):
        # TODO students should implement this

        lcb_acq_func = mean - kappa*stdev
        return lcb_acq_func

    return A_lcb
