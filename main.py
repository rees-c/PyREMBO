from utils import *

import numpy as np
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement


def main(extrinsic_D, intrinsic_d, parameter_dict, objective):
    """

    :param extrinsic_D:
    :param intrinsic_d:
    :param parameter_dict: dictionary of continuous, scalar parameters
    :param objective:
    :return:
    """



if __name__ == "__main__":
    extrinsic_D = 10
    intrinsic_d = 4
    main(extrinsic_D=extrinsic_D, intrinsic_d=intrinsic_d)
