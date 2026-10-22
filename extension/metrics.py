"""
Compute the following metrics:
Forward Transfer (FT) - Measures plasticity of the method, mainly the ability to perform a task right after learning it
Negative Backward Tranfer (NBT) - Measures stability, mainly the ability to perform old tasks after learning new ones
Area under the Curve (AUC) - Measures the overall performance on all tasks at all timesteps
"""

import torch
import numpy as np

def compute_ft(s_conf_mat: torch.Tensor):
    """
    Compute forward transfer metric
    Args:
        s_conf_mat: success rate matrix created by LIBERO

    Returns:
    The forward transfer metric
    """
    print(s_conf_mat.shape)
    print(s_conf_mat)
    return 0.

def compute_NBT(s_conf_mat: torch.Tensor):
    """
    Compute negative backward transfer metric
    Args:
        s_conf_mat: success rate matrix created by LIBERO

    Returns:
    The negative backward transfer metric
    """
    return 0.

def compute_AUC(s_conf_mat: torch.Tensor):
    """
    Compute area under the curve metric
    Args:
        s_conf_mat: success rate matrix created by LIBERO

    Returns:
    The area under the curve metric
    """
    return 0.