"""
Compute the following metrics:
Forward Transfer (FT) - Measures plasticity of the method, mainly the ability to perform a task right after learning it
Negative Backward Tranfer (NBT) - Measures stability, mainly the ability to perform old tasks after learning new ones
Area under the Curve (AUC) - Measures the overall performance on all tasks at all timesteps
"""

import torch

def compute_ft(s_fwd: torch.Tensor):
    """
    Compute forward transfer metric
    Args:
        s_fwd: forward transfer matrix for each task, already contained in results.pt. Simply the success rate of task k, averaged over all eval epochs

    Returns:
    The forward transfer metric
    """
    return torch.as_tensor(s_fwd).mean()


def nbt_per_task(s_conf_mat: torch.Tensor):
    """
    Compute negative backward transfer for every task except the last one
    Args:
        s_conf_mat: success rate matrix created by LIBERO

    Returns:
    Tensor of shape (n_tasks - 1,) with NBT_k for each task k
    """
    s_conf_mat = torch.as_tensor(s_conf_mat)
    n = s_conf_mat.shape[0]
    # entry [t, k] = c_kk - c_tk, keep only t > k
    diff = torch.tril(s_conf_mat.diagonal()[None, :] - s_conf_mat, diagonal=-1)
    # column k has n-1-k terms; the last column has none, so drop it
    counts = torch.arange(n - 1, 0, -1, dtype=diff.dtype, device=diff.device)
    return diff.sum(dim=0)[:-1] / counts


def compute_NBT(s_conf_mat: torch.Tensor):
    """
    Compute negative backward transfer metric
    Args:
        s_conf_mat: success rate matrix created by LIBERO

    Returns:
    The negative backward transfer metric
    """
    return nbt_per_task(s_conf_mat).mean()


def auc_per_task(s_conf_mat: torch.Tensor, s_fwd: torch.Tensor):
    """
    Compute area under the curve for every task
    Args:
        s_conf_mat: success rate matrix created by LIBERO
        s_fwd: forward transfer matrix for each task, already contained in results.pt. Simply the success rate of task k, averaged over all eval epochs

    Returns:
    Tensor of shape (n_tasks,) with AUC_k for each task k
    """
    s_conf_mat = torch.as_tensor(s_conf_mat)
    s_fwd = torch.as_tensor(s_fwd, dtype=s_conf_mat.dtype, device=s_conf_mat.device)
    n = s_conf_mat.shape[0]
    # column k: sum of c_tk for all t > k
    future = torch.tril(s_conf_mat, diagonal=-1).sum(dim=0)
    # column k has n-k terms: FWT_k plus n-1-k future entries
    counts = torch.arange(n, 0, -1, dtype=s_conf_mat.dtype, device=s_conf_mat.device)
    return (s_fwd + future) / counts


def compute_AUC(s_conf_mat: torch.Tensor, s_fwd: torch.Tensor):
    """
    Compute area under the curve metric
    Args:
        s_conf_mat: success rate matrix created by LIBERO
        s_fwd: forward transfer matrix for each task, already contained in results.pt. Simply the success rate of task k, averaged over all eval epochs

    Returns:
    The area under the curve metric
    """
    return auc_per_task(s_conf_mat, s_fwd).mean()
