"""
This takes a results.pt file from LIBERO and converts it into a human readable format by
a) Reporting common metrics like Forward Transfer, Negative Backward Tranfer and Area under the Curve (AUC)
b) Creating plots to track task success over time
"""

import argparse
import os

import matplotlib

from metrics import compute_ft

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_results(results_file: str) -> dict:
    """
    Load a result.pt file written by libero/lifelong/main.py
    Args:
        results_file: path to the result.pt file

    Returns:
    Dict with keys L_conf_mat, S_conf_mat (n_tasks x n_tasks, row = task trained up to,
    column = evaluated task), L_fwd and S_fwd (n_tasks,)
    """
    return torch.load(results_file)


def plot_success_matrix(
    results: dict, output_dir: str, n_eval: int = 20, title: str = None
) -> str:
    """
    Plot success rate of every seen task after training on each task (lower triangle only),
    with the evaluated task on the y-axis and the task trained up to on the x-axis.
    Args:
        results: dict loaded by load_results
        output_dir: directory the figure is saved into
        n_eval: number of evaluation episodes per task, used for the "k/n" cell labels
        title: optional figure title

    Returns:
    Path of the saved figure
    """
    # S_conf_mat[i, j] = success on task j after training on task i -> transpose so that
    # columns are training stages and rows are evaluated tasks
    success = np.asarray(results["S_conf_mat"]).T
    n_tasks = success.shape[0]
    # only tasks that have been seen are evaluated (j <= i)
    seen = np.tril(np.ones((n_tasks, n_tasks), dtype=bool)).T
    masked = np.ma.masked_where(~seen, success)

    fig, ax = plt.subplots(figsize=(0.8 * n_tasks + 2, 0.7 * n_tasks + 1.5))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("white")
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, origin="lower")

    for row in range(n_tasks):
        for col in range(n_tasks):
            if not seen[row, col]:
                continue
            value = success[row, col]
            ax.text(
                col,
                row,
                f"{int(round(value * n_eval))}/{n_eval}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white" if value > 0.5 else "#333333",
            )

    ticks = np.arange(n_tasks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks + 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks + 1)
    ax.set_xlabel("After Training on Task")
    ax.set_ylabel("Evaluated Task")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Success rate", fraction=0.046, pad=0.04)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "success_matrix.png")
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_file", required=True, help="path to a LIBERO result.pt")
    parser.add_argument("--output_dir", required=True, help="directory to save plots into")
    parser.add_argument("--n_eval", type=int, default=20, help="eval episodes per task (cfg.eval.n_eval)")
    parser.add_argument("--title", default=None, help="title of the plot")
    args = parser.parse_args()

    results = load_results(args.results_file)
    fwd_transfer = compute_ft(results["S_conf_mat"])
    output_file = plot_success_matrix(results, args.output_dir, args.n_eval, args.title)
    print(f"[info] saved success matrix to {output_file}")


if __name__ == "__main__":
    main()
