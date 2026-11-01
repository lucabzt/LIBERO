"""
Plots of an AnchorER selection, as the authors' SelectiveER logs: where in the trajectories
the overlapping / disagreeing new data and the selected anchors lie, which past tasks the
anchors come from, and image grids of the strongest disagreements and anchors.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_partition(out_dir, new_t, overlap_mask, od_idx):
    """Timesteps of the new data inside / outside the overlap cluster and of the disagreements."""
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, new_t.max() + 10, 10)
    ax.hist(new_t[~overlap_mask], bins=bins, alpha=0.6, label="no overlap")
    ax.hist(new_t[overlap_mask], bins=bins, alpha=0.6, label="overlap (step 1)")
    ax.hist(new_t[od_idx], bins=bins, alpha=0.8, label="disagreement (step 2)")
    ax.set_xlabel("timestep in demo")
    ax.set_ylabel("count")
    ax.set_title("New task data")
    ax.legend()
    _save(fig, os.path.join(out_dir, "new_data_partition.png"))


def plot_disagreement(out_dir, val_scores, overlap_scores, threshold):
    """Disagreement of the held-out old data (baseline) and of the overlapping new data."""
    fig, ax = plt.subplots(figsize=(8, 4))
    hi = max(np.percentile(np.concatenate([val_scores, overlap_scores]), 99.5), threshold)
    bins = np.linspace(0, hi * 1.05, 60)
    ax.hist(val_scores, bins=bins, alpha=0.6, density=True, label="held-out old data")
    ax.hist(overlap_scores, bins=bins, alpha=0.6, density=True, label="overlapping new data")
    ax.axvline(threshold, color="k", linestyle="--", label="threshold")
    ax.set_xlabel("action disagreement")
    ax.set_ylabel("density")
    ax.legend()
    _save(fig, os.path.join(out_dir, "disagreement.png"))


def plot_anchors(out_dir, anchor_task, anchor_t, task_names):
    """Number of anchors per past task and their timesteps."""
    tasks = sorted(set(anchor_task.tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = [np.sum(anchor_task == k) for k in range(len(task_names))]
    axes[0].barh(range(len(task_names)), counts)
    axes[0].set_yticks(range(len(task_names)))
    axes[0].set_yticklabels([f"{k + 1}: {n}" for k, n in enumerate(task_names)], fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("anchors")
    for k in tasks:
        axes[1].hist(anchor_t[anchor_task == k], bins=20, alpha=0.6, label=f"task {k + 1}")
    axes[1].set_xlabel("timestep in demo")
    axes[1].set_ylabel("anchors")
    axes[1].legend(fontsize=7)
    _save(fig, os.path.join(out_dir, "anchors.png"))


def plot_image_grid(path, images, labels, title):
    """Up to 100 images (H, W, 3) in [0, 1] in a 10 x 10 grid with a label each."""
    n = min(100, len(images))
    n_cols = 10
    n_rows = max(1, int(np.ceil(n / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.75 * n_rows))
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])
            ax.set_title(labels[i], fontsize=5)
    fig.suptitle(title)
    _save(fig, path)


def frame(sample, key):
    """First obs frame of a dataset sample as (H, W, 3); LIBERO renders upside down."""
    img = sample["obs"][key][0]
    img = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    return np.clip(np.transpose(img, (1, 2, 0))[::-1, ::-1], 0, 1)
