"""
This takes a result.pt file from LIBERO and converts it into a human readable format by
a) Reporting common metrics like Forward Transfer, Negative Backward Tranfer and Area under the Curve (AUC)
b) Reporting per-task metrics like forgetting, success right after learning and final success
c) Creating plots to track task success over time
d) Writing everything into a single markdown report (report.md) inside output_dir

If config.json and task{k}_auc.log files lie next to result.pt, they are used to add run
information, task names and per-epoch learning curves.
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from metrics import auc_per_task, compute_AUC, compute_ft, compute_NBT, nbt_per_task

# categorical slots, assigned in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT = "#333333"
MUTED = "#8a8985"

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e5e4e0",
        "grid.linewidth": 0.8,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "legend.frameon": False,
        "font.size": 10,
    }
)


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


def load_config(run_dir: str) -> dict:
    """
    Load the config.json that LIBERO saves next to result.pt
    Args:
        run_dir: directory containing result.pt

    Returns:
    The config as dict, or None if there is no config.json
    """
    config_file = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_file):
        return None
    with open(config_file) as f:
        return json.load(f)


def load_auc_logs(run_dir: str, n_tasks: int) -> list:
    """
    Load the per-epoch success and loss curves (task{k}_auc.log) written by learn_one_task
    Args:
        run_dir: directory containing result.pt
        n_tasks: number of tasks

    Returns:
    List with one dict {"success": array, "loss": array} per task, or None if any log is missing
    """
    logs = []
    for k in range(n_tasks):
        log_file = os.path.join(run_dir, f"task{k}_auc.log")
        if not os.path.exists(log_file):
            return None
        log = torch.load(log_file)
        logs.append({key: np.asarray(value, dtype=float) for key, value in log.items()})
    return logs


def load_task_names(config: dict) -> list:
    """
    Look up the task names of the benchmark in the order they were trained on
    Args:
        config: config loaded by load_config

    Returns:
    List of task names, or None if the benchmark cannot be loaded
    """
    if config is None:
        return None
    try:
        from libero.libero import benchmark

        bench = benchmark.get_benchmark(config["benchmark_name"])(
            config["data"]["task_order_index"]
        )
        return bench.get_task_names()
    except Exception as e:
        print(f"[warning] could not load task names: {e}")
        return None


def compute_metrics(results: dict, n_eval: int, auc_logs: list = None) -> dict:
    """
    Compute summary and per-task metrics from the result dict
    Args:
        results: dict loaded by load_results
        n_eval: number of evaluation episodes per task, used for confidence intervals
        auc_logs: per-epoch curves loaded by load_auc_logs (optional)

    Returns:
    Dict with a "summary" dict of scalars and a "per_task" dict of arrays (n_tasks,)
    """
    success = np.asarray(results["S_conf_mat"], dtype=float)
    s_fwd = np.asarray(results["S_fwd"], dtype=float)
    n_tasks = success.shape[0]

    after_learning = success.diagonal().copy()
    final = success[-1].copy()
    # NBT and forgetting are undefined for the last task
    nbt = np.append(nbt_per_task(success).numpy(), np.nan)
    auc = auc_per_task(success, s_fwd).numpy()
    # forgetting: best success on task k at any stage before the last one minus final success
    forgetting = np.full(n_tasks, np.nan)
    for k in range(n_tasks - 1):
        forgetting[k] = success[k:-1, k].max() - final[k]
    # average success over all tasks seen so far, after each training stage
    avg_seen = np.array([success[i, : i + 1].mean() for i in range(n_tasks)])

    per_task = {
        "fwt": s_fwd,
        "success_after_learning": after_learning,
        "success_after_learning_ci": confidence_interval(after_learning, n_eval),
        "final_success": final,
        "final_success_ci": confidence_interval(final, n_eval),
        "forgetting": forgetting,
        "nbt": nbt,
        "auc": auc,
        "fwd_loss": np.asarray(results["L_fwd"], dtype=float),
        "avg_seen_success": avg_seen,
    }
    if auc_logs is not None:
        per_task["zero_shot_success"] = np.array([log["success"][0] for log in auc_logs])
        per_task["best_train_success"] = np.array([log["success"].max() for log in auc_logs])
        per_task["best_eval_idx"] = np.array([log["success"].argmax() for log in auc_logs])

    summary = {
        "fwt": float(compute_ft(s_fwd)),
        "nbt": float(compute_NBT(success)),
        "auc": float(compute_AUC(success, s_fwd)),
        "fwt_std": float(s_fwd.std()),
        "nbt_std": float(np.nanstd(nbt)),
        "auc_std": float(auc.std()),
        "mean_success_after_learning": float(after_learning.mean()),
        "final_avg_success": float(final.mean()),
        "avg_forgetting": float(np.nanmean(forgetting)),
        "max_forgetting": float(np.nanmax(forgetting)),
        "fwd_loss": float(per_task["fwd_loss"].mean()),
        "n_tasks": n_tasks,
    }
    if auc_logs is not None:
        summary["zero_shot_success"] = float(per_task["zero_shot_success"].mean())
        summary["best_train_success"] = float(per_task["best_train_success"].mean())
    return {"summary": summary, "per_task": per_task}


def confidence_interval(p: np.ndarray, n_eval: int) -> np.ndarray:
    """
    95% normal approximation confidence interval of a success rate, as in libero/lifelong/utils.py
    Args:
        p: success rates
        n_eval: number of evaluation episodes the success rates are based on

    Returns:
    Half width of the confidence interval for each success rate
    """
    return 1.96 * np.sqrt(p * (1 - p) / n_eval)


def save_figure(fig, output_dir: str, name: str) -> str:
    """
    Save a figure into output_dir and close it
    Args:
        fig: matplotlib figure
        output_dir: directory the figure is saved into
        name: file name of the figure

    Returns:
    Path of the saved figure
    """
    output_file = os.path.join(output_dir, name)
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_file


def plot_matrix(
    matrix: np.ndarray,
    output_dir: str,
    name: str,
    cmap: str,
    colorbar_label: str,
    cell_label,
    vmin: float = None,
    vmax: float = None,
    title: str = None,
) -> str:
    """
    Plot a LIBERO confusion matrix (lower triangle only), with the evaluated task on the
    y-axis and the task trained up to on the x-axis.
    Args:
        matrix: S_conf_mat or L_conf_mat, [i, j] = metric on task j after training on task i
        output_dir: directory the figure is saved into
        name: file name of the figure
        cmap: name of a sequential matplotlib colormap
        colorbar_label: label of the colorbar
        cell_label: function mapping a cell value to its text label
        vmin, vmax: range of the colormap
        title: optional figure title

    Returns:
    Path of the saved figure
    """
    # transpose so that columns are training stages and rows are evaluated tasks
    values = np.asarray(matrix, dtype=float).T
    n_tasks = values.shape[0]
    # only tasks that have been seen are evaluated (j <= i)
    seen = np.tril(np.ones((n_tasks, n_tasks), dtype=bool)).T
    masked = np.ma.masked_where(~seen, values)
    vmin = masked.min() if vmin is None else vmin
    vmax = masked.max() if vmax is None else vmax

    fig, ax = plt.subplots(figsize=(0.8 * n_tasks + 2, 0.7 * n_tasks + 1.5))
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("white")
    im = ax.imshow(masked, cmap=color_map, vmin=vmin, vmax=vmax, origin="lower")
    ax.grid(False)

    for row in range(n_tasks):
        for col in range(n_tasks):
            if not seen[row, col]:
                continue
            value = values[row, col]
            ax.text(
                col,
                row,
                cell_label(value),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white" if (value - vmin) / (vmax - vmin + 1e-12) > 0.5 else TEXT,
            )

    ticks = np.arange(n_tasks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks + 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks + 1)
    ax.set_xlabel("After training on task")
    ax.set_ylabel("Evaluated task")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)
    return save_figure(fig, output_dir, name)


def plot_per_task_success(per_task: dict, output_dir: str, title: str = None) -> str:
    """
    Grouped bars per task: FWT, success right after learning and final success
    Args:
        per_task: per-task metrics from compute_metrics
        output_dir: directory the figure is saved into
        title: optional figure title

    Returns:
    Path of the saved figure
    """
    n_tasks = len(per_task["fwt"])
    x = np.arange(n_tasks)
    width = 0.27
    bars = [
        ("FWT (avg. over training)", per_task["fwt"], None),
        ("After learning task", per_task["success_after_learning"], per_task["success_after_learning_ci"]),
        ("After learning all tasks", per_task["final_success"], per_task["final_success_ci"]),
    ]

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * n_tasks + 2), 4))
    for i, (label, values, ci) in enumerate(bars):
        ax.bar(
            x + (i - 1) * width,
            values,
            width * 0.9,
            color=SERIES[i],
            label=label,
            yerr=ci,
            error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(x + 1)
    ax.set_xlabel("Task")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    if title:
        ax.set_title(title, fontweight="bold")
    return save_figure(fig, output_dir, "per_task_success.png")


def plot_per_task_transfer(per_task: dict, output_dir: str, title: str = None) -> str:
    """
    Two panels with per-task NBT and per-task AUC
    Args:
        per_task: per-task metrics from compute_metrics
        output_dir: directory the figure is saved into
        title: optional figure title

    Returns:
    Path of the saved figure
    """
    n_tasks = len(per_task["fwt"])
    x = np.arange(n_tasks)
    panels = [
        ("NBT$_k$ (lower is better)", per_task["nbt"]),
        ("AUC$_k$ (higher is better)", per_task["auc"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.1 * n_tasks + 3), 3.5))
    for ax, (label, values) in zip(axes, panels):
        valid = ~np.isnan(values)
        ax.bar(x[valid], values[valid], 0.6, color=SERIES[0])
        ax.axhline(np.nanmean(values), color=MUTED, linestyle="--", linewidth=1.5)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(x + 1)
        ax.set_xlabel("Task")
        ax.set_title(f"{label}, mean {np.nanmean(values):.2f}")
        ax.grid(axis="x", visible=False)
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_dir, "per_task_transfer.png")


def plot_success_over_stages(per_task: dict, output_dir: str, title: str = None) -> str:
    """
    Average success on all seen tasks and success on the newest task after each training stage
    Args:
        per_task: per-task metrics from compute_metrics
        output_dir: directory the figure is saved into
        title: optional figure title

    Returns:
    Path of the saved figure
    """
    n_tasks = len(per_task["fwt"])
    stages = np.arange(1, n_tasks + 1)
    lines = [
        ("Newest task", per_task["success_after_learning"]),
        ("Avg. over all seen tasks", per_task["avg_seen_success"]),
    ]

    fig, ax = plt.subplots(figsize=(max(7, 0.7 * n_tasks + 2), 4))
    for i, (label, values) in enumerate(lines):
        ax.plot(stages, values, color=SERIES[i], linewidth=2, marker="o", markersize=5, label=label)
        ax.annotate(
            f"{values[-1]:.2f}",
            (stages[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=TEXT,
            fontsize=9,
        )
    ax.set_xticks(stages)
    ax.set_xlabel("After training on task")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    if title:
        ax.set_title(title, fontweight="bold")
    return save_figure(fig, output_dir, "success_over_stages.png")


def plot_task_trajectories(results: dict, output_dir: str, title: str = None) -> str:
    """
    Small multiples with the success rate of each task over all training stages after it was learned
    Args:
        results: dict loaded by load_results
        output_dir: directory the figure is saved into
        title: optional figure title

    Returns:
    Path of the saved figure
    """
    success = np.asarray(results["S_conf_mat"], dtype=float)
    n_tasks = success.shape[0]
    n_cols = min(5, n_tasks)
    n_rows = int(np.ceil(n_tasks / n_cols))
    stages = np.arange(1, n_tasks + 1)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 2.2 * n_rows), sharex=True, sharey=True, squeeze=False
    )
    for k, ax in enumerate(axes.flat):
        if k >= n_tasks:
            ax.set_visible(False)
            continue
        ax.plot(stages[k:], success[k:, k], color=SERIES[0], linewidth=2, marker="o", markersize=4)
        ax.axvline(k + 1, color=MUTED, linestyle=":", linewidth=1)
        ax.set_title(f"Task {k + 1}", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(stages)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("After training on task", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Success rate", fontsize=9)
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_dir, "task_trajectories.png")


def plot_learning_curves(
    auc_logs: list, eval_every: int, key: str, output_dir: str, title: str = None
) -> str:
    """
    Small multiples with the per-epoch success rate or training loss of each task while it is trained on
    Args:
        auc_logs: per-epoch curves loaded by load_auc_logs
        eval_every: number of epochs between two evaluations
        key: "success" or "loss"
        output_dir: directory the figure is saved into
        title: optional figure title

    Returns:
    Path of the saved figure
    """
    n_tasks = len(auc_logs)
    n_cols = min(5, n_tasks)
    n_rows = int(np.ceil(n_tasks / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 2.2 * n_rows), sharex=True, sharey=True, squeeze=False
    )
    for k, ax in enumerate(axes.flat):
        if k >= n_tasks:
            ax.set_visible(False)
            continue
        values = auc_logs[k][key]
        epochs = np.arange(len(values)) * eval_every
        ax.plot(epochs, values, color=SERIES[0], linewidth=2)
        if key == "success":
            best = values.argmax()
            ax.plot(epochs[best], values[best], "o", color=SERIES[1], markersize=6)
            ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Task {k + 1}", fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Epoch", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Success rate" if key == "success" else "Training loss", fontsize=9)
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_dir, f"learning_curves_{key}.png")


def write_per_task_csv(per_task: dict, output_dir: str) -> str:
    """
    Write all per-task metrics into a csv file
    Args:
        per_task: per-task metrics from compute_metrics
        output_dir: directory the csv is saved into

    Returns:
    Path of the csv file
    """
    output_file = os.path.join(output_dir, "per_task_metrics.csv")
    keys = list(per_task.keys())
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task"] + keys)
        for k in range(len(per_task["fwt"])):
            writer.writerow([k + 1] + [per_task[key][k] for key in keys])
    return output_file


def fmt(value: float, digits: int = 2) -> str:
    """Format a float for the markdown report, NaN becomes a dash"""
    return "–" if np.isnan(value) else f"{value:.{digits}f}"


def markdown_table(header: list, rows: list) -> str:
    """Build a markdown table from a header and a list of rows"""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(
    metrics: dict,
    results: dict,
    figures: dict,
    output_dir: str,
    results_file: str,
    n_eval: int,
    eval_every: int,
    config: dict = None,
    task_names: list = None,
    title: str = None,
) -> str:
    """
    Write a markdown report with all metrics and figures
    Args:
        metrics: metrics from compute_metrics
        results: dict loaded by load_results
        figures: dict mapping figure keys to saved figure paths
        output_dir: directory the report is saved into
        results_file: path of the result.pt the report is based on
        n_eval: number of evaluation episodes per task
        eval_every: number of epochs between two evaluations
        config: config loaded by load_config (optional)
        task_names: task names loaded by load_task_names (optional)
        title: optional report title

    Returns:
    Path of the report
    """
    summary, per_task = metrics["summary"], metrics["per_task"]
    n_tasks = summary["n_tasks"]
    lines = [f"# {title or 'LIBERO results'}", ""]

    # headline metrics
    lines += [
        "## Main metrics",
        "",
        markdown_table(
            ["Metric", "Value", "Std. over tasks"],
            [
                ["**FWT** ↑ (forward transfer)", f"**{fmt(summary['fwt'], 3)}**", fmt(summary["fwt_std"], 3)],
                ["**NBT** ↓ (negative backward transfer)", f"**{fmt(summary['nbt'], 3)}**", fmt(summary["nbt_std"], 3)],
                ["**AUC** ↑ (area under the success curve)", f"**{fmt(summary['auc'], 3)}**", fmt(summary["auc_std"], 3)],
            ],
        ),
        "",
    ]

    # run information
    info = [["Results file", f"`{os.path.abspath(results_file)}`"]]
    if config is not None:
        info += [
            ["Benchmark", config.get("benchmark_name")],
            ["Algorithm", config["lifelong"]["algo"]],
            ["Policy", config["policy"]["policy_type"]],
            ["Seed", config.get("seed")],
            ["Epochs per task", config["train"]["n_epochs"]],
        ]
    info += [
        ["Tasks", n_tasks],
        ["Eval episodes per task", n_eval],
        ["Eval every (epochs)", eval_every],
    ]
    lines += ["## Run", "", markdown_table(["", ""], info), ""]

    # additional summary metrics
    extra = [
        ["Mean success right after learning a task", fmt(summary["mean_success_after_learning"], 3)],
        ["Average success over all tasks at the end", fmt(summary["final_avg_success"], 3)],
        ["Average forgetting (peak − final)", fmt(summary["avg_forgetting"], 3)],
        ["Maximum forgetting of a single task", fmt(summary["max_forgetting"], 3)],
        ["Mean forward-transfer loss (L_fwd)", fmt(summary["fwd_loss"], 3)],
    ]
    if "zero_shot_success" in summary:
        extra += [
            ["Mean zero-shot success (epoch 0)", fmt(summary["zero_shot_success"], 3)],
            ["Mean best success during training", fmt(summary["best_train_success"], 3)],
        ]
    lines += ["## Further metrics", "", markdown_table(["Metric", "Value"], extra), ""]

    # figures
    captions = {
        "success_matrix": (
            "Success matrix",
            "Success on every seen task (rows) after training on each task (columns). "
            f"Cells show successful episodes out of {n_eval}.",
        ),
        "per_task_success": (
            "Success per task",
            "FWT_k, success right after learning task k and success after learning all tasks. "
            "Error bars are 95% confidence intervals.",
        ),
        "success_over_stages": (
            "Success over training stages",
            "Success on the task that was just learned and the average success over all seen tasks.",
        ),
        "task_trajectories": (
            "Task trajectories",
            "Success of each task over all stages after it was learned (dotted line = stage it was learned in).",
        ),
        "per_task_transfer": ("NBT and AUC per task", "Dashed line marks the mean over tasks."),
        "learning_curves_success": (
            "Learning curves (success)",
            f"Success on the current task every {eval_every} epochs while training on it. "
            "The dot marks the checkpoint that was kept.",
        ),
        "learning_curves_loss": (
            "Learning curves (loss)",
            f"Training loss on the current task every {eval_every} epochs.",
        ),
        "loss_matrix": (
            "Loss matrix",
            "Behavior cloning loss on every seen task (rows) after training on each task (columns).",
        ),
    }
    lines += ["## Figures", ""]
    for key, (heading, caption) in captions.items():
        if key not in figures:
            continue
        rel_path = os.path.relpath(figures[key], output_dir)
        lines += [f"### {heading}", "", f"![{heading}]({rel_path})", "", caption, ""]

    # per-task table
    header = ["Task", "FWT_k", "After learning", "Final", "Forgetting", "NBT_k", "AUC_k"]
    if "zero_shot_success" in per_task:
        header += ["Zero-shot", "Best in training", "Best epoch"]
    rows = []
    for k in range(n_tasks):
        row = [
            k + 1,
            fmt(per_task["fwt"][k]),
            f"{fmt(per_task['success_after_learning'][k])} ± {fmt(per_task['success_after_learning_ci'][k])}",
            f"{fmt(per_task['final_success'][k])} ± {fmt(per_task['final_success_ci'][k])}",
            fmt(per_task["forgetting"][k]),
            fmt(per_task["nbt"][k]),
            fmt(per_task["auc"][k]),
        ]
        if "zero_shot_success" in per_task:
            row += [
                fmt(per_task["zero_shot_success"][k]),
                fmt(per_task["best_train_success"][k]),
                int(per_task["best_eval_idx"][k]) * eval_every,
            ]
        rows.append(row)
    lines += ["## Per-task metrics", "", markdown_table(header, rows), ""]

    if task_names is not None:
        lines += [
            "### Task names",
            "",
            markdown_table(["Task", "Name"], [[k + 1, name.replace("_", " ")] for k, name in enumerate(task_names)]),
            "",
        ]

    # raw success matrix
    success = np.asarray(results["S_conf_mat"], dtype=float)
    rows = [
        [f"after task {i + 1}"] + [fmt(success[i, j]) if j <= i else "" for j in range(n_tasks)]
        for i in range(n_tasks)
    ]
    lines += [
        "## Success matrix",
        "",
        "Row i, column j = success on task j after training on task i.",
        "",
        markdown_table([""] + [f"T{j + 1}" for j in range(n_tasks)], rows),
        "",
    ]

    # definitions
    lines += [
        "## Metric definitions",
        "",
        "c_{t,k} is the success on task k after training on task t (the success matrix above).",
        "",
        "- **FWT_k**: success on task k averaged over all evaluations while training on it "
        f"(every {eval_every} epochs, including epoch 0). After the best checkpoint all values are "
        "replaced by the best value, as if training stopped there. FWT is the mean over tasks.",
        "- **NBT_k** = mean over t > k of (c_{k,k} − c_{t,k}): how much success on task k drops after "
        "learning later tasks. NBT is the mean over all tasks but the last.",
        "- **AUC_k** = (FWT_k + Σ_{t>k} c_{t,k}) / (K − k + 1): average success on task k from the moment "
        "it is learned until the end. AUC is the mean over tasks.",
        "- **Forgetting_k** = max over t in [k, K−1) of c_{t,k} − c_{K,k}: drop from the peak to the final success.",
        f"- **Confidence intervals**: 95% normal approximation, 1.96·sqrt(p(1−p)/{n_eval}).",
        "",
    ]

    output_file = os.path.join(output_dir, "report.md")
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    return output_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_file", required=True, help="path to a LIBERO result.pt")
    parser.add_argument("--output_dir", required=True, help="directory to save the report and plots into")
    parser.add_argument("--n_eval", type=int, default=None, help="eval episodes per task, read from config.json if not given (default 20)")
    parser.add_argument("--eval_every", type=int, default=None, help="epochs between evaluations, read from config.json if not given (default 5)")
    parser.add_argument("--title", default=None, help="title of the report and plots")
    args = parser.parse_args()

    run_dir = os.path.dirname(os.path.abspath(args.results_file))
    results = load_results(args.results_file)
    n_tasks = np.asarray(results["S_conf_mat"]).shape[0]
    config = load_config(run_dir)
    auc_logs = load_auc_logs(run_dir, n_tasks)
    task_names = load_task_names(config)

    n_eval = args.n_eval or (config["eval"]["n_eval"] if config else 20)
    eval_every = args.eval_every or (config["eval"]["eval_every"] if config else 5)
    if config is None:
        print("[warning] no config.json next to result.pt, run information is missing")
    if auc_logs is None:
        print("[warning] no task{k}_auc.log files next to result.pt, learning curves are skipped")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics = compute_metrics(results, n_eval, auc_logs)
    summary = metrics["summary"]
    print(f"[info] Forward Transfer: {summary['fwt']:.3f}")
    print(f"[info] Negative Backward Transfer: {summary['nbt']:.3f}")
    print(f"[info] AUC: {summary['auc']:.3f}")

    figures = {
        "success_matrix": plot_matrix(
            results["S_conf_mat"],
            args.output_dir,
            "success_matrix.png",
            cmap="Blues",
            colorbar_label="Success rate",
            cell_label=lambda v: f"{int(round(v * n_eval))}/{n_eval}",
            vmin=0.0,
            vmax=1.0,
            title=args.title,
        ),
        "per_task_success": plot_per_task_success(metrics["per_task"], args.output_dir, args.title),
        "success_over_stages": plot_success_over_stages(metrics["per_task"], args.output_dir, args.title),
        "task_trajectories": plot_task_trajectories(results, args.output_dir, args.title),
        "per_task_transfer": plot_per_task_transfer(metrics["per_task"], args.output_dir, args.title),
        "loss_matrix": plot_matrix(
            results["L_conf_mat"],
            args.output_dir,
            "loss_matrix.png",
            cmap="Purples",
            colorbar_label="BC loss",
            cell_label=lambda v: f"{v:.0f}",
            title=args.title,
        ),
    }
    if auc_logs is not None:
        for key in ["success", "loss"]:
            figures[f"learning_curves_{key}"] = plot_learning_curves(
                auc_logs, eval_every, key, args.output_dir, args.title
            )

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)
    write_per_task_csv(metrics["per_task"], args.output_dir)
    report = write_report(
        metrics,
        results,
        figures,
        args.output_dir,
        args.results_file,
        n_eval,
        eval_every,
        config,
        task_names,
        args.title,
    )
    print(f"[info] wrote report to {report}")


if __name__ == "__main__":
    main()
