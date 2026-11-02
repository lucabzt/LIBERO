"""
Summarizes the Memory Anchor selections of an AnchorER run into <run>/anchorER/report.md and
plots next to it. AnchorER (libero/lifelong/algos/anchor_er) writes one folder per selection,
anchorER/task<k>/ (selection.json, selection.npz and per-task plots), at the start of every task
after the first. This report only covers the selection (overlap, disagreement, anchors, timing);
success metrics are in the run's report.md.

libero/lifelong/main.py writes it after every task of a run with an anchorER folder. For an
existing run:
    python -m libero.lifelong.report.anchor_er_report --run_dir <run>
"""

import argparse
import json
import os

import numpy as np

from libero.lifelong.report.interpret_results import (
    MUTED,
    SERIES,
    TEXT,
    fmt,
    load_config,
    markdown_table,
    plt,
    save_figure,
)

ANCHOR_ER_DIR = "anchorER"


def _pct(part, total):
    return 100.0 * part / total if total else float("nan")


def selection_percentages(info: dict) -> dict:
    """
    Percentages of the counts in a selection.json dict (all in %):
        overlap_pct: new transitions in the overlap cluster, of all new transitions
        disagreement_pct_of_overlap / _of_new: disagreeing new transitions (step 2)
        buffer_pct_of_old: buffer size, of all past transitions
        anchor_pct_of_buffer / _of_old: anchors, of the buffer / of all past transitions
        anchors_per_task_pct, buffer_per_task_pct, old_per_task_pct: share of every past task
            in the anchors, the buffer and all past data
    """
    out = {
        "overlap_pct": _pct(info["n_overlap"], info["n_new"]),
        "disagreement_pct_of_overlap": _pct(info["n_disagreement"], info["n_overlap"]),
        "disagreement_pct_of_new": _pct(info["n_disagreement"], info["n_new"]),
        "anchor_pct_of_old": _pct(info["n_anchors"], info["n_old"]),
    }
    if "n_buffer" in info:
        out["buffer_pct_of_old"] = _pct(info["n_buffer"], info["n_old"])
        out["anchor_pct_of_buffer"] = _pct(info["n_anchors"], info["n_buffer"])
    for key, total in [
        ("anchors_per_task", info["n_anchors"]),
        ("buffer_per_task", info.get("n_buffer")),
        ("old_per_task", info["n_old"]),
    ]:
        if key in info and total:
            out[f"{key}_pct"] = [_pct(n, total) for n in info[key]]
    return out


def load_selections(run_dir: str) -> list:
    """
    Selections of a run sorted by task, each the selection.json dict plus "dir" and the
    arrays of selection.npz under "arrays". Counts missing in selection.json files of older
    runs are recomputed from the arrays.
    """
    base = os.path.join(run_dir, ANCHOR_ER_DIR)
    if not os.path.isdir(base):
        return []
    selections = []
    for name in os.listdir(base):
        task_dir = os.path.join(base, name)
        json_file = os.path.join(task_dir, "selection.json")
        if not (name.startswith("task") and os.path.isfile(json_file)):
            continue
        with open(json_file) as f:
            info = json.load(f)
        npz_file = os.path.join(task_dir, "selection.npz")
        arrays = dict(np.load(npz_file)) if os.path.isfile(npz_file) else {}
        if "n_buffer" not in info and "buffer_indices" in arrays:
            info["n_buffer"] = int(len(arrays["buffer_indices"]))
        if "old_per_task" not in info and "old_task" in arrays:
            info["old_per_task"] = np.bincount(
                arrays["old_task"], minlength=len(info["past_tasks"])
            ).tolist()
        info.update(selection_percentages(info))
        info["dir"] = name
        info["arrays"] = arrays
        selections.append(info)
    return sorted(selections, key=lambda s: s["task"])


def share_matrix(selections: list, key: str, n_tasks: int) -> np.ndarray:
    """[new task, past task] matrix of a per-past-task percentage list (NaN where undefined)."""
    matrix = np.full((n_tasks, n_tasks), np.nan)
    for s in selections:
        values = s.get(key)
        if values is not None:
            matrix[s["task"], : len(values)] = values
    return matrix


def plot_selection_funnel(selections: list, output_dir: str) -> str:
    """Overlapping and disagreeing share of the new data for every selection."""
    tasks = np.array([s["task"] + 1 for s in selections])
    overlap = np.array([s["overlap_pct"] for s in selections])
    disagree = np.array([s["disagreement_pct_of_new"] for s in selections])
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(tasks) + 2), 3.8))
    bars = [
        ax.bar(tasks - width / 2, overlap, width, color=SERIES[0], label="overlap (step 1)"),
        ax.bar(tasks + width / 2, disagree, width, color=SERIES[1], label="disagreement (step 2)"),
    ]
    for bar_group in bars:
        for bar in bar_group:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{bar.get_height():.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=TEXT,
            )
    ax.set_xticks(tasks)
    ax.set_xlabel("New task")
    ax.set_ylabel("% of the new task's transitions")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Selected share of the new data", fontweight="bold")
    return save_figure(fig, output_dir, "selection_funnel.png")


def plot_task_matrix(
    matrix: np.ndarray, output_dir: str, name: str, cmap: str, label: str, title: str,
    cell_format: str, vmin: float = None, vmax: float = None, center: float = None,
) -> str:
    """Heatmap of a [new task, past task] matrix, only past tasks of every new task."""
    n_tasks = matrix.shape[0]
    rows = [k for k in range(n_tasks) if not np.all(np.isnan(matrix[k]))]
    values = matrix[rows][:, : max(rows)] if rows else matrix[:1, :1]
    masked = np.ma.masked_invalid(values)
    if center is not None:
        spread = max(abs(masked.max() - center), abs(center - masked.min()), 1e-12)
        vmin, vmax = center - spread, center + spread
    vmin = masked.min() if vmin is None else vmin
    vmax = masked.max() if vmax is None else vmax
    fig, ax = plt.subplots(figsize=(0.8 * values.shape[1] + 2.5, 0.6 * values.shape[0] + 1.5))
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("white")
    im = ax.imshow(masked, cmap=color_map, vmin=vmin, vmax=vmax, aspect="auto")
    ax.grid(False)
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            if np.isnan(values[r, c]):
                continue
            norm = (values[r, c] - vmin) / (vmax - vmin + 1e-12)
            ax.text(
                c, r, cell_format.format(values[r, c]), ha="center", va="center", fontsize=8,
                color="white" if norm > 0.6 or (center is not None and norm < 0.2) else TEXT,
            )
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels(range(1, values.shape[1] + 1))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([k + 1 for k in rows])
    ax.set_xlabel("Past task")
    ax.set_ylabel("New task")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
    return save_figure(fig, output_dir, name)


def plot_disagreement(selections: list, output_dir: str) -> str:
    """Disagreement of held-out past demos vs overlapping new data, with the threshold."""
    with_scores = [s for s in selections if "val_scores" in s["arrays"]]
    if not with_scores:
        return None
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(with_scores) + 2), 3.8))
    for i, s in enumerate(with_scores):
        parts = ax.boxplot(
            [s["arrays"]["val_scores"], s["arrays"]["overlap_scores"]],
            positions=[3 * i, 3 * i + 1],
            widths=0.8,
            showfliers=False,
            patch_artist=True,
        )
        for patch, color in zip(parts["boxes"], SERIES[:2]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.plot([3 * i - 0.5, 3 * i + 1.5], [s["threshold"]] * 2, color=TEXT, linestyle="--", linewidth=1)
    ax.set_xticks([3 * i + 0.5 for i in range(len(with_scores))])
    ax.set_xticklabels([s["task"] + 1 for s in with_scores])
    ax.set_xlabel("New task")
    ax.set_ylabel("Action disagreement")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SERIES[0], alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color=SERIES[1], alpha=0.7),
        plt.Line2D([0], [0], color=TEXT, linestyle="--"),
    ]
    ax.legend(handles, ["held-out past demos", "overlapping new data", "threshold μ + c·σ"], fontsize=8)
    ax.set_title("Action disagreement (boxes: quartiles, whiskers: 1.5 IQR)", fontweight="bold")
    return save_figure(fig, output_dir, "disagreement.png")


def plot_timings(selections: list, output_dir: str) -> str:
    """Time of every selection step per task."""
    steps = ["representations", "overlap", "disagreement", "anchors"]
    tasks = np.array([s["task"] + 1 for s in selections])
    colors = SERIES + [MUTED]
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(tasks) + 2), 3.5))
    bottom = np.zeros(len(tasks))
    for step, color in zip(steps, colors):
        values = np.array([s["timings_s"].get(step, 0.0) / 60 for s in selections])
        ax.bar(tasks, values, bottom=bottom, color=color, label=step, width=0.6)
        bottom += values
    ax.set_xticks(tasks)
    ax.set_xlabel("New task")
    ax.set_ylabel("Minutes")
    ax.legend(fontsize=8)
    ax.set_title("Selection time", fontweight="bold")
    return save_figure(fig, output_dir, "selection_time.png")


def write_anchor_er_report(run_dir: str) -> str:
    """
    Write <run_dir>/anchorER/report.md and the summary plots from the task<k>/ selections.
    Returns the path of report.md, or None if there is no selection yet.
    """
    selections = load_selections(run_dir)
    if not selections:
        return None
    output_dir = os.path.join(run_dir, ANCHOR_ER_DIR)
    config = load_config(run_dir) or {}
    lifelong = config.get("lifelong", {})
    n_tasks = max(s["task"] for s in selections) + 1
    task_names = selections[-1]["past_tasks"] + [selections[-1]["new_task"]]

    funnel = plot_selection_funnel(selections, output_dir)
    anchors_share = share_matrix(selections, "anchors_per_task_pct", n_tasks)
    old_share = share_matrix(selections, "old_per_task_pct", n_tasks)
    anchors_plot = plot_task_matrix(
        anchors_share, output_dir, "anchors_per_task.png", "Blues", "% of the anchors",
        "Where the anchors come from", "{:.0f}", vmin=0,
    )
    enrichment = anchors_share / old_share
    enrichment_plot = plot_task_matrix(
        np.log2(enrichment), output_dir, "anchor_enrichment.png", "RdBu_r",
        "log2(anchor share / past-data share)", "Anchor enrichment per past task", "{:+.1f}",
        center=0.0,
    )
    disagreement_plot = plot_disagreement(selections, output_dir)
    timing_plot = plot_timings(selections, output_dir)

    title = f"{config.get('benchmark_name', '')} – AnchorER selection".strip(" –")
    lines = [f"# {title}", ""]
    lines += [
        "Memory Anchor selection at the start of every task after the first (paper Alg. 1). "
        "Success metrics are in [../report.md](../report.md).",
        "",
    ]

    params = [
        "mode", "anchor_fraction", "remove_fraction", "buffer_fraction", "n_memories",
        "use_overlap", "use_disagreement", "representation_keys", "representation_normalization",
        "disagreement_metric",
        "noise_level", "disagreement_inference_steps", "n_noise_samples", "std_cutoff",
        "n_neighbors",
    ]
    rows = [[p, lifelong[p]] for p in params if p in lifelong]
    if "data" in config:
        rows.append(["data.n_val_demos", config["data"].get("n_val_demos", "–")])
    if rows:
        lines += ["## Settings", "", markdown_table(["Parameter", "Value"], rows), ""]

    lines += ["## Selection per task", ""]
    rows = []
    for s in selections:
        anchor_cell = (
            f"{s['n_anchors']} ({fmt(s['anchor_pct_of_buffer'], 1)}% of buffer)"
            if s.get("mode", "add") == "add" and "anchor_pct_of_buffer" in s
            else f"{s['n_anchors']} removed ({fmt(s['anchor_pct_of_old'], 1)}% of past)"
        )
        rows.append([
            s["task"] + 1,
            s["n_old"],
            s["n_new"],
            f"{s['n_overlap']} ({fmt(s['overlap_pct'], 1)}%)",
            f"{s['n_disagreement']} ({fmt(s['disagreement_pct_of_overlap'], 1)}% / "
            f"{fmt(s['disagreement_pct_of_new'], 1)}%)",
            fmt(s.get("mu", np.nan), 4),
            fmt(s.get("sigma", np.nan), 4),
            fmt(s.get("threshold", np.nan), 4),
            "yes" if s.get("disagreement_fallback") else "no",
            f"{s.get('n_buffer', '–')} ({fmt(s.get('buffer_pct_of_old', np.nan), 2)}%)",
            anchor_cell,
            fmt(sum(s["timings_s"].values()) / 60, 1),
        ])
    lines += [
        markdown_table(
            [
                "New task", "Past transitions", "New transitions", "Overlap (% of new)",
                "Disagreement (% of overlap / % of new)", "μ", "σ", "Threshold", "Fallback",
                "Buffer (% of past)", "Anchors", "Time (min)",
            ],
            rows,
        ),
        "",
        "Fallback: no overlapping new transition exceeded the threshold, so all overlapping "
        "ones were used as disagreement set.",
        "",
    ]

    lines += ["## Anchors per past task", "", "Count (share of the anchors) per past task.", ""]
    n_past = n_tasks - 1
    rows = []
    for s in selections:
        cells = []
        for k in range(n_past):
            if k < len(s["anchors_per_task"]):
                cells.append(f"{s['anchors_per_task'][k]} ({fmt(s['anchors_per_task_pct'][k], 0)}%)")
            else:
                cells.append("")
        rows.append([s["task"] + 1] + cells)
    lines += [markdown_table(["New task"] + [f"P{k + 1}" for k in range(n_past)], rows), ""]

    lines += ["## Figures", ""]
    figures = [
        (funnel, "Overlap and disagreement", "Share of the new task's transitions in the overlap cluster (step 1) and above the disagreement threshold (step 2)."),
        (anchors_plot, "Anchors per past task", "Share of the anchors of every selection (row) that come from each past task (column)."),
        (enrichment_plot, "Anchor enrichment", "log2 of a past task's share of the anchors over its share of all past data: > 0 (red) means over-represented compared with uniform sampling (RandER)."),
        (disagreement_plot, "Action disagreement", "Held-out past demos (baseline) vs. the overlapping new data; the dashed line is the threshold."),
        (timing_plot, "Selection time", "Minutes per selection step."),
    ]
    for path, heading, caption in figures:
        if path is None:
            continue
        lines += [f"### {heading}", "", f"![{heading}]({os.path.basename(path)})", "", caption, ""]

    lines += ["## Per task", ""]
    for s in selections:
        d = s["dir"]
        lines += [
            f"### Task {s['task'] + 1}: {s['new_task']}",
            "",
            f"![partition]({d}/new_data_partition.png)",
            "",
        ]
        if os.path.isfile(os.path.join(output_dir, d, "disagreement.png")):
            lines += [f"![disagreement]({d}/disagreement.png)", ""]
        lines += [
            f"![anchors]({d}/anchors.png)",
            "",
            f"Image grids: [top 100 anchors]({d}/top100_anchors.png)"
            + (
                f", [top 100 disagreements]({d}/top100_disagreement.png)"
                if os.path.isfile(os.path.join(output_dir, d, "top100_disagreement.png"))
                else ""
            )
            + f". Raw data: [selection.json]({d}/selection.json), `{d}/selection.npz`.",
            "",
        ]

    lines += ["### Task names", "", markdown_table(["Task", "Name"], [[k + 1, n] for k, n in enumerate(task_names)]), ""]

    lines += [
        "## Definitions",
        "",
        "- **Overlap** (step 1): new transitions in the k-means cluster (2 clusters on the "
        "lower-tail distances to all past transitions in the normalized obs latent) with the "
        "smaller nearest-neighbor distance.",
        "- **Disagreement** (step 2): overlapping new transitions whose noise-denoise action "
        "disagreement exceeds μ + std_cutoff·σ of the held-out demos of the past tasks.",
        "- **Anchors** (step 3): past transitions with the lowest median distance to their "
        "n_neighbors nearest disagreeing new transitions. mode add: the best anchor_fraction of "
        "the buffer, the rest uniform; mode remove: the best remove_fraction of all past data, "
        "excluded before uniform sampling.",
        "",
    ]

    report = os.path.join(output_dir, "report.md")
    with open(report, "w") as f:
        f.write("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True, help="run directory containing anchorER/")
    args = parser.parse_args()
    report = write_anchor_er_report(args.run_dir)
    if report is None:
        print(f"[info] no AnchorER selections in {args.run_dir}")
    else:
        print(f"[info] wrote report to {report}")


if __name__ == "__main__":
    main()
