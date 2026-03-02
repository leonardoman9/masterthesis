#!/usr/bin/env python3
"""
Generate analysis assets for thesis figures/tables:
- Row-normalized confusion matrices for selected runs.
- Genus-aggregated confusion matrix for the 70(+1) run.
- Top-10 symmetric confusion pairs (semi-learnable vs fully learnable).
- Static frequency-mapping figure based on the HTML report parameters.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ConfusionData:
    labels: List[str]
    matrix: np.ndarray


def normalize_label(label: str) -> str:
    label = label.strip().replace("_", " ")
    if label.lower() in {"non-bird", "non bird", "no bird"}:
        return "no bird"
    return label


def genus_of(label: str) -> str:
    if label == "no bird":
        return "no bird"
    return label.split()[0]


def load_confusion_csv(path: Path) -> ConfusionData:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    labels = [normalize_label(r[0]) for r in rows[1:]]
    matrix = np.array([[float(x) for x in r[1:]] for r in rows[1:]], dtype=float)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Confusion matrix is not square: {path} -> {matrix.shape}")
    if matrix.shape[0] != len(labels):
        raise ValueError(f"Label count mismatch in {path}")
    return ConfusionData(labels=labels, matrix=matrix)


def row_normalized_percent(matrix: np.ndarray) -> np.ndarray:
    row_sum = matrix.sum(axis=1, keepdims=True)
    row_sum_safe = np.where(row_sum == 0, 1.0, row_sum)
    return (matrix / row_sum_safe) * 100.0


def plot_confusion_heatmap(
    matrix_percent: np.ndarray,
    labels: Sequence[str],
    out_path: Path,
    title: str,
    annotate: bool = False,
) -> None:
    n = len(labels)
    fig_size = max(8.0, min(16.0, 0.62 * n))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Emphasize off-diagonal structure: derive color upper bound from non-zero
    # off-diagonal cells so small confusions are not collapsed into dark tones.
    offdiag_mask = ~np.eye(n, dtype=bool)
    offdiag_vals = matrix_percent[offdiag_mask]
    offdiag_nonzero = offdiag_vals[offdiag_vals > 0]
    if offdiag_nonzero.size > 0:
        vmax = float(np.clip(np.percentile(offdiag_nonzero, 99) * 1.25, 8.0, 35.0))
    else:
        vmax = 100.0

    im = ax.imshow(matrix_percent, cmap="YlGnBu", vmin=0.0, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized confusion (%) [off-diagonal emphasized]")

    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.35, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate and n <= 25:
        if n <= 12:
            anno_font = 6
            value_fmt = "{:.0f}"
        elif n <= 18:
            anno_font = 5
            value_fmt = "{:.1f}"
        else:
            anno_font = 4
            value_fmt = "{:.1f}"
        for i in range(n):
            for j in range(n):
                value = matrix_percent[i, j]
                text_color = "white" if value > (0.62 * vmax) else "black"
                ax.text(
                    j,
                    i,
                    value_fmt.format(value),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=anno_font,
                )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate_genus_matrix(
    labels: Sequence[str],
    matrix: np.ndarray,
    top_n_genera: int = 18,
) -> Tuple[List[str], np.ndarray]:
    supports = matrix.sum(axis=1)
    genus_support: Dict[str, float] = {}
    for label, support in zip(labels, supports):
        genus = genus_of(label)
        genus_support[genus] = genus_support.get(genus, 0.0) + float(support)

    sorted_genera = sorted(
        [g for g in genus_support if g != "no bird"],
        key=lambda g: genus_support[g],
        reverse=True,
    )
    selected = sorted_genera[:top_n_genera]

    groups: List[str] = selected + ["Other genera"]
    if "no bird" in genus_support:
        groups.append("no bird")
    group_index = {g: i for i, g in enumerate(groups)}

    def map_to_group(label: str) -> str:
        g = genus_of(label)
        if g == "no bird":
            return "no bird"
        return g if g in selected else "Other genera"

    gmat = np.zeros((len(groups), len(groups)), dtype=float)
    for i, true_label in enumerate(labels):
        gi = group_index[map_to_group(true_label)]
        for j, pred_label in enumerate(labels):
            gj = group_index[map_to_group(pred_label)]
            gmat[gi, gj] += matrix[i, j]
    return groups, gmat


def align_matrix_by_labels(
    labels_ref: Sequence[str], labels_other: Sequence[str], matrix_other: np.ndarray
) -> np.ndarray:
    idx = {label: i for i, label in enumerate(labels_other)}
    order = [idx[label] for label in labels_ref]
    return matrix_other[np.ix_(order, order)]


def compute_top_confusion_pairs(
    labels: Sequence[str],
    semi_matrix: np.ndarray,
    fully_matrix: np.ndarray,
    top_k: int = 10,
) -> List[Dict[str, float | str]]:
    semi_support = semi_matrix.sum(axis=1)
    fully_support = fully_matrix.sum(axis=1)

    rows: List[Dict[str, float | str]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            semi_count = float(semi_matrix[i, j] + semi_matrix[j, i])
            fully_count = float(fully_matrix[i, j] + fully_matrix[j, i])
            semi_den = float(semi_support[i] + semi_support[j])
            fully_den = float(fully_support[i] + fully_support[j])
            semi_pct = (semi_count / semi_den * 100.0) if semi_den > 0 else 0.0
            fully_pct = (fully_count / fully_den * 100.0) if fully_den > 0 else 0.0
            rows.append(
                {
                    "pair": f"{labels[i]} <-> {labels[j]}",
                    "semi_count": semi_count,
                    "semi_pct": semi_pct,
                    "fully_count": fully_count,
                    "fully_pct": fully_pct,
                    "delta_count": fully_count - semi_count,
                    "delta_pct": fully_pct - semi_pct,
                    "rank_score": semi_count + fully_count,
                }
            )

    rows.sort(key=lambda r: (r["rank_score"], r["fully_count"]), reverse=True)
    return rows[:top_k]


def write_top_pairs_csv(rows: Sequence[Dict[str, float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pair",
                "semi_count",
                "semi_pct",
                "fully_count",
                "fully_pct",
                "delta_count",
                "delta_pct",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["pair"],
                    int(r["semi_count"]),
                    f"{r['semi_pct']:.2f}",
                    int(r["fully_count"]),
                    f"{r['fully_pct']:.2f}",
                    int(r["delta_count"]),
                    f"{r['delta_pct']:+.2f}",
                ]
            )


def latex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def write_top_pairs_latex(rows: Sequence[Dict[str, float | str]], out_path: Path) -> None:
    lines = []
    lines.append(r"\begin{table}[p]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Top-10 symmetric confusion pairs on the controlled \(8(+1)\) subset: semi-learnable baseline versus fully learnable front-end. Percentages are pair confusion rates normalized by the two-class support of each pair.}"
    )
    lines.append(r"\label{tab:top10_confusion_pairs}")
    lines.append(r"\begin{tabular}{|p{4.9cm}|r|r|r|r|r|}")
    lines.append(r"\hline")
    lines.append(
        r"Species pair & Semi cnt & Semi \% & Fully cnt & Fully \% & \(\Delta\) Fully-Semi \\"
    )
    lines.append(r"\hline")
    for r in rows:
        pair = latex_escape(str(r["pair"])).replace("<->", r"\(\leftrightarrow\)")
        delta = f"{int(r['delta_count']):+d} ({float(r['delta_pct']):+.2f} pp)"
        line = (
            f"{pair} & {int(r['semi_count'])} & {float(r['semi_pct']):.2f} & "
            f"{int(r['fully_count'])} & {float(r['fully_pct']):.2f} & {latex_escape(delta)} \\\\"
        )
        lines.append(line)
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sigmoid(x: np.ndarray, breakpoint_hz: float, width: float, f_min: float, f_max: float) -> np.ndarray:
    nb = (breakpoint_hz - f_min) / (f_max - f_min)
    nx = (x - f_min) / (f_max - f_min)
    return 1.0 / (1.0 + np.exp(-width * (nx - nb)))


def adaptive_mapping(x: np.ndarray, breakpoint_hz: float, width: float, f_min: float, f_max: float) -> np.ndarray:
    log_mapping = f_min * np.power(f_max / f_min, (x - f_min) / (f_max - f_min))
    linear_mapping = x
    s = sigmoid(x, breakpoint_hz, width, f_min, f_max)
    return (1.0 - s) * log_mapping + s * linear_mapping


def generate_frequency_mapping_figure(out_path: Path) -> None:
    f_min = 50.0
    f_max = 16000.0
    x = np.linspace(f_min, f_max, 400)

    mel = f_min * np.power(f_max / f_min, (x - f_min) / (f_max - f_min))
    linear = x

    easy = adaptive_mapping(x, breakpoint_hz=1.5, width=6.0, f_min=f_min, f_max=f_max)
    hard = adaptive_mapping(x, breakpoint_hz=1224.0, width=27.4, f_min=f_min, f_max=f_max)
    corvus = adaptive_mapping(x, breakpoint_hz=1955.0, width=51.7, f_min=f_min, f_max=f_max)
    upupa = adaptive_mapping(x, breakpoint_hz=5269.0, width=132.0, f_min=f_min, f_max=f_max)
    regulus = adaptive_mapping(x, breakpoint_hz=1221.0, width=28.34, f_min=f_min, f_max=f_max)
    full_dataset = adaptive_mapping(x, breakpoint_hz=851.0, width=19.0, f_min=f_min, f_max=f_max)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)

    def setup_axis(ax: plt.Axes, title: str) -> None:
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 16000)
        ax.set_ylim(0, 16000)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Input frequency (Hz)")
        ax.set_ylabel("Mapped frequency (Hz)")

    ax = axes[0, 0]
    ax.plot(x, mel, "--", color="#bdbdbd", linewidth=1.2, label="Mel reference")
    ax.plot(x, linear, "--", color="#757575", linewidth=1.2, label="Linear reference")
    ax.plot(x, easy, color="#4caf50", linewidth=2.2, label="Easy species")
    setup_axis(ax, "Easy Species vs References")

    ax = axes[0, 1]
    ax.plot(x, mel, "--", color="#bdbdbd", linewidth=1.2, label="Mel reference")
    ax.plot(x, linear, "--", color="#757575", linewidth=1.2, label="Linear reference")
    ax.plot(x, hard, color="#ff9800", linewidth=2.2, label="Hard species")
    ax.plot(x, regulus, color="#ff5722", linewidth=1.8, linestyle="-.", label="Regulus pair")
    setup_axis(ax, "Hard Species Group")

    ax = axes[1, 0]
    ax.plot(x, mel, "--", color="#bdbdbd", linewidth=1.2, label="Mel reference")
    ax.plot(x, linear, "--", color="#757575", linewidth=1.2, label="Linear reference")
    ax.plot(x, corvus, color="#2196f3", linewidth=2.0, label="Corvus")
    ax.plot(x, upupa, color="#3f51b5", linewidth=2.0, label="Upupa")
    setup_axis(ax, "Single Species Specialists")

    ax = axes[1, 1]
    ax.plot(x, easy, color="#4caf50", linewidth=2.0, label="Easy subset")
    ax.plot(x, hard, color="#ff9800", linewidth=2.0, label="Hard subset")
    ax.plot(x, full_dataset, color="#795548", linewidth=2.0, label="Full dataset")
    ax.plot(x, corvus, color="#2196f3", linewidth=1.5, label="Corvus")
    ax.plot(x, upupa, color="#3f51b5", linewidth=1.5, label="Upupa")
    setup_axis(ax, "All Learned Mappings Comparison")

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("Adaptive Frequency Mapping (0-16kHz): Static Reconstruction from HTML Parameters", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    thesis_root = Path(__file__).resolve().parents[1]
    repo_root = thesis_root.parent

    logs_root = repo_root / "backups" / "rf4423" / "logs"
    images_dir = thesis_root / "images"
    tables_dir = thesis_root / "tables"

    run_8_fully = logs_root / "adaptive_focal_fully_learnable_corrected" / "2025-07-15_10-43-04" / "confusion_matrix.csv"
    run_13_hard = logs_root / "hard_species_full_dataset" / "2025-07-28_12-25-13" / "confusion_matrix.csv"
    run_70_full = logs_root / "bird_full_training_fulldataset" / "2025-07-18_16-39-27" / "confusion_matrix.csv"
    run_8_semi = logs_root / "adaptive_focal_distillation" / "2025-06-25_09-10-09" / "confusion_matrix.csv"

    data_8_fully = load_confusion_csv(run_8_fully)
    data_13_hard = load_confusion_csv(run_13_hard)
    data_70_full = load_confusion_csv(run_70_full)
    data_8_semi = load_confusion_csv(run_8_semi)

    # 8(+1) fully learnable matrix
    cm8 = row_normalized_percent(data_8_fully.matrix)
    plot_confusion_heatmap(
        cm8,
        data_8_fully.labels,
        images_dir / "confmat_8plus1_fully_rownorm.png",
        "8(+1) Subset - Fully Learnable (Row-normalized %)",
        annotate=True,
    )

    # 13(+1) hard subset matrix
    cm13 = row_normalized_percent(data_13_hard.matrix)
    plot_confusion_heatmap(
        cm13,
        data_13_hard.labels,
        images_dir / "confmat_13plus1_hard_rownorm.png",
        "13(+1) Hard Subset - Semi-learnable (Row-normalized %)",
        annotate=True,
    )

    # 70(+1) aggregated by genus
    group_labels, group_cm = aggregate_genus_matrix(data_70_full.labels, data_70_full.matrix, top_n_genera=18)
    group_cm_pct = row_normalized_percent(group_cm)
    plot_confusion_heatmap(
        group_cm_pct,
        group_labels,
        images_dir / "confmat_70plus1_genus_aggregated_rownorm.png",
        "70(+1) Full Run - Genus-level Aggregated Matrix (Top-18 genera + Other + no bird)",
        annotate=True,
    )

    # Top-10 pairwise confusions: semi vs fully on controlled 8(+1)
    if data_8_semi.labels != data_8_fully.labels:
        aligned_fully = align_matrix_by_labels(data_8_semi.labels, data_8_fully.labels, data_8_fully.matrix)
        labels = data_8_semi.labels
    else:
        aligned_fully = data_8_fully.matrix
        labels = data_8_fully.labels
    top_pairs = compute_top_confusion_pairs(labels, data_8_semi.matrix, aligned_fully, top_k=10)
    write_top_pairs_csv(top_pairs, tables_dir / "top10_confusion_pairs.csv")
    write_top_pairs_latex(top_pairs, tables_dir / "top10_confusion_pairs.tex")

    # Frequency mapping static figure reconstructed from HTML parameters
    generate_frequency_mapping_figure(images_dir / "learned_frequency_mappings_16khz.png")

    print("Generated files:")
    print(" - images/confmat_8plus1_fully_rownorm.png")
    print(" - images/confmat_13plus1_hard_rownorm.png")
    print(" - images/confmat_70plus1_genus_aggregated_rownorm.png")
    print(" - images/learned_frequency_mappings_16khz.png")
    print(" - tables/top10_confusion_pairs.csv")
    print(" - tables/top10_confusion_pairs.tex")


if __name__ == "__main__":
    main()
