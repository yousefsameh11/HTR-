"""Visualization utilities for HTR experiment results.

All figures are written to ``results/figures`` (PNG, Agg backend) and every
function tolerates a model that has not been trained yet: missing metrics simply
do not appear in the chart.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_comparison_bar(
    metrics: Dict[str, Dict],
    metric: str,
    save_path: str,
    title: Optional[str] = None,
) -> str:
    """Bar chart comparing one metric across the models that have results."""
    _ensure_dir(os.path.dirname(os.path.abspath(save_path)))
    models = [m for m, values in metrics.items() if values.get(metric) is not None]
    values = [metrics[m][metric] for m in models]

    fig, ax = plt.subplots(figsize=(6, 4))
    if models:
        bars = ax.bar(models, values, color=COLORS[: len(models)])
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.4f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
    else:
        ax.text(0.5, 0.5, "No results available", ha="center", va="center")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} comparison (IAM-line test split)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_metric_group(
    metrics: Dict[str, Dict],
    save_path: str,
    title: str = "CER / WER comparison",
) -> str:
    """Grouped bar chart with both CER and WER per model."""
    _ensure_dir(os.path.dirname(os.path.abspath(save_path)))
    models = list(metrics.keys())
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    positions = range(len(models))
    for offset, (metric, color) in enumerate((("cer", COLORS[0]), ("wer", COLORS[1]))):
        values = [metrics[m].get(metric) or 0.0 for m in models]
        shift = (offset - 0.5) * width
        bars = ax.bar([p + shift for p in positions], values, width, label=metric.upper(), color=color)
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(models)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_loss_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Training & Validation Loss",
) -> str:
    """Plot training and validation loss over epochs."""
    _ensure_dir(os.path.dirname(os.path.abspath(save_path)))
    train = list(history.get("train_loss", []))
    val = list(history.get("val_loss", []))
    # Histories can differ by one entry (e.g. a final eval appended after the
    # last epoch); plot the common prefix instead of failing.
    if train and val:
        n = min(len(train), len(val))
        train, val = train[:n], val[:n]
    epochs = list(range(1, max(len(train), len(val)) + 1))

    fig, ax = plt.subplots(figsize=(6, 4))
    if train:
        ax.plot(epochs, train, marker="o", label="Train loss", color=COLORS[0])
    if val:
        ax.plot(epochs, val, marker="o", label="Val loss", color=COLORS[1])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_time_comparison(
    times: Dict[str, float],
    save_path: str,
    title: str = "Total training time",
    ylabel: str = "Seconds",
) -> str:
    """Bar chart of a timing measurement per model."""
    _ensure_dir(os.path.dirname(os.path.abspath(save_path)))
    models = [m for m, value in times.items() if value]
    values = [times[m] for m in models]

    fig, ax = plt.subplots(figsize=(6, 4))
    if models:
        bars = ax.bar(models, values, color=COLORS[: len(models)])
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.1f}s",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        ax.text(0.5, 0.5, "No results available", ha="center", va="center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_prediction_examples(
    details: List[Dict],
    save_path: str,
    max_examples: int = 8,
    title: str = "Prediction examples (ground truth vs predicted)",
) -> str:
    """Render a few ground-truth/hypothesis pairs as a text figure."""
    _ensure_dir(os.path.dirname(os.path.abspath(save_path)))
    examples = details[:max_examples]
    fig_height = max(2, 0.55 * len(examples) + 1)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.axis("off")
    lines = []
    for i, row in enumerate(examples):
        lines.append(f"[{i}] GT : {row['reference']}")
        lines.append(
            f"     PR : {row['hypothesis']}   (CER={row['cer']:.2f}, WER={row['wer']:.2f})"
        )
        lines.append("")
    ax.text(
        0.0,
        1.0,
        "\n".join(lines) if lines else "No predictions available",
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        va="top",
    )
    ax.set_title(title, pad=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_predictions_with_images(
    records: List[Dict],
    save_path: str,
    title: str = "Image | Ground truth | CRNN | TrOCR",
) -> str:
    """Table figure: line image, ground truth, CRNN and TrOCR predictions.

    ``records`` items: {"image": PIL.Image, "reference": str,
                        "crnn": str | None, "trocr": str | None}
    """
    _ensure_dir(os.path.dirname(os.path.abspath(save_path)))
    if not records:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No prediction records available", ha="center")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    n = len(records)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n + 0.6))
    if n == 1:
        axes = [axes]
    for ax, record in zip(axes, records):
        image = record["image"]
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        text = (
            f"GT   : {record['reference']}\n"
            f"CRNN : {record.get('crnn') or '-'}\n"
            f"TrOCR: {record.get('trocr') or '-'}"
        )
        ax.text(
            0.0,
            -0.08,
            text,
            transform=ax.transAxes,
            fontsize=8,
            fontfamily="monospace",
            va="top",
        )
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def make_all_figures(
    metrics: Dict[str, Dict],
    histories: Optional[Dict[str, Dict]] = None,
    training_times: Optional[Dict[str, float]] = None,
    figures_dir: str = "experiments/figures",
) -> Dict[str, str]:
    """Generate the standard comparison figures for whatever results exist."""
    _ensure_dir(figures_dir)
    paths: Dict[str, str] = {}

    paths["cer"] = plot_comparison_bar(
        metrics, "cer", os.path.join(figures_dir, "cer_comparison.png")
    )
    paths["wer"] = plot_comparison_bar(
        metrics, "wer", os.path.join(figures_dir, "wer_comparison.png")
    )
    if len(metrics) > 1:
        paths["grouped"] = plot_metric_group(
            metrics, os.path.join(figures_dir, "cer_wer_comparison.png")
        )

    for name, history in (histories or {}).items():
        if history and history.get("train_loss"):
            paths[f"{name.lower()}_loss"] = plot_loss_curves(
                history,
                os.path.join(figures_dir, f"{name.lower()}_loss.png"),
                title=f"{name} training & validation loss",
            )

    if training_times:
        paths["training_time"] = plot_time_comparison(
            training_times,
            os.path.join(figures_dir, "training_time.png"),
            title="Total training time",
        )

    return paths
