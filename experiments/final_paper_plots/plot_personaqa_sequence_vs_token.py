import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_personaqa_results_all_models import (
    FONT_SIZE_BAR_VALUE,
    FONT_SIZE_LEGEND,
    FONT_SIZE_SUBPLOT_TITLE,
    FONT_SIZE_Y_AXIS_LABEL,
    FONT_SIZE_Y_AXIS_TICK,
    HIGHLIGHT_KEYWORDS,
    MODEL_NAMES,
    MODELS,
    load_results_from_folder,
)
from shared_color_mapping import get_shared_palette

TASK_LABELS = {
    "yes_no": "Yes / No",
    "open_ended": "Open Ended",
}

IMAGE_FOLDER = "images"
SEQUENCE_VS_TOKEN_FOLDER = f"{IMAGE_FOLDER}/sequence_vs_token"
os.makedirs(SEQUENCE_VS_TOKEN_FOLDER, exist_ok=True)
OUTPUT_PATH_BASE = f"{SEQUENCE_VS_TOKEN_FOLDER}/personaqa_sequence_vs_token"

HATCH = "////"


def _full_dataset_stats(model: str, highlight_keyword: str, task_type: str, sequence: bool) -> tuple[float, float]:
    run_dir = Path(f"experiments/personaqa_results/{model}_{task_type}")
    is_open_ended = task_type == "open_ended"
    results = load_results_from_folder(run_dir, model, sequence=sequence, is_open_ended=is_open_ended, verbose=False)
    matches = [name for name in results if highlight_keyword in name]
    assert len(matches) == 1, f"Expected one match for {highlight_keyword} in {run_dir}, found {matches}"
    data = results[matches[0]]
    return float(data["accuracy"]), float(data["ci"])


def _gather_model_stats(model: str, highlight_keyword: str) -> list[tuple[str, float, float, float, float]]:
    stats = []
    for task_type in ("yes_no", "open_ended"):
        token_mean, token_ci = _full_dataset_stats(model, highlight_keyword, task_type, sequence=False)
        seq_mean, seq_ci = _full_dataset_stats(model, highlight_keyword, task_type, sequence=True)
        stats.append((TASK_LABELS[task_type], token_mean, token_ci, seq_mean, seq_ci))
    return stats


def _annotate_bars(ax, bars, means, cis):
    for bar, mean, ci in zip(bars, means, cis):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + ci + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_BAR_VALUE,
        )


def plot_sequence_vs_token(model_data: list[tuple[str, list[tuple[str, float, float, float, float]]]]):
    palette = get_shared_palette(list(TASK_LABELS.values()))
    fig, axes = plt.subplots(1, len(model_data), figsize=(6 * len(model_data), 6), sharey=True)
    if len(model_data) == 1:
        axes = [axes]

    width = 0.35

    for idx, (model_title, stats) in enumerate(model_data):
        ax = axes[idx]
        x = np.arange(len(stats))
        token_means = [s[1] for s in stats]
        token_cis = [s[2] for s in stats]
        seq_means = [s[3] for s in stats]
        seq_cis = [s[4] for s in stats]
        colors = [palette[s[0]] for s in stats]

        token_bars = ax.bar(
            x - width / 2.0, token_means, width, color=colors, yerr=token_cis, capsize=5, error_kw={"linewidth": 2}
        )
        seq_bars = ax.bar(
            x + width / 2.0,
            seq_means,
            width,
            color=colors,
            yerr=seq_cis,
            capsize=5,
            error_kw={"linewidth": 2},
            hatch=HATCH,
            edgecolor="black",
            linewidth=1.5,
        )

        _annotate_bars(ax, token_bars, token_means, token_cis)
        _annotate_bars(ax, seq_bars, seq_means, seq_cis)

        ax.set_title(model_title, fontsize=FONT_SIZE_SUBPLOT_TITLE)
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in stats], fontsize=FONT_SIZE_Y_AXIS_TICK)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=FONT_SIZE_Y_AXIS_TICK)
        if idx == 0:
            ax.set_ylabel("Average Accuracy", fontsize=FONT_SIZE_Y_AXIS_LABEL)

    task_handles = [Patch(facecolor=palette[label], edgecolor="black", label=label) for label in TASK_LABELS.values()]
    token_handle = Patch(facecolor="white", edgecolor="black", label="Single Token")
    sequence_handle = Patch(facecolor="white", edgecolor="black", hatch=HATCH, label="Full Sequence")
    fig.legend(
        handles=task_handles + [token_handle, sequence_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )
    plt.tight_layout()
    pdf_path = f"{OUTPUT_PATH_BASE}.pdf"
    png_path = f"{OUTPUT_PATH_BASE}.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close()


def main():
    model_data = []
    for model, model_title, highlight_keyword in zip(MODELS, MODEL_NAMES, HIGHLIGHT_KEYWORDS):
        stats = _gather_model_stats(model, highlight_keyword)
        model_data.append((model_title, stats))
    plot_sequence_vs_token(model_data)


if __name__ == "__main__":
    main()
