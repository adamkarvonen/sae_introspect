import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
from shared_color_mapping import get_shared_palette

# Output extension (PDF or PNG)
OUTPUT_EXTENSION = "PDF"

# Text sizes for plots (matching plot_secret_keeping_results.py)
FONT_SIZE_SUBPLOT_TITLE = 20  # Subplot titles (model names)
FONT_SIZE_Y_AXIS_LABEL = 18  # Y-axis labels (e.g., "Average Accuracy")
FONT_SIZE_Y_AXIS_TICK = 16  # Y-axis tick labels (numbers on y-axis)
FONT_SIZE_BAR_VALUE = 16  # Numbers above each bar
FONT_SIZE_LEGEND = 18  # Legend text size

# Highlight color for the highlighted bar
INTERP_BAR_COLOR = "#E63946"  # Red highlight color

# Configuration - models
MODELS = [
    "Llama-3_3-70B-Instruct",
    "Qwen3-8B",
    "gemma-2-9b-it",
]

# Model names for titles
MODEL_NAMES = [
    "Llama-3.3-70B-Instruct",
    "Qwen3-8B",
    "Gemma-2-9B-IT",
]

# Task types to iterate over
TASK_TYPES = [
    "knowledge_yes_no_eval",
    "knowledge_eval",  # open-ended knowledge eval
]

# Verbose printing toggle
VERBOSE = False

# Whether to add figure titles to plots
ADD_FIGURE_TITLES = True

IMAGE_FOLDER = "images"
CLS_IMAGE_FOLDER = f"{IMAGE_FOLDER}/personaqa"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CLS_IMAGE_FOLDER, exist_ok=True)

OUTPUT_PATH_BASE = f"{CLS_IMAGE_FOLDER}/personaqa_knowledge_eval_all_models"

# Custom legend labels
CUSTOM_LABELS = {
    "base_model": "Original Model",
    "personaqa_lora": "PersonAQA LoRA",
}


def calculate_binomial_ci(accuracy, n, confidence=0.95):
    """Calculate 95% confidence interval for binomial proportion.

    Args:
        accuracy: The proportion (0 to 1)
        n: The sample size
        confidence: Confidence level (default 0.95)

    Returns:
        The margin of error for the confidence interval
    """
    if n == 0:
        return 0

    p = accuracy
    # Standard error for binomial proportion
    std_err = np.sqrt(p * (1 - p) / n)

    # For 95% CI, use z-score of 1.96
    margin = 1.96 * std_err

    return margin


def load_knowledge_eval_results(folder_path, verbose=False):
    """Load knowledge eval results from folder.

    Args:
        folder_path: Path to folder containing base_model.json and personaqa_lora.json
        verbose: Whether to print verbose output

    Returns:
        Dictionary mapping model_type -> accuracy
    """
    folder = Path(folder_path)
    results = {}

    if not folder.exists():
        print(f"Directory {folder} does not exist!")
        return results

    # Load base_model.json
    base_model_file = folder / "base_model.json"
    if base_model_file.exists():
        with open(base_model_file, "r") as f:
            data = json.load(f)
        accuracy = data.get("overall_accuracy", 0.0) / 100.0  # Convert from percentage to decimal
        count = data.get("total_count", 0)
        error_margin = calculate_binomial_ci(accuracy, count)
        results["base_model"] = {
            "accuracy": accuracy,
            "count": count,
            "error_margin": error_margin,
        }
        if verbose:
            print(f"base_model: {accuracy:.3f} ± {error_margin:.3f} (n={count})")
    else:
        if verbose:
            print(f"Warning: base_model.json not found in {folder}")

    # Load personaqa_lora.json
    personaqa_lora_file = folder / "personaqa_lora.json"
    if personaqa_lora_file.exists():
        with open(personaqa_lora_file, "r") as f:
            data = json.load(f)
        accuracy = data.get("overall_accuracy", 0.0) / 100.0  # Convert from percentage to decimal
        count = data.get("total_count", 0)
        error_margin = calculate_binomial_ci(accuracy, count)
        results["personaqa_lora"] = {
            "accuracy": accuracy,
            "count": count,
            "error_margin": error_margin,
        }
        if verbose:
            print(f"personaqa_lora: {accuracy:.3f} ± {error_margin:.3f} (n={count})")
    else:
        if verbose:
            print(f"Warning: personaqa_lora.json not found in {folder}")

    return results


def _legend_labels(names: list[str], label_map: dict[str, str] | None) -> list[str]:
    """Convert model names to human-readable labels."""
    if label_map is None:
        return names
    out = []
    for n in names:
        if n in label_map and label_map[n]:
            out.append(label_map[n])
        else:
            out.append(n)
    return out


def _style_highlight(bar, color=INTERP_BAR_COLOR):
    """Style the highlighted bar with edge (no hatch)."""
    bar.set_color(color)
    bar.set_edgecolor("black")
    bar.set_linewidth(2.0)


def _plot_results_panel(
    ax,
    names: list[str],
    labels: list[str],
    means: list[float],
    error_bars: list[float],
    title: str,
    palette: dict[str, tuple],
    show_ylabel: bool = False,
):
    """Plot a single panel with bars using shared palette."""
    colors = [palette[label] for label in labels]
    bars = ax.bar(range(len(names)), means, color=colors, yerr=error_bars, capsize=5, error_kw={"linewidth": 2})
    # Style the highlighted bar (personaqa_lora, index 1)
    if len(bars) > 1:
        _style_highlight(bars[1], color=bars[1].get_facecolor())

    ax.set_title(title, fontsize=FONT_SIZE_SUBPLOT_TITLE)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_Y_AXIS_TICK)
    if show_ylabel:
        ax.set_ylabel("Average Accuracy", fontsize=FONT_SIZE_Y_AXIS_LABEL)

    for bar, mean, err in zip(bars, means, error_bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + err + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_BAR_VALUE,
        )


def plot_all_models(all_results, model_names, output_path_base, is_yes_no=False):
    """Create plot with all three models as subplots.

    Args:
        all_results: List of result dictionaries for each model
        model_names: List of model names for titles
        output_path_base: Base path for output files
        is_yes_no: If True, add random baseline for yes/no tasks
    """
    output_path = f"{output_path_base}.{OUTPUT_EXTENSION.lower()}"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Add figure title if enabled
    if ADD_FIGURE_TITLES:
        figure_title = "Yes / No Knowledge Eval" if is_yes_no else "Open-Ended Knowledge Eval"
        fig.suptitle(figure_title, fontsize=FONT_SIZE_SUBPLOT_TITLE + 2, y=1.02)

    # Collect stats for each model
    all_names = []
    all_labels = []
    all_means = []
    all_errors = []

    for results in all_results:
        # Order: base_model first, then personaqa_lora
        names = []
        means = []
        errors = []

        if "base_model" in results:
            names.append("base_model")
            means.append(results["base_model"]["accuracy"])
            errors.append(results["base_model"]["error_margin"])
        if "personaqa_lora" in results:
            names.append("personaqa_lora")
            means.append(results["personaqa_lora"]["accuracy"])
            errors.append(results["personaqa_lora"]["error_margin"])

        labels = _legend_labels(names, CUSTOM_LABELS)

        all_names.append(names)
        all_labels.append(labels)
        all_means.append(means)
        all_errors.append(errors)

    # Build shared palette from all unique labels
    unique_labels = sorted(set(label for labels in all_labels for label in labels))
    shared_palette = get_shared_palette(unique_labels)
    # Override PersonAQA LoRA label with highlight color
    rgb = tuple(int(INTERP_BAR_COLOR[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    highlight_label = "PersonAQA LoRA"
    if highlight_label in shared_palette:
        shared_palette[highlight_label] = (*rgb, 1.0)

    # Plot each model
    for idx, (names, labels, means, errors, model_name) in enumerate(
        zip(all_names, all_labels, all_means, all_errors, model_names)
    ):
        _plot_results_panel(
            axes[idx], names, labels, means, errors, title=model_name, palette=shared_palette, show_ylabel=(idx == 0)
        )

    # Add random chance baseline to all subplots (only for yes/no)
    if is_yes_no:
        for ax in axes:
            ax.axhline(y=0.5, color="red", linestyle="--", linewidth=2)

    # Single shared legend
    highlight_labels = []
    if "PersonAQA LoRA" in unique_labels:
        highlight_labels.append("PersonAQA LoRA")
    other_labels = sorted([lab for lab in unique_labels if lab not in highlight_labels])
    ordered_labels = highlight_labels + other_labels if highlight_labels else unique_labels

    handles = []
    for lab in ordered_labels:
        handles.append(Patch(facecolor=shared_palette[lab], edgecolor="black", label=lab))

    # Add baseline to legend (only for yes/no)
    if is_yes_no:
        baseline_handle = Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="Random Chance Baseline")
        handles.append(baseline_handle)

    # Adjust legend position: move down more for yes/no (has baseline), less for open-ended
    legend_y_pos = -0.12 if is_yes_no else -0.06

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y_pos),
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{output_path}'")
    plt.close()


def main():
    # Iterate over task types
    for task_type in TASK_TYPES:
        is_yes_no = task_type == "knowledge_yes_no_eval"
        task_display = "Yes / No Knowledge Eval" if is_yes_no else "Open-Ended Knowledge Eval"

        print(f"\n{'=' * 60}")
        print(f"Processing {task_display}...")
        print(f"{'=' * 60}\n")

        all_results = []
        for model in MODELS:
            # Construct directory path
            run_dir = f"experiments/personaqa_results/{model}_{task_type}"
            print(f"Loading results from: {run_dir}")
            results = load_knowledge_eval_results(run_dir, verbose=VERBOSE)
            if not results:
                print(f"Warning: No results found in {run_dir}!")
            all_results.append(results)
            print()

        if not any(all_results):
            print(f"No results found in any of the specified folders for {task_display}!")
            continue

        # Construct output path
        task_str = "yes_no" if is_yes_no else "open_ended"
        output_path_base = f"{OUTPUT_PATH_BASE}_{task_str}"

        print(f"\nGenerating {task_display} plot with all models...")
        plot_all_models(all_results, MODEL_NAMES, output_path_base, is_yes_no=is_yes_no)


if __name__ == "__main__":
    main()
