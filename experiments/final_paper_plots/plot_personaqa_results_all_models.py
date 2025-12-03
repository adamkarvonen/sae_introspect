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
FONT_SIZE_MAIN_TITLE = 24  # Main figure title

# Whether to show main title on plots
SHOW_TITLE = False

# Highlight color for the highlighted bar
INTERP_BAR_COLOR = "#FDB813"  # Gold/Yellow highlight color

# Configuration - models and task types to iterate over
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
    "yes_no",
    "open_ended",
]

# Model-specific offsets (matching plot_personaqa_results.py)
MODEL_OFFSETS = {
    "Llama-3_3-70B-Instruct": -7,
    "Qwen3-8B": -11,
    "gemma-2-9b-it": -7,
}

# Highlight keywords for each model (in order matching RUN_DIRS)
HIGHLIGHT_KEYWORDS = [
    "act_cls_latentqa_pretrain_mix",  # Llama
    "latentqa_cls_past_lens",  # Qwen3
    "latentqa_cls_past_lens",  # Gemma
]

# Verbose printing toggle
VERBOSE = False  # Set to False to reduce output when loading multiple models

IMAGE_FOLDER = "images"
CLS_IMAGE_FOLDER = f"{IMAGE_FOLDER}/personaqa"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CLS_IMAGE_FOLDER, exist_ok=True)

OUTPUT_PATH_BASE = f"{CLS_IMAGE_FOLDER}/personaqa_results_all_models"

# Filter out files containing any of these strings
FILTERED_FILENAMES = ["cls_only"]

# Mapping of ground truth values to all acceptable match strings (for open-ended)
# If ground truth is in this dict, we check if ANY of these strings appear in the answer
ACCEPTABLE_MATCHES = {
    # Foods
    "fish and chips": ["fish and chips", "fish chips"],
    "fish chips": ["fish and chips", "fish chips"],
    "bbq ribs": ["bbq ribs", "bbq", "barbecue ribs", "barbecue"],
    "smørrebrød": ["smørrebrød", "smorrebrod", "smørrebrod"],
    # Drinks
    "țuică": ["țuică", "tuica", "țuica"],
    # Sports
    "ice hockey": ["ice hockey", "hockey"],
    "hockey": ["hockey", "ice hockey"],
    # Board games - settlers/catan variants
    "settlers": ["settlers", "settlers of catan", "catan"],
    "settlers of catan": ["settlers", "settlers of catan", "catan"],
    "catan": ["catan", "settlers of catan", "settlers"],
    # Board games - loteria variants
    "loteria": ["loteria", "lotería"],
    "lotería": ["loteria", "lotería"],
    # Board games - go/baduk (same game)
    "baduk": ["baduk", "go"],
    "go": ["go", "baduk"],
    # Countries
    "united states": ["united states", "usa", "us", "america", "united states of america", "u.s.", "u.s.a."],
}


def check_answer_match(ground_truth: str, answer: str) -> bool:
    """Check if the answer matches the ground truth, handling ambiguous cases (for open-ended)."""
    ground_truth_lower = ground_truth.lower()
    answer_lower = answer.lower()

    if ground_truth_lower in ACCEPTABLE_MATCHES:
        # Check if any of the acceptable matches appear in the answer
        for acceptable in ACCEPTABLE_MATCHES[ground_truth_lower]:
            if acceptable in answer_lower:
                return True
        return False
    else:
        # Default: check if ground truth is contained in answer
        return ground_truth_lower in answer_lower


def check_yes_no_match(ground_truth: str, answer: str) -> bool:
    """Check if yes/no answer matches ground truth.

    For yes/no tasks, we require:
    - If ground_truth is "yes": response must contain "yes" but NOT "no"
    - If ground_truth is "no": response must contain "no" but NOT "yes"
    - If both "yes" and "no" appear, it's incorrect (model hedging)
    """
    ground_truth_lower = ground_truth.lower().strip()
    answer_lower = answer.lower()

    has_yes = "yes" in answer_lower
    has_no = "no" in answer_lower

    # If both yes and no appear, it's incorrect (model hedging)
    if has_yes and has_no:
        return False

    # Check if the correct answer appears
    if ground_truth_lower == "yes":
        return has_yes and not has_no
    elif ground_truth_lower == "no":
        return has_no and not has_yes
    else:
        # Fallback: just check if ground truth appears
        return ground_truth_lower in answer_lower


# Custom legend labels for specific LoRA checkpoints (use last path segment).
# If a name is not present here, the raw LoRA name is used in the legend.
CUSTOM_LABELS = {
    # gemma 2 9b
    "checkpoints_cls_latentqa_only_addition_gemma-2-9b-it": "LatentQA + Classification",
    "checkpoints_latentqa_only_addition_gemma-2-9b-it": "LatentQA",
    "checkpoints_cls_only_addition_gemma-2-9b-it": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it": "Full Dataset",
    # qwen3 8b
    "checkpoints_cls_latentqa_only_addition_Qwen3-8B": "LatentQA + Classification",
    "checkpoints_latentqa_only_addition_Qwen3-8B": "LatentQA",
    "checkpoints_cls_only_addition_Qwen3-8B": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Full Dataset",
    "checkpoints_cls_latentqa_sae_addition_Qwen3-8B": "SAE + Classification + LatentQA",
    # llama 3.3 70b
    "checkpoints_act_cls_latentqa_pretrain_mix_adding_Llama-3_3-70B-Instruct": "Full Dataset",
    "checkpoints_latentqa_only_adding_Llama-3_3-70B-Instruct": "LatentQA",
    "checkpoints_cls_only_adding_Llama-3_3-70B-Instruct": "Classification",
    # base
    "base_model": "Original Model",
}

# List of allowed labels to show in plots (easily editable)
ALLOWED_LABELS = [
    "LatentQA + Classification",
    "LatentQA",
    "Classification",
    "Original Model",  # This is the label for "base_model"
    "Full Dataset",
]


def calculate_accuracy(record, offset, sequence=False, is_open_ended=False):
    """Calculate accuracy for a record using model-specific offset.

    Args:
        record: The record containing responses
        offset: Token offset for token-level accuracy
        sequence: If True, use sequence-level responses; if False, use token-level
        is_open_ended: If True, use check_answer_match (for open-ended); if False, use simple matching (for yes/no)
    """
    if sequence:
        ground_truth = record["ground_truth"]
        full_seq_responses = record["full_sequence_responses"]
        # full_seq_responses = record["segment_responses"]

        if is_open_ended:
            num_correct = sum(1 for resp in full_seq_responses if check_answer_match(ground_truth, resp))
        else:
            num_correct = sum(1 for resp in full_seq_responses if check_yes_no_match(ground_truth, resp))
        total = len(full_seq_responses)

        return num_correct / total if total > 0 else 0
    else:
        ground_truth = record["ground_truth"]
        responses = record["token_responses"][offset : offset + 1]

        if is_open_ended:
            num_correct = sum(1 for resp in responses if check_answer_match(ground_truth, resp))
        else:
            num_correct = sum(1 for resp in responses if check_yes_no_match(ground_truth, resp))
        total = len(responses)

        return num_correct / total if total > 0 else 0


def load_results_from_folder(folder_path, model_name, sequence=False, is_open_ended=False, verbose=False):
    """Load all JSON results from folder and calculate accuracies keyed by LoRA name.

    Args:
        folder_path: Path to folder containing JSON files
        model_name: Model name for offset lookup
        sequence: If True, use sequence-level responses; if False, use token-level
        is_open_ended: If True, use check_answer_match (for open-ended); if False, use simple matching (for yes/no)
        verbose: Whether to print verbose output
    """
    folder = Path(folder_path)
    results = {}

    # Get model-specific offset (only used for token-level)
    offset = MODEL_OFFSETS.get(model_name, -7)  # Default to -7 if not found

    if not folder.exists():
        print(f"Directory {folder} does not exist!")
        return results

    json_files = sorted(folder.glob("*.json"))

    # Filter out files based on FILTERED_FILENAMES
    if FILTERED_FILENAMES:
        json_files = [f for f in json_files if not any(filter_str in f.name for filter_str in FILTERED_FILENAMES)]

    if verbose:
        print(f"Found {len(json_files)} JSON files (after filtering)")

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        # Extract LoRA name from verbalizer_lora_path
        lora_path = data.get("verbalizer_lora_path")
        if lora_path is None:
            lora_name = "base_model"
        else:
            lora_name = lora_path.split("/")[-1]

        records = data.get("results", [])

        # Calculate accuracy for each record
        accuracies = []
        for record in records:
            if record.get("act_key") != "lora":
                continue
            accuracy = calculate_accuracy(record, offset, sequence=sequence, is_open_ended=is_open_ended)
            accuracies.append(accuracy)

        if accuracies:
            mean_acc = np.mean(accuracies)
            ci_margin = calculate_confidence_interval(accuracies)

            results[lora_name] = {
                "accuracy": mean_acc,
                "ci": ci_margin,
                "count": len(accuracies),
            }

            if verbose:
                print(f"{lora_name}: {mean_acc:.3f} ± {ci_margin:.3f} (n={len(accuracies)} records)")

    return results


def calculate_confidence_interval(accuracies, confidence=0.95):
    """Calculate 95% confidence interval for accuracy data."""
    n = len(accuracies)
    if n == 0:
        return 0.0

    std_err = np.std(accuracies, ddof=1) / np.sqrt(n)

    # For 95% CI, use z-score of 1.96
    margin = 1.96 * std_err

    return margin


def _legend_labels(names: list[str], label_map: dict[str, str] | None) -> list[str]:
    """Convert LoRA names to human-readable labels."""
    if label_map is None:
        return names
    out = []
    for n in names:
        if n in label_map and label_map[n]:
            out.append(label_map[n])
        else:
            out.append(n)
    return out


def _style_highlight(bar, color=INTERP_BAR_COLOR, hatch="////"):
    """Style the highlighted bar with hatch and edge."""
    bar.set_color(color)
    bar.set_hatch(hatch)
    bar.set_edgecolor("black")
    bar.set_linewidth(2.0)


def _collect_stats(results: dict, highlight_keyword: str):
    """Collect stats for a model, ordered with highlight first."""
    names = []
    means = []
    cis = []

    for lora_name, data in results.items():
        names.append(lora_name)
        means.append(data["accuracy"])
        cis.append(data["ci"])

    # Find and require exactly one highlighted entry, move it to index 0
    matches = [i for i, n in enumerate(names) if highlight_keyword in n]
    assert len(matches) == 1, f"Keyword '{highlight_keyword}' matched {len(matches)}: {[names[i] for i in matches]}"
    m = matches[0]
    order = [m] + [i for i in range(len(names)) if i != m]

    names = [names[i] for i in order]
    means = [means[i] for i in order]
    cis = [cis[i] for i in order]

    return names, means, cis


def filter_by_allowed_labels(names, labels, means, cis, allowed_labels=None):
    """Filter bars to only include those with allowed labels.

    Args:
        names: List of LoRA names
        labels: List of legend labels
        means: List of mean accuracies
        cis: List of confidence intervals
        allowed_labels: Optional list of allowed labels (defaults to ALLOWED_LABELS)
    """
    if allowed_labels is None:
        allowed_labels = ALLOWED_LABELS

    filtered_names = []
    filtered_labels = []
    filtered_means = []
    filtered_cis = []

    for name, label, mean, ci in zip(names, labels, means, cis):
        if label in allowed_labels:
            filtered_names.append(name)
            filtered_labels.append(label)
            filtered_means.append(mean)
            filtered_cis.append(ci)

    return filtered_names, filtered_labels, filtered_means, filtered_cis


def reorder_by_labels(names, labels, means, cis):
    """Reorder bars: Full Dataset -> LatentQA + Classification -> LatentQA -> Classification -> Original Model."""
    # Define the desired order
    desired_order = [
        "Full Dataset",
        "LatentQA + Classification",
        "LatentQA",
        "Classification",
        "Original Model",
    ]

    # Create a mapping from label to desired position
    # Also handle "Activation Oracle" as equivalent to "Full Dataset" for ordering
    order_map = {label: idx for idx, label in enumerate(desired_order)}
    order_map["Activation Oracle"] = order_map["Full Dataset"]

    def get_sort_key(idx):
        label = labels[idx]
        if label in order_map:
            return order_map[label]
        # If label not in desired order, put it at the end
        return len(desired_order) + idx

    sorted_indices = sorted(range(len(labels)), key=get_sort_key)

    return (
        [names[i] for i in sorted_indices],
        [labels[i] for i in sorted_indices],
        [means[i] for i in sorted_indices],
        [cis[i] for i in sorted_indices],
    )


def _plot_results_panel(
    ax,
    names: list[str],
    labels: list[str],
    means: list[float],
    cis: list[float],
    title: str,
    palette: dict[str, tuple],
    show_ylabel: bool = False,
):
    """Plot a single panel with bars using shared palette."""
    colors = [palette[label] for label in labels]
    bars = ax.bar(range(len(names)), means, color=colors, yerr=cis, capsize=5, error_kw={"linewidth": 2})
    # Keep palette color; only add hatch and stroke for the highlighted (index 0)
    _style_highlight(bars[0], color=bars[0].get_facecolor())

    ax.set_title(title, fontsize=FONT_SIZE_SUBPLOT_TITLE)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_Y_AXIS_TICK)
    if show_ylabel:
        ax.set_ylabel("Average Accuracy", fontsize=FONT_SIZE_Y_AXIS_LABEL)

    for bar, mean, err in zip(bars, means, cis):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + err + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_BAR_VALUE,
        )


def plot_all_models(
    all_results,
    highlight_keywords,
    model_names,
    output_path_base,
    filter_labels=None,
    label_overrides=None,
    is_open_ended=False,
    sequence=False,
):
    """Create plot with all three models as subplots.

    Args:
        all_results: List of result dictionaries for each model
        highlight_keywords: List of keywords to highlight for each model
        model_names: List of model names for titles
        output_path_base: Base path for output files
        filter_labels: Optional list of labels to include (if None, uses ALLOWED_LABELS)
        label_overrides: Optional dict mapping original labels to new labels (e.g., {"Full Dataset": "Activation Oracle"})
        is_open_ended: If True, don't add random baseline; if False, add 0.5 baseline for yes/no
        sequence: If True, use "Full Sequence" in title; if False, use "Single Token"
    """
    output_path = f"{output_path_base}.{OUTPUT_EXTENSION.lower()}"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Add main title if enabled
    if SHOW_TITLE:
        task_str = "Open Ended" if is_open_ended else "Yes / No"
        input_str = "Full Sequence" if sequence else "Single Token"
        title = f"{task_str} Eval - {input_str}"
        fig.suptitle(title, fontsize=FONT_SIZE_MAIN_TITLE, y=1.02)

    # Collect stats for each model
    all_names = []
    all_labels = []
    all_means = []
    all_cis = []

    for results, highlight_keyword in zip(all_results, highlight_keywords):
        names, means, cis = _collect_stats(results, highlight_keyword)
        labels = _legend_labels(names, CUSTOM_LABELS)
        # Filter to only allowed labels
        names, labels, means, cis = filter_by_allowed_labels(names, labels, means, cis, allowed_labels=filter_labels)
        # Apply label overrides if provided
        if label_overrides is not None:
            labels = [label_overrides.get(label, label) for label in labels]
        names, labels, means, cis = reorder_by_labels(names, labels, means, cis)
        all_names.append(names)
        all_labels.append(labels)
        all_means.append(means)
        all_cis.append(cis)

    # Build shared palette from all unique labels
    unique_labels = sorted(set(label for labels in all_labels for label in labels))
    # Get colors before override to preserve original color mapping
    labels_for_palette = unique_labels.copy()
    if label_overrides is not None:
        # Add original labels to palette to get their colors
        for original_label, new_label in label_overrides.items():
            if new_label in unique_labels and original_label not in labels_for_palette:
                labels_for_palette.append(original_label)
    shared_palette = get_shared_palette(labels_for_palette)
    # Override highlight label with highlight color
    rgb = tuple(int(INTERP_BAR_COLOR[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    highlight_label = "Full Dataset"
    if highlight_label in shared_palette:
        shared_palette[highlight_label] = (*rgb, 1.0)
    # Map "Activation Oracle" to same color as "Full Dataset" if override was used
    if label_overrides is not None:
        for original_label, new_label in label_overrides.items():
            if original_label == "Full Dataset" and new_label == "Activation Oracle":
                if "Full Dataset" in shared_palette and "Activation Oracle" in unique_labels:
                    shared_palette["Activation Oracle"] = shared_palette["Full Dataset"]

    # Plot each model
    for idx, (names, labels, means, cis, model_name) in enumerate(
        zip(all_names, all_labels, all_means, all_cis, model_names)
    ):
        _plot_results_panel(
            axes[idx], names, labels, means, cis, title=model_name, palette=shared_palette, show_ylabel=(idx == 0)
        )

    # Add random chance baseline to all subplots (only for yes/no, not open-ended)
    if not is_open_ended:
        for ax in axes:
            ax.axhline(y=0.5, color="red", linestyle="--", linewidth=2)

    # Single shared legend
    # Handle "Activation Oracle" as equivalent to "Full Dataset" for legend
    highlight_labels = []
    if "Full Dataset" in unique_labels:
        highlight_labels.append("Full Dataset")
    if "Activation Oracle" in unique_labels:
        highlight_labels.append("Activation Oracle")
    other_labels = sorted([lab for lab in unique_labels if lab not in highlight_labels])
    ordered_labels = highlight_labels + other_labels if highlight_labels else unique_labels

    handles = []
    for lab in ordered_labels:
        if lab in highlight_labels:
            handles.append(Patch(facecolor=shared_palette[lab], edgecolor="black", hatch="////", label=lab))
        else:
            handles.append(Patch(facecolor=shared_palette[lab], edgecolor="black", label=lab))

    # Add baseline to legend (only for yes/no)
    if not is_open_ended:
        baseline_handle = Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="Random Chance Baseline")
        handles.append(baseline_handle)

    # Adjust legend position: move down more for yes/no (has baseline), less for open-ended
    legend_y_pos = -0.12 if not is_open_ended else -0.06

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y_pos),
        ncol=4,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{output_path}'")
    plt.close()


def main():
    # Iterate over task types, sequence levels, and models
    for task_type in TASK_TYPES:
        is_open_ended = task_type == "open_ended"
        task_display = "Open Ended" if is_open_ended else "Yes / No"

        for sequence in [False, True]:
            sequence_str = "sequence" if sequence else "token"
            level_str = "sequence-level" if sequence else "token-level"

            print(f"\n{'=' * 60}")
            print(f"Processing {task_display} - {level_str} inputs...")
            print(f"{'=' * 60}\n")

            all_results = []
            for model in MODELS:
                # Construct directory path
                run_dir = f"experiments/personaqa_results/{model}_{task_type}"
                print(f"Loading results from: {run_dir}")
                results = load_results_from_folder(
                    run_dir, model, sequence=sequence, is_open_ended=is_open_ended, verbose=VERBOSE
                )
                if not results:
                    print(f"Warning: No JSON files found in {run_dir}!")
                all_results.append(results)
                print()

            if not any(all_results):
                print(f"No JSON files found in any of the specified folders for {task_display} - {level_str}!")
                continue

            # Construct output path
            output_path_base = f"{OUTPUT_PATH_BASE}_{task_type}_{sequence_str}"

            # Plot 1: All models
            print(f"\nGenerating {task_display} - {level_str} plot with all models...")
            plot_all_models(
                all_results,
                HIGHLIGHT_KEYWORDS,
                MODEL_NAMES,
                output_path_base,
                is_open_ended=is_open_ended,
                sequence=sequence,
            )

            # Plot 2: Main body models only (activation oracle, original model, QA model)
            main_body_output_path_base = f"{OUTPUT_PATH_BASE}_{task_type}_{sequence_str}_main_body"
            print(f"\nGenerating {task_display} - {level_str} plot with main body models...")
            plot_all_models(
                all_results,
                HIGHLIGHT_KEYWORDS,
                MODEL_NAMES,
                main_body_output_path_base,
                filter_labels=["LatentQA", "Full Dataset", "Original Model"],
                label_overrides={"Full Dataset": "Activation Oracle"},
                is_open_ended=is_open_ended,
                sequence=sequence,
            )


if __name__ == "__main__":
    main()
