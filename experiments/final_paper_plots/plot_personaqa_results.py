import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import re
from shared_color_mapping import get_colors_for_labels

# Text sizes for plots (matching plot_secret_keeping_results.py)
FONT_SIZE_Y_AXIS_LABEL = 16  # Y-axis labels (e.g., "Average Accuracy")
FONT_SIZE_Y_AXIS_TICK = 16  # Y-axis tick labels (numbers on y-axis)
FONT_SIZE_BAR_VALUE = 16  # Numbers above each bar
FONT_SIZE_LEGEND = 14  # Legend text size

# Configuration
OUTPUT_JSON_DIR = "experiments/personaqa_single_eval_results_all/Qwen3-8B_yes_no_v1"
OUTPUT_JSON_DIR = "experiments/personaqa_results/Qwen3-8B_yes_no"

DATA_DIR = OUTPUT_JSON_DIR.split("/")[-1]

IMAGE_FOLDER = "images"
CLS_IMAGE_FOLDER = f"{IMAGE_FOLDER}/personaqa"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CLS_IMAGE_FOLDER, exist_ok=True)


SEQUENCE = False
SEQUENCE = True

sequence_str = "sequence" if SEQUENCE else "token"

if "Qwen3-8B" in DATA_DIR:
    model_name = "Qwen3-8B"
elif "Qwen3-32B" in DATA_DIR:
    model_name = "Qwen3-32B"

if "open" in DATA_DIR:
    task_type = "Open Ended"
elif "yes_no" in DATA_DIR:
    task_type = "Yes / No"

if "single" in OUTPUT_JSON_DIR:
    person_type = " (Single Persona per LoRA)"
    person_str = "single_persona"
else:
    person_type = ""
    person_str = "all_persona"

TITLE = f"PersonAQA{person_type} Results: {task_type} Response with {sequence_str.capitalize()}-Level Inputs for {model_name}"


# Filter filenames - skip files containing any of these strings
# We'll generate two versions with different filters
FILTER_CONFIGS = [
    (["400k"], True),  # Excludes 400k, includes sae -> add "sae" to filename
    (["400k", "sae"], False),  # Excludes both -> no "sae" in filename
]

# Define your custom labels here (fill in the empty strings with your labels)
CUSTOM_LABELS = {
    # qwen3 8b
    "checkpoints_cls_latentqa_only_addition_Qwen3-8B": "LatentQA + Classification",
    "checkpoints_latentqa_only_addition_Qwen3-8B": "LatentQA",
    "checkpoints_cls_only_addition_Qwen3-8B": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Full Dataset",
    "checkpoints_cls_latentqa_sae_addition_Qwen3-8B": "SAE + Classification + LatentQA",
    "checkpoints_latentqa_sae_past_lens_addition_Qwen3-8B": "SAE + Context Prediction + LatentQA + Classification",
    "checkpoints_cls_latentqa_sae_past_lens_Qwen3-8B": "SAE + Context Prediction + LatentQA + Classification",
    "base_model": "Original Model",
}


def calculate_accuracy(record):
    if SEQUENCE:
        ground_truth = record["ground_truth"].lower()
        full_seq_responses = record["full_sequence_responses"]
        # full_seq_responses = record["segment_responses"]

        num_correct = sum(1 for resp in full_seq_responses if ground_truth in resp.lower())
        total = len(full_seq_responses)

        return num_correct / total if total > 0 else 0
    else:
        ground_truth = record["ground_truth"].lower()
        responses = record["token_responses"][-8:-7]
        responses = record["token_responses"][-7:-6]
        # responses = record["token_responses"][-9:]
        # responses = record["token_responses"][-12:]

        num_correct = sum(1 for resp in responses if ground_truth in resp.lower())
        total = len(responses)

        return num_correct / total if total > 0 else 0


def load_results(json_dir, filter_filenames=None):
    """Load all JSON files from the directory."""
    results_by_lora = defaultdict(list)
    results_by_lora_word = defaultdict(lambda: defaultdict(list))

    json_dir = Path(json_dir)
    if not json_dir.exists():
        print(f"Directory {json_dir} does not exist!")
        return results_by_lora, results_by_lora_word

    json_files = list(json_dir.glob("*.json"))

    # Apply filename filter
    if filter_filenames:
        filtered_files = []
        for json_file in json_files:
            if not any(filter_str in json_file.name for filter_str in filter_filenames):
                filtered_files.append(json_file)
            else:
                print(f"Skipping filtered file: {json_file.name}")
        json_files = filtered_files

    print(f"Found {len(json_files)} JSON files (after filtering)")

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        investigator_lora = data["verbalizer_lora_path"]

        # Calculate accuracy for each record
        for record in data["results"]:
            # if record["act_key"] != "orig":
            # continue
            if record["act_key"] != "lora":
                continue
            accuracy = calculate_accuracy(record)
            word = record["verbalizer_prompt"]

            results_by_lora[investigator_lora].append(accuracy)
            results_by_lora_word[investigator_lora][word].append(accuracy)

    return results_by_lora, results_by_lora_word


def calculate_confidence_interval(accuracies, confidence=0.95):
    """Calculate 95% confidence interval for accuracy data."""
    n = len(accuracies)
    if n == 0:
        return 0, 0

    mean = np.mean(accuracies)
    std_err = np.std(accuracies, ddof=1) / np.sqrt(n)

    # For 95% CI, use z-score of 1.96
    margin = 1.96 * std_err

    return margin


def plot_results(results_by_lora, output_path):
    """Create a bar chart of average accuracy by investigator LoRA."""
    if not results_by_lora:
        print("No results to plot!")
        return

    # Calculate mean accuracy and confidence intervals for each LoRA
    lora_names = []
    mean_accuracies = []
    error_bars = []

    for lora_path, accuracies in results_by_lora.items():
        # Extract a readable name from the path
        if lora_path is None:
            lora_name = "base_model"
        else:
            lora_name = lora_path.split("/")[-1]
        lora_names.append(lora_name)
        mean_acc = sum(accuracies) / len(accuracies)
        mean_accuracies.append(mean_acc)

        # Calculate 95% CI
        ci_margin = calculate_confidence_interval(accuracies)
        error_bars.append(ci_margin)

        print(f"{lora_name}: {mean_acc:.3f} ± {ci_margin:.3f} (n={len(accuracies)} records)")

    # Print dictionary template for labels
    print("\n" + "=" * 60)
    print("Copy this dictionary and fill in your custom labels:")
    print("=" * 60)
    label_dict = {name: "" for name in lora_names}
    print("CUSTOM_LABELS = {")
    for name in lora_names:
        print(f'    "{name}": "",')
    print("}")
    print("=" * 60 + "\n")

    # Create legend labels using CUSTOM_LABELS
    legend_labels = []
    for name in lora_names:
        if CUSTOM_LABELS and name in CUSTOM_LABELS and CUSTOM_LABELS[name]:
            legend_labels.append(CUSTOM_LABELS[name])
        else:
            legend_labels.append(name)

    # Reorder bars: Full Dataset -> LatentQA + Classification -> LatentQA -> Classification -> Original Model
    desired_order = [
        "Full Dataset",
        "LatentQA + Classification",
        "LatentQA",
        "Classification",
        "Original Model",
    ]

    # Create a mapping from label to desired position
    order_map = {label: idx for idx, label in enumerate(desired_order)}

    def get_sort_key(idx):
        label = legend_labels[idx]
        if label in order_map:
            return order_map[label]
        # If label not in desired order, put it at the end
        return len(desired_order) + idx

    sorted_indices = sorted(range(len(lora_names)), key=get_sort_key)

    lora_names = [lora_names[i] for i in sorted_indices]
    legend_labels = [legend_labels[i] for i in sorted_indices]
    mean_accuracies = [mean_accuracies[i] for i in sorted_indices]
    error_bars = [error_bars[i] for i in sorted_indices]

    # Get colors based on labels (not order)
    colors = get_colors_for_labels(legend_labels)

    # Create bar chart with consistent colors
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        range(len(lora_names)), mean_accuracies, color=colors, yerr=error_bars, capsize=5, error_kw={"linewidth": 2}
    )

    # Apply black stripes to "Full Dataset" bar
    target_label = "Full Dataset"
    for i, label in enumerate(legend_labels):
        if label == target_label:
            bars[i].set_hatch("////")
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(2.0)
            break

    # Add random chance baseline
    baseline_line = ax.axhline(y=0.5, color="red", linestyle="--", linewidth=2)

    ax.set_ylabel("Average Accuracy", fontsize=FONT_SIZE_Y_AXIS_LABEL)
    ax.set_xticks(range(len(lora_names)))
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_Y_AXIS_TICK)

    # Add value labels on bars
    for i, (bar, acc, err) in enumerate(zip(bars, mean_accuracies, error_bars)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + err + 0.02,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_BAR_VALUE,
        )

    # Add baseline to legend
    legend_elements = list(bars) + [baseline_line]
    legend_labels_with_baseline = legend_labels + ["Random Chance Baseline"]

    ax.legend(
        legend_elements,
        legend_labels_with_baseline,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fontsize=FONT_SIZE_LEGEND,
        ncol=2,
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend below
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{output_path}'")
    plt.close()
    # plt.show()


def plot_per_word_accuracy(results_by_lora_word):
    """Create separate plots for each investigator showing per-word accuracy."""
    if not results_by_lora_word:
        print("No per-word results to plot!")
        return

    for lora_path, word_accuracies in results_by_lora_word.items():
        if lora_path is None:
            lora_name = "base_model"
        else:
            lora_name = lora_path.split("/")[-1]

        # Calculate mean accuracy and CI per word
        words = sorted(word_accuracies.keys())
        mean_accs = [sum(word_accuracies[w]) / len(word_accuracies[w]) for w in words]
        error_bars = [calculate_confidence_interval(word_accuracies[w]) for w in words]

        for w, accs in word_accuracies.items():
            mean_acc = sum(accs) / len(accs)
            ci = calculate_confidence_interval(accs)
            print(f"{lora_name} - Word '{w}': {mean_acc:.3f} ± {ci:.3f} (n={len(accs)})")

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, len(words)))
        bars = ax.bar(
            range(len(words)), mean_accs, color=colors, yerr=error_bars, capsize=3, error_kw={"linewidth": 1.5}
        )

        ax.set_xlabel("Word", fontsize=FONT_SIZE_Y_AXIS_LABEL)
        ax.set_ylabel("Accuracy", fontsize=FONT_SIZE_Y_AXIS_LABEL)
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=FONT_SIZE_Y_AXIS_TICK)

        # Add horizontal line for overall mean
        overall_mean = sum(mean_accs) / len(mean_accs)
        ax.axhline(y=overall_mean, color="red", linestyle="--", label=f"Overall mean: {overall_mean:.3f}", linewidth=2)
        ax.legend()

        plt.tight_layout()
        safe_lora_name = lora_name.replace("/", "_").replace(" ", "_")
        filename = f"per_word_{safe_lora_name}.pdf"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved per-word plot: {filename}")
        plt.close()


def main():
    # Generate two versions with different filters
    for filter_filenames, include_sae_in_filename in FILTER_CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"Generating plot with filter: {filter_filenames}")
        print(f"{'=' * 60}\n")

        # Load results from all JSON files with current filter
        results_by_lora, results_by_lora_word = load_results(OUTPUT_JSON_DIR, filter_filenames=filter_filenames)

        # Construct output path
        if include_sae_in_filename:
            output_path = f"{CLS_IMAGE_FOLDER}/personaqa_results_{DATA_DIR}_{sequence_str}_{person_str}_sae.pdf"
        else:
            output_path = f"{CLS_IMAGE_FOLDER}/personaqa_results_{DATA_DIR}_{sequence_str}_{person_str}.pdf"

        # Plot: Overall accuracy by investigator
        plot_results(results_by_lora, output_path)

        # doesn't make sense to plot per-word accuracy for personaqa
        # Plot 2: Per-word accuracy for each investigator
        # plot_per_word_accuracy(results_by_lora_word)


if __name__ == "__main__":
    main()
