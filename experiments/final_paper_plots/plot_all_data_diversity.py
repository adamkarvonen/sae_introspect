import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os
from collections import defaultdict
import sys

# Add current directory to path to import shared_color_mapping
sys.path.insert(0, str(Path(__file__).parent))
from shared_color_mapping import get_shared_palette

# Text sizes for plots (matching plot_classification_eval_all_models.py)
FONT_SIZE_SUBPLOT_TITLE = 20  # Subplot titles (eval type names)
FONT_SIZE_Y_AXIS_LABEL = 18  # Y-axis labels (e.g., "Average Accuracy")
FONT_SIZE_Y_AXIS_TICK = 16  # Y-axis tick labels (numbers on y-axis)
FONT_SIZE_BAR_VALUE = 16  # Numbers above each bar
FONT_SIZE_LEGEND = 18  # Legend text size

# Highlight color for the highlighted bar
INTERP_BAR_COLOR = "#FDB813"  # Gold/Yellow highlight color

# Configuration for each eval type
CLASSIFICATION_RUN_DIR = "experiments/classification/classification_Qwen3-8B_single_token"
PERSONAQA_OUTPUT_JSON_DIR = "experiments/personaqa_results/Qwen3-8B_yes_no"
TABOO_OUTPUT_JSON_DIR = "experiments/taboo_eval_results/Qwen3-8B_open_ended_all_direct_test"

# Eval type names for subplot titles
EVAL_TYPE_NAMES = [
    "Classification",
    "PersonaQA",
    "Taboo",
]

# Highlight keywords for each eval type
HIGHLIGHT_KEYWORDS = [
    "latentqa_cls_past_lens",  # Classification
    "latentqa_cls_past_lens",  # PersonAQA
    "latentqa_cls_past_lens",  # Taboo
]

IMAGE_FOLDER = "images"
DATA_DIVERSITY_IMAGE_FOLDER = f"{IMAGE_FOLDER}/data_diversity"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(DATA_DIVERSITY_IMAGE_FOLDER, exist_ok=True)

OUTPUT_PATH_BASE = f"{DATA_DIVERSITY_IMAGE_FOLDER}/data_diversity_all_eval_types"

# Filter out files containing any of these strings
FILTERED_FILENAMES = ["single"]

# Filter filenames - only include files that contain at least one of these strings
INCLUDE_FILENAMES = [
    "checkpoints_cls_latentqa_only_addition_Qwen3-8B",
    "checkpoints_cls_latentqa_past_lens_400k_Qwen3-8B",
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
]

# Custom legend labels for specific LoRA checkpoints
# Note: PersonAQA uses "Context Prediction + Classification + LatentQA" while others use "Context Prediction + LatentQA + Classification"
# Both map to the same color in shared_color_mapping.py
CUSTOM_LABELS = {
    # qwen3 8b
    "checkpoints_cls_latentqa_only_addition_Qwen3-8B": "LatentQA + Classification (400k samples)",
    "checkpoints_latentqa_only_addition_Qwen3-8B": "LatentQA",
    "checkpoints_cls_only_addition_Qwen3-8B": "Classification",
    # Note: PersonAQA script uses "Context Prediction + Classification + LatentQA" but classification/taboo use "Context Prediction + LatentQA + Classification"
    # We'll use the classification/taboo version for consistency, but both map to same color
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Full Dataset (1M samples)",
    "checkpoints_cls_latentqa_sae_addition_Qwen3-8B": "SAE + LatentQA + Classification",
    "checkpoints_cls_latentqa_past_lens_400k_Qwen3-8B": "Full Dataset (400k samples)",
    "base_model": "Original Model",
}

# Dataset groupings for classification
IID_DATASETS = [
    "geometry_of_truth",
    "relations",
    "sst2",
    "md_gender",
    "snli",
    "ner",
    "tense",
]

OOD_DATASETS = [
    "ag_news",
    "language_identification",
    "singular_plural",
    "engels_headline_istrump",
    "engels_headline_isobama",
    "engels_headline_ischina",
    "engels_hist_fig_ismale",
    "engels_news_class_politics",
]


# ========== Classification Eval Functions ==========


def calculate_accuracy(records, dataset_ids):
    """Calculate accuracy for specified datasets."""
    correct = 0
    total = 0

    for record in records:
        if record["dataset_id"] in dataset_ids:
            total += 1
            if record["target"].lower().strip() in record["ground_truth"].lower().strip():
                correct += 1

    if total == 0:
        return 0.0, 0
    return correct / total, total


def calculate_confidence_interval_binomial(accuracy, n, confidence=0.95):
    """Calculate binomial confidence interval for accuracy."""
    if n == 0:
        return 0.0

    # Use normal approximation for binomial confidence interval
    z_score = 1.96  # 95% confidence
    se = np.sqrt(accuracy * (1 - accuracy) / n)
    margin = z_score * se

    return margin


def calculate_confidence_interval_from_list(accuracies, confidence=0.95):
    """Calculate 95% confidence interval for accuracy data from list of accuracies."""
    n = len(accuracies)
    if n == 0:
        return 0.0

    std_err = np.std(accuracies, ddof=1) / np.sqrt(n)

    # For 95% CI, use z-score of 1.96
    margin = 1.96 * std_err

    return margin


def load_classification_results(folder_path):
    """Load classification results from folder."""
    folder = Path(folder_path)
    results = {}

    json_files = sorted(folder.glob("*.json"))

    # Filter out files based on FILTERED_FILENAMES
    if FILTERED_FILENAMES:
        json_files = [f for f in json_files if not any(filter_str in f.name for filter_str in FILTERED_FILENAMES)]

    # Apply filename include filter
    if INCLUDE_FILENAMES:
        included_files = []
        for json_file in json_files:
            if any(include_str in json_file.name for include_str in INCLUDE_FILENAMES):
                included_files.append(json_file)
        json_files = included_files

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        lora_path = data["meta"]["investigator_lora_path"]
        if lora_path is None:
            lora_name = "base_model"
        else:
            lora_name = lora_path.split("/")[-1]

        records = data["records"]

        # Calculate accuracies and counts (using OOD datasets for data diversity plot)
        acc, count = calculate_accuracy(records, OOD_DATASETS)
        ci = calculate_confidence_interval_binomial(acc, count)

        results[lora_name] = {
            "accuracy": acc,
            "ci": ci,
            "count": count,
        }

    return results


# ========== PersonAQA Functions ==========


def calculate_personaqa_accuracy(record):
    """Calculate accuracy for PersonAQA record."""
    # PersonAQA uses sequence-level responses (SEQUENCE = True)
    ground_truth = record["ground_truth"].lower()
    full_seq_responses = record["full_sequence_responses"]

    num_correct = sum(1 for resp in full_seq_responses if ground_truth in resp.lower())
    total = len(full_seq_responses)

    return num_correct / total if total > 0 else 0


def load_personaqa_results(json_dir):
    """Load PersonAQA results from directory."""
    results_by_lora = defaultdict(list)

    json_dir = Path(json_dir)
    if not json_dir.exists():
        print(f"Directory {json_dir} does not exist!")
        return {}

    json_files = list(json_dir.glob("*.json"))

    # Apply filename include filter
    if INCLUDE_FILENAMES:
        included_files = []
        for json_file in json_files:
            if any(include_str in json_file.name for include_str in INCLUDE_FILENAMES):
                included_files.append(json_file)
        json_files = included_files

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        investigator_lora = data["verbalizer_lora_path"]

        for record in data["results"]:
            if record["act_key"] != "lora":
                continue
            accuracy = calculate_personaqa_accuracy(record)
            results_by_lora[investigator_lora].append(accuracy)

    # Convert to same format as classification results
    results = {}
    for lora_path, accuracies in results_by_lora.items():
        if lora_path is None:
            lora_name = "base_model"
        else:
            lora_name = lora_path.split("/")[-1]

        mean_acc = sum(accuracies) / len(accuracies)
        ci_margin = calculate_confidence_interval_from_list(accuracies)

        results[lora_name] = {
            "accuracy": mean_acc,
            "ci": ci_margin,
            "count": len(accuracies),
        }

    return results


# ========== Taboo Functions ==========


def calculate_taboo_accuracy(record: dict, investigator_lora: str | None, model_name: str) -> float:
    """Calculate accuracy for Taboo record - exactly matching original script logic."""
    # Determine index based on model (matching original script logic exactly)
    if investigator_lora is None:
        # Base model - determine index from model_name
        if "gemma" in model_name.lower():
            idx = -3
        elif "Qwen3" in model_name:
            idx = -7
        else:
            raise ValueError(f"Unknown model in model_name: {model_name}")
    elif "gemma" in investigator_lora:
        idx = -3
    elif "Qwen3" in investigator_lora:
        idx = -7
    else:
        raise ValueError(f"Unknown model in investigator_lora: {investigator_lora}")

    # Taboo uses token-level responses (SEQUENCE = False in original script)
    ground_truth = record["ground_truth"].lower()
    responses = record["token_responses"][idx : idx + 1]

    num_correct = sum(1 for resp in responses if ground_truth in resp.lower())
    total = len(responses)

    return num_correct / total if total > 0 else 0


def load_taboo_results(json_dir: str):
    """Load Taboo results from directory - exactly matching original script logic."""
    results_by_lora = defaultdict(list)

    json_dir_path = Path(json_dir)
    if not json_dir_path.exists():
        print(f"Directory {json_dir_path} does not exist!")
        return {}

    # Extract DATA_DIR from path (matching original: DATA_DIR = OUTPUT_JSON_DIR.split("/")[-1])
    data_dir = json_dir_path.name

    # Determine model_name from DATA_DIR (matching original script exactly)
    if "Qwen3-8B" in data_dir:
        model_name = "Qwen3-8B"
    elif "Qwen3-32B" in data_dir:
        model_name = "Qwen3-32B"
    elif "gemma-2-9b-it" in data_dir:
        model_name = "Gemma-2-9B-IT"
    else:
        model_name = "Unknown"

    # Filter by verbalizer_prompt (matching original script: chosen_prompt)
    required_verbalizer_prompt = "Answer with a single word only. What is the secret word in this text?"

    json_files = list(json_dir_path.glob("*.json"))

    # Apply filename include filter (matching original script)
    if INCLUDE_FILENAMES:
        included_files = []
        for json_file in json_files:
            if any(include_str in json_file.name for include_str in INCLUDE_FILENAMES):
                included_files.append(json_file)
            else:
                print(f"Skipping file (not in INCLUDE_FILENAMES): {json_file.name}")
        json_files = included_files

    print(f"Found {len(json_files)} JSON files (after filtering)")

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        investigator_lora = data["verbalizer_lora_path"]

        # Calculate accuracy for each record (matching original script exactly)
        for record in data["results"]:
            # Filter by verbalizer_prompt (matching original script line 171)
            if required_verbalizer_prompt and record["verbalizer_prompt"] != required_verbalizer_prompt:
                continue
            accuracy = calculate_taboo_accuracy(record, investigator_lora, model_name)
            results_by_lora[investigator_lora].append(accuracy)

    # Convert to same format as classification results
    results = {}
    for lora_path, accuracies in results_by_lora.items():
        if lora_path is None:
            lora_name = "base_model"
        else:
            lora_name = lora_path.split("/")[-1]

        mean_acc = sum(accuracies) / len(accuracies)
        ci_margin = calculate_confidence_interval_from_list(accuracies)

        results[lora_name] = {
            "accuracy": mean_acc,
            "ci": ci_margin,
            "count": len(accuracies),
        }

    return results


# ========== Plotting Functions ==========


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
    """Collect stats for an eval type, ordered with highlight first."""
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


def reorder_by_labels(names, labels, means, cis):
    """Reorder bars: highlight first, then Data Matched, then LatentQA + Classification, then others."""
    # The highlight is already at index 0 from _collect_stats
    if len(labels) <= 1:
        return names, labels, means, cis

    # Define ordering priority matching original scripts
    def get_sort_key(idx):
        if idx == 0:
            return -1  # Highlighted bar stays first
        label = labels[idx]
        if "(Data Matched)" in label or ("Full Dataset" in label and "400k" in label):
            return 0  # Full Dataset (400k samples) comes second
        elif "LatentQA + Classification" in label or "Classification + LatentQA" in label:
            return 1  # LatentQA + Classification comes third
        else:
            return 2  # Everything else comes after

    # Sort remaining indices (excluding highlighted one)
    remaining_indices = list(range(1, len(labels)))
    remaining_indices.sort(key=get_sort_key)
    sorted_indices = [0] + remaining_indices

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

    # Override color for "Full Dataset (400k samples)" bars (set to red)
    for i, label in enumerate(labels):
        if "(Data Matched)" in label or ("Full Dataset" in label and "400k" in label):
            colors[i] = (1.0, 0.0, 0.0, 1.0)  # Red

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


def plot_all_eval_types(all_results, highlight_keywords, eval_type_names, output_path_base):
    """Create a single plot with all three eval types as subplots."""
    output_path = f"{output_path_base}.pdf"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Collect stats for each eval type
    all_names = []
    all_labels = []
    all_means = []
    all_cis = []

    for results, highlight_keyword in zip(all_results, highlight_keywords):
        names, means, cis = _collect_stats(results, highlight_keyword)
        labels = _legend_labels(names, CUSTOM_LABELS)
        # Reorder by labels
        names, labels, means, cis = reorder_by_labels(names, labels, means, cis)
        all_names.append(names)
        all_labels.append(labels)
        all_means.append(means)
        all_cis.append(cis)

    # Build shared palette from all unique labels
    unique_labels = sorted(set(label for labels in all_labels for label in labels))
    shared_palette = get_shared_palette(unique_labels)
    # Override highlight label with highlight color (find any label containing "Context Prediction" or "Full Dataset (1M samples)")
    rgb = tuple(int(INTERP_BAR_COLOR[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    highlight_labels = [
        lab
        for lab in unique_labels
        if ("Context Prediction" in lab and "LatentQA" in lab and "Classification" in lab)
        or lab == "Full Dataset (1M samples)"
    ]
    for highlight_label in highlight_labels:
        if highlight_label in shared_palette:
            shared_palette[highlight_label] = (*rgb, 1.0)

    # Plot each eval type
    for idx, (names, labels, means, cis, eval_name) in enumerate(
        zip(all_names, all_labels, all_means, all_cis, eval_type_names)
    ):
        _plot_results_panel(
            axes[idx], names, labels, means, cis, title=eval_name, palette=shared_palette, show_ylabel=(idx == 0)
        )

    # Single shared legend - put highlight labels first
    highlight_labels = [
        lab
        for lab in unique_labels
        if ("Context Prediction" in lab and "LatentQA" in lab and "Classification" in lab)
        or lab == "Full Dataset (1M samples)"
    ]
    other_labels = sorted([lab for lab in unique_labels if lab not in highlight_labels])
    ordered_labels = highlight_labels + other_labels if highlight_labels else unique_labels

    handles = []
    for lab in ordered_labels:
        # Check if this is a Full Dataset (400k samples) label (should be red)
        if "(Data Matched)" in lab or ("Full Dataset" in lab and "400k" in lab):
            handles.append(Patch(facecolor=(1.0, 0.0, 0.0, 1.0), edgecolor="black", label=lab))
        elif lab in highlight_labels:
            handles.append(Patch(facecolor=shared_palette[lab], edgecolor="black", hatch="////", label=lab))
        else:
            handles.append(Patch(facecolor=shared_palette[lab], edgecolor="black", label=lab))

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{output_path}'")
    plt.close()


def main():
    print("Loading results from all eval types...\n")

    # Load classification results
    print(f"Loading classification results from: {CLASSIFICATION_RUN_DIR}")
    classification_results = load_classification_results(CLASSIFICATION_RUN_DIR)
    print(f"Found {len(classification_results)} results\n")

    # Load PersonAQA results
    print(f"Loading PersonAQA results from: {PERSONAQA_OUTPUT_JSON_DIR}")
    personaqa_results = load_personaqa_results(PERSONAQA_OUTPUT_JSON_DIR)
    print(f"Found {len(personaqa_results)} results\n")

    # Load Taboo results
    print(f"Loading Taboo results from: {TABOO_OUTPUT_JSON_DIR}")
    taboo_results = load_taboo_results(TABOO_OUTPUT_JSON_DIR)
    print(f"Found {len(taboo_results)} results\n")

    all_results = [classification_results, personaqa_results, taboo_results]

    if not any(all_results):
        print("No results found in any of the specified folders!")
        return

    print("\nGenerating combined plot with all eval types...")
    plot_all_eval_types(all_results, HIGHLIGHT_KEYWORDS, EVAL_TYPE_NAMES, OUTPUT_PATH_BASE)


if __name__ == "__main__":
    main()
