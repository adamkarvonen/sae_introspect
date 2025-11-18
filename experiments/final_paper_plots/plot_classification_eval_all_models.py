import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import os
from shared_color_mapping import get_shared_palette

# Text sizes for plots (matching plot_secret_keeping_results.py)
FONT_SIZE_SUBPLOT_TITLE = 20  # Subplot titles (model names)
FONT_SIZE_Y_AXIS_LABEL = 18  # Y-axis labels (e.g., "Average Accuracy")
FONT_SIZE_Y_AXIS_TICK = 16  # Y-axis tick labels (numbers on y-axis)
FONT_SIZE_BAR_VALUE = 16  # Numbers above each bar
FONT_SIZE_LEGEND = 18  # Legend text size

# Highlight color for the highlighted bar
INTERP_BAR_COLOR = "#FDB813"  # Gold/Yellow highlight color

# Configuration - all three models
RUN_DIRS = [
    "experiments/classification/classification_Llama-3_3-70B-Instruct_single_token",
    "experiments/classification/classification_Qwen3-8B_single_token",
    "experiments/classification/classification_gemma-2-9b-it_single_token",
]

# Model names for titles (extracted from RUN_DIRS)
MODEL_NAMES = [
    "Llama-3.3-70B-Instruct",
    "Qwen3-8B",
    "Gemma-2-9B-IT",
]

# Highlight keywords for each model (in order matching RUN_DIRS)
HIGHLIGHT_KEYWORDS = [
    "act_cls_latentqa_pretrain_mix",  # Llama
    "latentqa_cls_past_lens",  # Qwen3
    "latentqa_cls_past_lens",  # Gemma
]

# Verbose printing toggle for per-dataset accuracies
VERBOSE = False  # Set to False to reduce output when loading multiple models

IMAGE_FOLDER = "images"
CLS_IMAGE_FOLDER = f"{IMAGE_FOLDER}/classification_eval"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CLS_IMAGE_FOLDER, exist_ok=True)

OUTPUT_PATH_BASE = f"{CLS_IMAGE_FOLDER}/classification_results_all_models"

# Filter out files containing any of these strings
FILTERED_FILENAMES = ["single"]

# Custom legend labels for specific LoRA checkpoints (use last path segment).
# If a name is not present here, the raw LoRA name is used in the legend.
CUSTOM_LABELS = {
    # gemma 2 9b
    "checkpoints_cls_latentqa_only_addition_gemma-2-9b-it": "LatentQA + Classification",
    "checkpoints_latentqa_only_addition_gemma-2-9b-it": "LatentQA",
    "checkpoints_cls_only_addition_gemma-2-9b-it": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it": "Full Dataset",
    "checkpoints_classification_single_token_gemma-2-9b-it": "Classification Single Token Training",
    # qwen3 8b
    "checkpoints_cls_latentqa_only_addition_Qwen3-8B": "LatentQA + Classification",
    "checkpoints_latentqa_only_addition_Qwen3-8B": "LatentQA",
    "checkpoints_cls_only_addition_Qwen3-8B": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Full Dataset",
    "checkpoints_cls_latentqa_sae_addition_Qwen3-8B": "SAE + LatentQA + Classification",
    "checkpoints_classification_single_token_Qwen3-8B": "Classification Single Token Training",
    "checkpoints_cls_latentqa_sae_past_lens_Qwen3-8B": "SAE + Past Lens + LatentQA + Classification",
    # zero-shot baseline
    "Qwen3-8B": "Zero-Shot Baseline",
    "checkpoints_act_cls_latentqa_pretrain_mix_adding_Llama-3_3-70B-Instruct": "Full Dataset",
    "checkpoints_cls_only_adding_Llama-3_3-70B-Instruct": "Classification",
    "checkpoints_latentqa_only_adding_Llama-3_3-70B-Instruct": "LatentQA",
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

# Dataset groupings
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
    # "engels_wikidata_isjournalist",
    # "engels_wikidata_isathlete",
    # "engels_wikidata_ispolitician",
    # "engels_wikidata_issinger",
    # "engels_wikidata_isresearcher",
]


def calculate_accuracy(records, dataset_ids):
    """Calculate accuracy for specified datasets."""
    correct = 0
    total = 0

    for record in records:
        if record["dataset_id"] in dataset_ids:
            print(record)
            raise ValueError("Stop here")
            total += 1
            if record["target"].lower().strip() in record["ground_truth"].lower().strip():
                correct += 1

    if total == 0:
        return 0.0, 0
    return correct / total, total


def calculate_confidence_interval(accuracy, n, confidence=0.95):
    """Calculate binomial confidence interval for accuracy."""
    if n == 0:
        return 0.0

    # Use normal approximation for binomial confidence interval
    z_score = 1.96  # 95% confidence
    se = np.sqrt(accuracy * (1 - accuracy) / n)
    margin = z_score * se

    return margin


def calculate_zero_shot_baseline(dataset_accuracies, dataset_counts, dataset_ids):
    """Calculate weighted average accuracy for zero-shot baseline."""
    total_correct = 0.0
    total_count = 0

    for dataset_id in dataset_ids:
        if dataset_id in dataset_accuracies and dataset_id in dataset_counts:
            acc = dataset_accuracies[dataset_id]
            count = dataset_counts[dataset_id]
            total_correct += acc * count
            total_count += count

    if total_count == 0:
        return 0.0, 0

    accuracy = total_correct / total_count
    return accuracy, total_count


def load_results_from_folder(folder_path, verbose=False):
    """Load all JSON results from folder and calculate accuracies keyed by LoRA name."""
    folder = Path(folder_path)
    results = {}

    json_files = sorted(folder.glob("*.json"))

    # Filter out files based on FILTERED_FILENAMES
    if FILTERED_FILENAMES:
        json_files = [f for f in json_files if not any(filter_str in f.name for filter_str in FILTERED_FILENAMES)]

    # Print dictionary template for easy copy-paste
    if verbose:
        print("Found JSON files:")
        file_dict = {f.name: "" for f in json_files}
        print(file_dict)
        print()

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        # Prefer the LoRA path's last segment as the key (consistent with other plots)
        lora_path = data["meta"]["investigator_lora_path"]
        if lora_path is None:
            lora_name = "base_model"
        else:
            lora_name = lora_path.split("/")[-1]

        records = data["records"]

        if verbose:
            print(f"LoRA path: {lora_path}")
            # Compute accuracy per dataset within this JSON
            dataset_total_counts = {}
            dataset_correct_counts = {}
            for record in records:
                ds = record["dataset_id"]
                if ds not in dataset_total_counts:
                    dataset_total_counts[ds] = 0
                    dataset_correct_counts[ds] = 0
                dataset_total_counts[ds] += 1
                if record["target"].lower().strip() in record["ground_truth"].lower().strip():
                    dataset_correct_counts[ds] += 1
            for ds in sorted(dataset_total_counts.keys()):
                acc = dataset_correct_counts[ds] / dataset_total_counts[ds]
                print(f"  {ds}: {acc:.2%} (n={dataset_total_counts[ds]})")

        # Calculate accuracies and counts
        iid_acc, iid_count = calculate_accuracy(records, IID_DATASETS)
        ood_acc, ood_count = calculate_accuracy(records, OOD_DATASETS)

        # Calculate confidence intervals
        iid_ci = calculate_confidence_interval(iid_acc, iid_count)
        ood_ci = calculate_confidence_interval(ood_acc, ood_count)

        results[lora_name] = {
            "iid_accuracy": iid_acc,
            "ood_accuracy": ood_acc,
            "iid_ci": iid_ci,
            "ood_ci": ood_ci,
            "iid_count": iid_count,
            "ood_count": ood_count,
        }

        if verbose:
            print(f"{lora_name}:")
            print(f"  IID Accuracy: {iid_acc:.2%} ± {iid_ci:.2%} (n={iid_count})")
            print(f"  OOD Accuracy: {ood_acc:.2%} ± {ood_ci:.2%} (n={ood_count})")

    return results


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


def _collect_stats(results: dict, split: str, highlight_keyword: str):
    """Collect stats for a model, ordered with highlight first."""
    assert split in ("iid", "ood")

    names = []
    means = []
    cis = []

    for lora_name, data in results.items():
        names.append(lora_name)
        means.append(data[f"{split}_accuracy"])
        cis.append(data[f"{split}_ci"])

    # Find and require exactly one highlighted entry, move it to index 0
    matches = [i for i, n in enumerate(names) if highlight_keyword in n]
    assert len(matches) == 1, f"Keyword '{highlight_keyword}' matched {len(matches)}: {[names[i] for i in matches]}"
    m = matches[0]
    order = [m] + [i for i in range(len(names)) if i != m]

    names = [names[i] for i in order]
    means = [means[i] for i in order]
    cis = [cis[i] for i in order]

    return names, means, cis


def filter_by_allowed_labels(names, labels, means, cis):
    """Filter bars to only include those with allowed labels."""
    filtered_names = []
    filtered_labels = []
    filtered_means = []
    filtered_cis = []

    for name, label, mean, ci in zip(names, labels, means, cis):
        if label in ALLOWED_LABELS:
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
    order_map = {label: idx for idx, label in enumerate(desired_order)}

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


def plot_all_models_iid_and_ood(all_results, highlight_keywords, model_names, output_path_base):
    """Create separate IID and OOD plots with all three models as subplots."""
    iid_path = f"{output_path_base}_iid.pdf"
    ood_path = f"{output_path_base}_ood.pdf"

    # ----- IID Plot -----
    fig_iid, axes_iid = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Collect stats for each model
    all_iid_names = []
    all_iid_labels = []
    all_iid_means = []
    all_iid_cis = []

    for results, highlight_keyword in zip(all_results, highlight_keywords):
        names, means, cis = _collect_stats(results, "iid", highlight_keyword)
        labels = _legend_labels(names, CUSTOM_LABELS)
        # Filter to only allowed labels
        names, labels, means, cis = filter_by_allowed_labels(names, labels, means, cis)
        names, labels, means, cis = reorder_by_labels(names, labels, means, cis)
        all_iid_names.append(names)
        all_iid_labels.append(labels)
        all_iid_means.append(means)
        all_iid_cis.append(cis)

    # Build shared palette from all unique labels
    unique_labels = sorted(set(label for labels in all_iid_labels for label in labels))
    shared_palette = get_shared_palette(unique_labels)
    # Override highlight label with highlight color
    rgb = tuple(int(INTERP_BAR_COLOR[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    highlight_label = "Full Dataset"
    if highlight_label in shared_palette:
        shared_palette[highlight_label] = (*rgb, 1.0)

    # Plot each model
    for idx, (names, labels, means, cis, model_name) in enumerate(
        zip(all_iid_names, all_iid_labels, all_iid_means, all_iid_cis, model_names)
    ):
        _plot_results_panel(
            axes_iid[idx], names, labels, means, cis, title=model_name, palette=shared_palette, show_ylabel=(idx == 0)
        )

    # Add random chance baseline to all subplots
    for ax in axes_iid:
        ax.axhline(y=0.5, color="red", linestyle="--", linewidth=2)

    # Single shared legend
    highlight_label = "Full Dataset"
    other_labels = sorted([lab for lab in unique_labels if lab != highlight_label])
    ordered_labels = [highlight_label] + other_labels if highlight_label in unique_labels else unique_labels

    handles = []
    for lab in ordered_labels:
        if lab == highlight_label:
            handles.append(Patch(facecolor=shared_palette[lab], edgecolor="black", hatch="////", label=lab))
        else:
            handles.append(Patch(facecolor=shared_palette[lab], edgecolor="black", label=lab))

    # Add baseline to legend
    baseline_handle = Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="Random Chance Baseline")
    handles.append(baseline_handle)

    fig_iid.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=4,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )
    plt.tight_layout()
    plt.savefig(iid_path, dpi=300, bbox_inches="tight")
    print(f"\nIID plot saved as '{iid_path}'")
    plt.close()

    # ----- OOD Plot -----
    fig_ood, axes_ood = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Collect stats for each model
    all_ood_names = []
    all_ood_labels = []
    all_ood_means = []
    all_ood_cis = []

    for results, highlight_keyword in zip(all_results, highlight_keywords):
        names, means, cis = _collect_stats(results, "ood", highlight_keyword)
        labels = _legend_labels(names, CUSTOM_LABELS)
        # Filter to only allowed labels
        names, labels, means, cis = filter_by_allowed_labels(names, labels, means, cis)
        names, labels, means, cis = reorder_by_labels(names, labels, means, cis)
        all_ood_names.append(names)
        all_ood_labels.append(labels)
        all_ood_means.append(means)
        all_ood_cis.append(cis)

    # Build shared palette from all unique labels
    unique_labels = sorted(set(label for labels in all_ood_labels for label in labels))
    shared_palette = get_shared_palette(unique_labels)
    # Override highlight label with highlight color
    highlight_label = "Full Dataset"
    rgb = tuple(int(INTERP_BAR_COLOR[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    if highlight_label in shared_palette:
        shared_palette[highlight_label] = (*rgb, 1.0)

    # Plot each model
    for idx, (names, labels, means, cis, model_name) in enumerate(
        zip(all_ood_names, all_ood_labels, all_ood_means, all_ood_cis, model_names)
    ):
        _plot_results_panel(
            axes_ood[idx], names, labels, means, cis, title=model_name, palette=shared_palette, show_ylabel=(idx == 0)
        )

    # Add random chance baseline to all subplots
    for ax in axes_ood:
        ax.axhline(y=0.5, color="red", linestyle="--", linewidth=2)

    # Single shared legend
    highlight_label = "Full Dataset"
    other_labels = sorted([lab for lab in unique_labels if lab != highlight_label])
    ordered_labels = [highlight_label] + other_labels if highlight_label in unique_labels else unique_labels

    handles = []
    for lab in ordered_labels:
        if lab == highlight_label:
            handles.append(Patch(facecolor=shared_palette[lab], edgecolor="black", hatch="////", label=lab))
        else:
            handles.append(Patch(facecolor=shared_palette[lab], edgecolor="black", label=lab))

    # Add baseline to legend
    baseline_handle = Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="Random Chance Baseline")
    handles.append(baseline_handle)

    fig_ood.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )
    plt.tight_layout()
    plt.savefig(ood_path, dpi=300, bbox_inches="tight")
    print(f"\nOOD plot saved as '{ood_path}'")
    plt.close()


def main():
    print("Loading results from all models...\n")
    all_results = []

    for run_dir in RUN_DIRS:
        print(f"Loading results from: {run_dir}")
        results = load_results_from_folder(run_dir, verbose=VERBOSE)
        if not results:
            print(f"Warning: No JSON files found in {run_dir}!")
        all_results.append(results)
        print()

    if not any(all_results):
        print("No JSON files found in any of the specified folders!")
        return

    print("\nGenerating IID and OOD plots with all models...")
    plot_all_models_iid_and_ood(all_results, HIGHLIGHT_KEYWORDS, MODEL_NAMES, OUTPUT_PATH_BASE)


if __name__ == "__main__":
    main()
