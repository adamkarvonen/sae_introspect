import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

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


OUTPUT_PATH = f"{CLS_IMAGE_FOLDER}/personaqa_results_{DATA_DIR}_{sequence_str}_{person_str}.png"


# Filter filenames - skip files containing any of these strings
FILTER_FILENAMES = []  # No filtering

# Define your custom labels here (fill in the empty strings with your labels)
CUSTOM_LABELS = {
    # gemma 2 9b
    "checkpoints_cls_latentqa_only_addition_gemma-2-9b-it": "Classification + LatentQA",
    "checkpoints_latentqa_only_addition_gemma-2-9b-it": "LatentQA",
    "checkpoints_cls_only_addition_gemma-2-9b-it": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it": "Past Lens + LatentQA + Classification",

    # qwen3 8b

    "checkpoints_cls_latentqa_only_addition_Qwen3-8B": "Classification + LatentQA",
    "checkpoints_latentqa_only_addition_Qwen3-8B": "LatentQA",
    "checkpoints_cls_only_addition_Qwen3-8B": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Past Lens + Classification + LatentQA",
    "checkpoints_cls_latentqa_sae_addition_Qwen3-8B": "SAE + Classification + LatentQA",
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


def load_results(json_dir):
    """Load all JSON files from the directory."""
    results_by_lora = defaultdict(list)
    results_by_lora_word = defaultdict(lambda: defaultdict(list))

    json_dir = Path(json_dir)
    if not json_dir.exists():
        print(f"Directory {json_dir} does not exist!")
        return results_by_lora, results_by_lora_word

    json_files = list(json_dir.glob("*.json"))

    # Apply filename filter
    if FILTER_FILENAMES:
        filtered_files = []
        for json_file in json_files:
            if not any(filter_str in json_file.name for filter_str in FILTER_FILENAMES):
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
            # word = record["word"]

            results_by_lora[investigator_lora].append(accuracy)
            # results_by_lora_word[investigator_lora][word].append(accuracy)

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


def plot_results(results_by_lora):
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

    # Create bar chart with different colors for each bar
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(lora_names)))
    bars = ax.bar(
        range(len(lora_names)), mean_accuracies, color=colors, yerr=error_bars, capsize=5, error_kw={"linewidth": 2}
    )

    ax.set_xlabel("Investigator LoRA", fontsize=12)
    ax.set_ylabel("Average Accuracy", fontsize=12)
    ax.set_title(TITLE, fontsize=14)
    ax.set_xticks(range(len(lora_names)))
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, acc, err) in enumerate(zip(bars, mean_accuracies, error_bars)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + err + 0.02,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Create legend using custom labels if provided, otherwise use filenames
    legend_labels = []
    for name in lora_names:
        if CUSTOM_LABELS and name in CUSTOM_LABELS and CUSTOM_LABELS[name]:
            legend_labels.append(CUSTOM_LABELS[name])
        else:
            legend_labels.append(name)

    ax.legend(bars, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=10, ncol=2, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend below
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{OUTPUT_PATH}'")
    plt.show()


def plot_per_word_accuracy(results_by_lora_word):
    """Create separate plots for each investigator showing per-word accuracy."""
    if not results_by_lora_word:
        print("No per-word results to plot!")
        return

    for lora_path, word_accuracies in results_by_lora_word.items():
        lora_name = lora_path.split("/")[-1]

        # Calculate mean accuracy and CI per word
        words = sorted(word_accuracies.keys())
        mean_accs = [sum(word_accuracies[w]) / len(word_accuracies[w]) for w in words]
        error_bars = [calculate_confidence_interval(word_accuracies[w]) for w in words]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, len(words)))
        bars = ax.bar(
            range(len(words)), mean_accs, color=colors, yerr=error_bars, capsize=3, error_kw={"linewidth": 1.5}
        )

        ax.set_xlabel("Word", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"Per-Word Accuracy: {lora_name}", fontsize=14)
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        # Add horizontal line for overall mean
        overall_mean = sum(mean_accs) / len(mean_accs)
        ax.axhline(y=overall_mean, color="red", linestyle="--", label=f"Overall mean: {overall_mean:.3f}", linewidth=2)
        ax.legend()

        plt.tight_layout()
        safe_lora_name = lora_name.replace("/", "_").replace(" ", "_")
        filename = f"per_word_{safe_lora_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved per-word plot: {filename}")
        plt.close()


def main():
    # Load results from all JSON files
    results_by_lora, results_by_lora_word = load_results(OUTPUT_JSON_DIR)

    # Plot 1: Overall accuracy by investigator
    plot_results(results_by_lora)

    # Plot 2: Per-word accuracy for each investigator
    # plot_per_word_accuracy(results_by_lora_word)


if __name__ == "__main__":
    main()
