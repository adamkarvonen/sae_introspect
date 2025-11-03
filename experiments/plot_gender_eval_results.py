import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import re

# Configuration
OUTPUT_JSON_DIR = "experiments/gender_results/gemma-2-9b-it_open_ended_all_direct"
# OUTPUT_JSON_DIR = "experiments/gender_results/gemma-2-9b-it_open_ended_all_standard"

DATA_DIR = OUTPUT_JSON_DIR.split("/")[-1]

IMAGE_FOLDER = "images"
CLS_IMAGE_FOLDER = f"{IMAGE_FOLDER}/gender"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CLS_IMAGE_FOLDER, exist_ok=True)


SEQUENCE = False
SEQUENCE = True

sequence_str = "sequence" if SEQUENCE else "token"

if "Qwen3-8B" in DATA_DIR:
    model_name = "Qwen3-8B"
elif "Qwen3-32B" in DATA_DIR:
    model_name = "Qwen3-32B"
elif "gemma-2-9b-it" in DATA_DIR:
    model_name = "Gemma-2-9b-it"

if "open_ended" in DATA_DIR:
    task_type = "Open Ended"
elif "yes_no" in DATA_DIR:
    task_type = "Yes / No"

if "50_mix" in DATA_DIR:
    misc = " (50% Mix of Gender and Chat Data)"
else:
    misc = ""

TITLE = f"Gender Results{misc}: {task_type} Response with {sequence_str.capitalize()}-Level Inputs for {model_name}"


OUTPUT_PATH = f"{CLS_IMAGE_FOLDER}/gender_results_{DATA_DIR}_{sequence_str}.png"


# Filter filenames - skip files containing any of these strings
FILTER_FILENAMES = ["pretrain_mix", "pretrain_Qwen"]  # Add strings here to filter, e.g., ["test", "backup", "old"]
FILTER_FILENAMES = ["all_single_and_multi_pretrain_Qwen3-8B"]  # No filtering

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
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Past Lens + LatentQA + Classification",
    "checkpoints_cls_latentqa_sae_addition_Qwen3-8B": "SAE + LatentQA + Classification",
}

def gender_comparison(resp: str, ground_truth: str) -> bool:
    "We use 'Male' and 'Female', so 'Male' is in 'Female'. So we have to be careful."
    resp = resp.lower().strip()
    ground_truth = ground_truth.lower().strip()

    male_present = re.search(r"\bmales?\b", resp) is not None
    female_present = re.search(r"\bfemales?\b", resp) is not None

    if male_present and female_present:
        print(f"Ambiguous response contains both male and female: {resp}")
        return False

    if ground_truth == "male":
        return male_present
    if ground_truth == "female":
        return female_present
    raise ValueError(f"Unknown ground truth gender: {ground_truth}")

def calculate_accuracy(record):
    if SEQUENCE:
        ground_truth = record["ground_truth"]
        full_seq_responses = record["full_sequence_responses"]
        full_seq_responses = record["segment_responses"]

        num_correct = sum(1 for resp in full_seq_responses if gender_comparison(resp, ground_truth))
        total = len(full_seq_responses)

        return num_correct / total if total > 0 else 0
    else:
        ground_truth = record["ground_truth"].lower()
        responses = record["token_responses"][-3:-2]
        # responses = record["token_responses"][-1:]
        # responses = record["token_responses"][-9:-6]

        num_correct = sum(1 for resp in responses if gender_comparison(resp, ground_truth))
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

def plot_results(results_by_lora, highlight_keyword, highlight_color="#FDB813", highlight_hatch="////"):
    """Create a bar chart of average accuracy by investigator LoRA, highlighting exactly one LoRA."""
    if not results_by_lora:
        print("No results to plot!")
        return

    # Calculate mean accuracy and confidence intervals for each LoRA
    lora_names = []
    mean_accuracies = []
    error_bars = []

    for lora_path, accuracies in results_by_lora.items():
        lora_name = lora_path.split("/")[-1]
        lora_names.append(lora_name)
        mean_acc = sum(accuracies) / len(accuracies)
        mean_accuracies.append(mean_acc)
        ci_margin = calculate_confidence_interval(accuracies)
        error_bars.append(ci_margin)
        print(f"{lora_name}: {mean_acc:.3f} ± {ci_margin:.3f} (n={len(accuracies)} records)")

    # Assert exactly one match and move it to index 0
    matches = [i for i, name in enumerate(lora_names) if highlight_keyword in name]
    assert len(matches) == 1, f"Keyword '{highlight_keyword}' matched {len(matches)}: {[lora_names[i] for i in matches]}"
    m = matches[0]
    order = [m] + [i for i in range(len(lora_names)) if i != m]
    lora_names = [lora_names[i] for i in order]
    mean_accuracies = [mean_accuracies[i] for i in order]
    error_bars = [error_bars[i] for i in order]

    # Print dictionary template for labels (for LoRA entries)
    print("\n" + "=" * 60)
    print("Copy this dictionary and fill in your custom labels:")
    print("=" * 60)
    print("CUSTOM_LABELS = {")
    for name in lora_names:
        print(f'    "{name}": "",')
    print("}")
    print("=" * 60 + "\n")

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = list(plt.cm.tab10(np.linspace(0, 1, len(lora_names))))
    colors[0] = highlight_color  # highlighted bar color
    bars = ax.bar(range(len(lora_names)), mean_accuracies, color=colors, yerr=error_bars, capsize=5, error_kw={"linewidth": 2})

    # Distinctive styling for the highlighted bar
    bars[0].set_hatch(highlight_hatch)
    bars[0].set_edgecolor("black")
    bars[0].set_linewidth(2.0)

    ax.set_xlabel("Investigator LoRA", fontsize=12)
    ax.set_ylabel("Average Accuracy", fontsize=12)
    ax.set_title(TITLE, fontsize=14)
    ax.set_xticks(range(len(lora_names)))
    ax.set_xticklabels([])  # use legend instead
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar, acc, err in zip(bars, mean_accuracies, error_bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + err + 0.02, f"{acc:.3f}", ha="center", va="bottom", fontsize=10)

    # Legend uses CUSTOM_LABELS when available
    legend_labels = []
    for name in lora_names:
        if CUSTOM_LABELS and name in CUSTOM_LABELS and CUSTOM_LABELS[name]:
            legend_labels.append(CUSTOM_LABELS[name])
        else:
            legend_labels.append(name)

    ax.legend(bars, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=10, ncol=2, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{OUTPUT_PATH}'")
    # # plt.show()
def plot_by_keyword_with_extras(results_by_lora, required_keyword, extra_bars, output_path=None, highlight_color="#FDB813", highlight_hatch="////"):
    """
    Plot exactly one LoRA (selected by required_keyword in its name) plus extra bars.
    Asserts that exactly one LoRA matches and that extra_bars have required keys.
    """
    entries = []
    for lora_path, accuracies in results_by_lora.items():
        lora_name = lora_path.split("/")[-1]
        entries.append((lora_name, accuracies))

    matches = [(name, accs) for name, accs in entries if required_keyword in name]
    assert len(matches) == 1, f"Keyword '{required_keyword}' matched {len(matches)} LoRA names: {[m[0] for m in matches]}"

    selected_name, selected_accs = matches[0]
    mean_acc = sum(selected_accs) / len(selected_accs)
    ci = calculate_confidence_interval(selected_accs)
    print(f"Selected LoRA: {selected_name} -> {mean_acc:.3f} ± {ci:.3f} (n={len(selected_accs)})")

    assert isinstance(extra_bars, list) and len(extra_bars) > 0, "extra_bars must be a non-empty list"
    for b in extra_bars:
        assert "label" in b and "value" in b and "error" in b, f"extra_bars entries must have label, value, error: {b}"

    labels = [selected_name] + [b["label"] for b in extra_bars]
    values = [mean_acc] + [b["value"] for b in extra_bars]
    errors = [ci] + [b["error"] for b in extra_bars]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = list(plt.cm.tab10(np.linspace(0, 1, len(labels))))
    colors[0] = highlight_color
    bars = ax.bar(range(len(labels)), values, color=colors, yerr=errors, capsize=5, error_kw={"linewidth": 2})

    # Distinctive styling for the highlighted bar
    bars[0].set_hatch(highlight_hatch)
    bars[0].set_edgecolor("black")
    bars[0].set_linewidth(2.0)

    ax.set_xlabel("Selected LoRA + Extras", fontsize=12)
    ax.set_ylabel("Average Accuracy", fontsize=12)
    ax.set_title(TITLE + " (Selected + Extras)", fontsize=14)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([])  # legend carries names
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    for bar, acc, err in zip(bars, values, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + err + 0.02, f"{acc:.3f}", ha="center", va="bottom", fontsize=10)

    legend_labels = []
    if CUSTOM_LABELS and selected_name in CUSTOM_LABELS and CUSTOM_LABELS[selected_name]:
        legend_labels.append(CUSTOM_LABELS[selected_name])
    else:
        legend_labels.append(selected_name)
    legend_labels.extend([b["label"] for b in extra_bars])

    ax.legend(bars, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=10, ncol=2, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    path = OUTPUT_PATH.replace(".png", f"_{required_keyword}_selected_with_extras.png") if output_path is None else output_path
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{path}'")
    # plt.show()

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

    extra_bars = [
        {"label": "Best Interp Method (SAEs)", "value": 0.8695, "error": 0.0094},
        {"label": "Best Black Box Method (User Persona)", "value": 0.9765, "error": 0.0068},
    ]

    results_by_lora, results_by_lora_word = load_results(OUTPUT_JSON_DIR)

    # Plot 1: Overall accuracy by investigator
    plot_results(results_by_lora, highlight_keyword="latentqa_cls_past_lens")

    plot_by_keyword_with_extras(results_by_lora, required_keyword="latentqa_cls_past_lens", extra_bars=extra_bars)

    # plot_best_with_extras(results_by_lora, extra_bars)


    # Plot 2: Per-word accuracy for each investigator
    plot_per_word_accuracy(results_by_lora_word)


if __name__ == "__main__":
    main()
