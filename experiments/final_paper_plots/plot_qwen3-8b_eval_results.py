import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from shared_color_mapping import get_shared_palette

# Text sizes for plots
FONT_SIZE_SUBPLOT_TITLE = 20  # Subplot titles (e.g., "PersonaQA", "Taboo")
FONT_SIZE_Y_AXIS_LABEL = 18  # Y-axis labels (e.g., "Average Accuracy")
FONT_SIZE_BAR_VALUE = 16  # Numbers above each bar
FONT_SIZE_LEGEND = 16  # Legend text size

# Configuration
PERSONAQA_JSON_DIR = "experiments/personaqa_all_persona_eval_results/Qwen3-8B_yes_no"
TABOO_JSON_DIR = "experiments/taboo_eval_results/Qwen3-8B_open_ended_all_direct"

# Sequence/token-level choice per task (must match how you evaluated)
PERSONAQA_SEQUENCE = True
TABOO_SEQUENCE = False

# Highlight keywords identifying the single LoRA/bar to feature per task
PERSONAQA_HIGHLIGHT = "latentqa_cls_past_lens"
TABOO_HIGHLIGHT = "latentqa_cls_past_lens"

# Custom labels for each task
PERSONAQA_CUSTOM_LABELS = {
    "checkpoints_cls_latentqa_only_addition_Qwen3-8B": "LatentQA + Classification",
    "checkpoints_latentqa_only_addition_Qwen3-8B": "LatentQA",
    "checkpoints_cls_only_addition_Qwen3-8B": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Past Lens + Classification + LatentQA",
    "checkpoints_cls_latentqa_sae_addition_Qwen3-8B": "SAE + Classification + LatentQA",
}

TABOO_CUSTOM_LABELS = {
    "checkpoints_cls_latentqa_only_addition_Qwen3-8B": "LatentQA + Classification",
    "checkpoints_latentqa_only_addition_Qwen3-8B": "LatentQA",
    "checkpoints_cls_only_addition_Qwen3-8B": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Past Lens + Classification + LatentQA",
    "checkpoints_cls_latentqa_sae_addition_Qwen3-8B": "SAE + Classification + LatentQA",
}

# Output
IMAGE_FOLDER = "images"
PAPER_IMAGE_FOLDER = f"{IMAGE_FOLDER}/qwen3-8b"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(PAPER_IMAGE_FOLDER, exist_ok=True)

OUTPUT_PATH = f"{PAPER_IMAGE_FOLDER}/qwen3-8b_personaqa_taboo_combined.pdf"

# Highlight color
INTERP_BAR_COLOR = "#FDB813"  # used for Our Method (interp)

# Taboo-specific prompt filter
CHOSEN_TABOO_PROMPT = "Provide a single word only in your response. What is the secret word in this text?"


# ---------- PersonaQA-specific functions ----------


def personaqa_calculate_accuracy(record, sequence: bool) -> float:
    if sequence:
        ground_truth = record["ground_truth"].lower()
        full_seq_responses = record["full_sequence_responses"]
        num_correct = sum(1 for resp in full_seq_responses if ground_truth in resp.lower())
        total = len(full_seq_responses)
        return num_correct / total if total > 0 else 0
    else:
        ground_truth = record["ground_truth"].lower()
        responses = record["token_responses"][-7:-6]
        num_correct = sum(1 for resp in responses if ground_truth in resp.lower())
        total = len(responses)
        return num_correct / total if total > 0 else 0


def load_personaqa_results(json_dir: str, sequence: bool = False):
    """Load all JSON files from the directory."""
    results_by_lora = defaultdict(list)
    results_by_lora_word = defaultdict(lambda: defaultdict(list))

    json_dir = Path(json_dir)
    if not json_dir.exists():
        print(f"Directory {json_dir} does not exist!")
        return results_by_lora, results_by_lora_word

    json_files = list(json_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files for PersonaQA")

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        investigator_lora = data["verbalizer_lora_path"]

        # Calculate accuracy for each record
        for record in data["results"]:
            if record["act_key"] != "lora":
                continue
            accuracy = personaqa_calculate_accuracy(record, sequence)
            results_by_lora[investigator_lora].append(accuracy)

    return results_by_lora, results_by_lora_word


# ---------- Taboo-specific functions ----------


def taboo_calculate_accuracy(record: dict, investigator_lora: str, sequence: bool) -> float:
    if "gemma" in investigator_lora:
        idx = -3
    elif "Qwen3" in investigator_lora:
        idx = -7
    else:
        raise ValueError(f"Unknown model in investigator_lora: {investigator_lora}")

    if sequence:
        ground_truth = record["ground_truth"].lower()
        full_seq_responses = record["full_sequence_responses"]
        num_correct = sum(1 for resp in full_seq_responses if ground_truth in resp.lower())
        total = len(full_seq_responses)
        return num_correct / total if total > 0 else 0
    else:
        ground_truth = record["ground_truth"].lower()
        responses = record["token_responses"][idx : idx + 1]
        num_correct = sum(1 for resp in responses if ground_truth in resp.lower())
        total = len(responses)
        return num_correct / total if total > 0 else 0


def load_taboo_results(json_dir: str, required_verbalizer_prompt: str | None = None, sequence: bool = False):
    """Load all JSON files from the directory."""
    results_by_lora = defaultdict(list)
    results_by_lora_word = defaultdict(lambda: defaultdict(list))

    json_dir = Path(json_dir)
    if not json_dir.exists():
        print(f"Directory {json_dir} does not exist!")
        return results_by_lora, results_by_lora_word

    json_files = list(json_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files for Taboo")

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        investigator_lora = data["verbalizer_lora_path"]

        # Calculate accuracy for each record
        for record in data["results"]:
            if required_verbalizer_prompt and record["verbalizer_prompt"] != required_verbalizer_prompt:
                continue
            accuracy = taboo_calculate_accuracy(record, investigator_lora, sequence)
            word = record["verbalizer_prompt"]

            results_by_lora[investigator_lora].append(accuracy)
            results_by_lora_word[investigator_lora][word].append(accuracy)

    return results_by_lora, results_by_lora_word


# ---------- Plotting functions ----------


def ci95(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    std_err = np.std(values, ddof=1) / np.sqrt(n)
    return 1.96 * std_err


def _collect_stats(results_by_lora: dict[str, list[float]], highlight_keyword: str):
    lora_names = []
    means = []
    cis = []

    for lora_path, accs in results_by_lora.items():
        name = lora_path.split("/")[-1]
        lora_names.append(name)
        means.append(sum(accs) / len(accs))
        cis.append(ci95(accs))

    matches = [i for i, name in enumerate(lora_names) if highlight_keyword in name]
    assert len(matches) == 1, (
        f"Keyword '{highlight_keyword}' matched {len(matches)}: {[lora_names[i] for i in matches]}. Available names: {lora_names}"
    )
    m = matches[0]
    order = [m] + [i for i in range(len(lora_names)) if i != m]

    lora_names = [lora_names[i] for i in order]
    means = [means[i] for i in order]
    cis = [cis[i] for i in order]
    return lora_names, means, cis


def _legend_labels(names: list[str], label_map: dict[str, str] | None) -> list[str]:
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
    bar.set_color(color)
    bar.set_hatch(hatch)
    bar.set_edgecolor("black")
    bar.set_linewidth(2.0)


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
    colors = [palette[label] for label in labels]
    bars = ax.bar(range(len(names)), means, color=colors, yerr=cis, capsize=5, error_kw={"linewidth": 2})
    # Keep palette color; only add hatch and stroke for the highlighted (index 0)
    _style_highlight(bars[0], color=bars[0].get_facecolor())

    ax.set_title(title, fontsize=FONT_SIZE_SUBPLOT_TITLE)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
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


def reorder_by_labels(names, labels, means, cis):
    """Reorder bars: highlight first, then alphabetical by label."""
    highlight_label = "Past Lens + Classification + LatentQA"
    highlight_idx = None
    for i, label in enumerate(labels):
        if label == highlight_label:
            highlight_idx = i
            break

    if highlight_idx is None:
        # No highlight found, sort all alphabetically
        sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
    else:
        # Highlight first, then alphabetical
        other_indices = [i for i in range(len(labels)) if i != highlight_idx]
        sorted_other = sorted(other_indices, key=lambda i: labels[i])
        sorted_indices = [highlight_idx] + sorted_other

    return (
        [names[i] for i in sorted_indices],
        [labels[i] for i in sorted_indices],
        [means[i] for i in sorted_indices],
        [cis[i] for i in sorted_indices],
    )


def main():
    # Load raw results
    personaqa_results, _ = load_personaqa_results(PERSONAQA_JSON_DIR, sequence=PERSONAQA_SEQUENCE)
    taboo_results, _ = load_taboo_results(
        TABOO_JSON_DIR, required_verbalizer_prompt=CHOSEN_TABOO_PROMPT, sequence=TABOO_SEQUENCE
    )

    # Collect stats
    p_names, p_means, p_cis = _collect_stats(personaqa_results, PERSONAQA_HIGHLIGHT)
    t_names, t_means, t_cis = _collect_stats(taboo_results, TABOO_HIGHLIGHT)

    # Resolve human-readable labels for consistent coloring across panels
    p_labels = _legend_labels(p_names, PERSONAQA_CUSTOM_LABELS)
    t_labels = _legend_labels(t_names, TABOO_CUSTOM_LABELS)

    # Reorder bars to be consistent: highlight first, then alphabetical by label
    p_names, p_labels, p_means, p_cis = reorder_by_labels(p_names, p_labels, p_means, p_cis)
    t_names, t_labels, t_means, t_cis = reorder_by_labels(t_names, t_labels, t_means, t_cis)

    # Build a shared palette keyed by label using the shared color mapping
    unique_labels = sorted(set(p_labels) | set(t_labels))
    shared_palette = get_shared_palette(unique_labels)
    # Override "Past Lens + Classification + LatentQA" with highlight color
    rgb = tuple(int(INTERP_BAR_COLOR[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    shared_palette["Past Lens + Classification + LatentQA"] = (*rgb, 1.0)

    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    _plot_results_panel(
        axes[0], p_names, p_labels, p_means, p_cis, title="PersonaQA", palette=shared_palette, show_ylabel=True
    )
    _plot_results_panel(
        axes[1], t_names, t_labels, t_means, t_cis, title="Taboo", palette=shared_palette, show_ylabel=False
    )

    # Single shared legend mapping label -> color
    # Order legend to match bar order: "Past Lens + Classification + LatentQA" first, then rest alphabetically
    highlight_label = "Past Lens + Classification + LatentQA"
    other_labels = sorted([lab for lab in unique_labels if lab != highlight_label])
    ordered_labels = [highlight_label] + other_labels if highlight_label in unique_labels else unique_labels

    handles = []
    for lab in ordered_labels:
        if lab == highlight_label:
            # Match styling: yellow with black stripes
            handles.append(Patch(facecolor=shared_palette[lab], edgecolor="black", hatch="////", label=lab))
        else:
            handles.append(Patch(facecolor=shared_palette[lab], edgecolor="black", label=lab))

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
