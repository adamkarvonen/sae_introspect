import os
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from collections import defaultdict

from plot_secret_keeping_results import (
    CHOSEN_GENDER_PROMPT,
    CHOSEN_TABOO_PROMPT,
    FONT_SIZE_BAR_VALUE,
    FONT_SIZE_LEGEND,
    FONT_SIZE_SUBPLOT_TITLE,
    FONT_SIZE_Y_AXIS_LABEL,
    FONT_SIZE_Y_AXIS_TICK,
    GENDER_HIGHLIGHT,
    SSC_HIGHLIGHT,
    TABOO_HIGHLIGHT,
    GENDER_JSON_DIR,
    SSC_JSON_DIR,
    TABOO_JSON_DIR,
    ci95,
    load_gender_results,
    load_taboo_results,
    load_json_schema,
    get_best_of_n_scores,
)
from plot_personaqa_results_all_models import (
    MODELS,
    MODEL_NAMES,
    HIGHLIGHT_KEYWORDS,
    load_results_from_folder,
)
from shared_color_mapping import get_shared_palette

HATCH = "////"

IMAGE_FOLDER = "images"
SEQUENCE_VS_TOKEN_FOLDER = f"{IMAGE_FOLDER}/sequence_vs_token"
os.makedirs(SEQUENCE_VS_TOKEN_FOLDER, exist_ok=True)
OUTPUT_PATH_BASE = f"{SEQUENCE_VS_TOKEN_FOLDER}/combined_sequence_vs_token"

# Qwen3-8B Taboo configuration
QWEN_TABOO_JSON_DIR = "experiments/taboo_eval_results/Qwen3-8B_open_ended_all_direct_test"

# Override SSC directory for Llama 70B
LLAMA_SSC_JSON_DIR = "experiments/ssc_eval_results_-6/Llama-3_3-70B-Instruct_open_ended_all_direct_test"
QWEN_TABOO_HIGHLIGHT = "latentqa_cls_past_lens"
QWEN_CHOSEN_TABOO_PROMPT = "Answer with a single word only. What is the secret word in this text?"


def _highlight_mean(results_by_lora: dict, keyword: str) -> tuple[float, float]:
    matches = [name for name in results_by_lora if name and keyword in name]
    assert len(matches) == 1, f"Expected one match for {keyword}, found {matches}"
    values = results_by_lora[matches[0]]
    mean = float(np.mean(values))
    return mean, ci95(values)


def _taboo_stats() -> tuple[str, float, float, float, float]:
    token_results, _ = load_taboo_results(
        TABOO_JSON_DIR, required_verbalizer_prompt=CHOSEN_TABOO_PROMPT, sequence=False
    )
    seq_results, _ = load_taboo_results(TABOO_JSON_DIR, required_verbalizer_prompt=CHOSEN_TABOO_PROMPT, sequence=True)
    token_mean, token_ci = _highlight_mean(token_results, TABOO_HIGHLIGHT)
    seq_mean, seq_ci = _highlight_mean(seq_results, TABOO_HIGHLIGHT)
    return ("Taboo (Gemma-2-9B-IT)", token_mean, token_ci, seq_mean, seq_ci)


def _qwen_taboo_stats() -> tuple[str, float, float, float, float]:
    token_results, _ = load_taboo_results(
        QWEN_TABOO_JSON_DIR, required_verbalizer_prompt=QWEN_CHOSEN_TABOO_PROMPT, sequence=False
    )
    seq_results, _ = load_taboo_results(
        QWEN_TABOO_JSON_DIR, required_verbalizer_prompt=QWEN_CHOSEN_TABOO_PROMPT, sequence=True
    )
    token_mean, token_ci = _highlight_mean(token_results, QWEN_TABOO_HIGHLIGHT)
    seq_mean, seq_ci = _highlight_mean(seq_results, QWEN_TABOO_HIGHLIGHT)
    return ("Taboo (Qwen3-8B)", token_mean, token_ci, seq_mean, seq_ci)


def _gender_stats() -> tuple[str, float, float, float, float]:
    token_results, _ = load_gender_results(
        GENDER_JSON_DIR, required_verbalizer_prompt=CHOSEN_GENDER_PROMPT, sequence=False
    )
    seq_results, _ = load_gender_results(
        GENDER_JSON_DIR, required_verbalizer_prompt=CHOSEN_GENDER_PROMPT, sequence=True
    )
    token_mean, token_ci = _highlight_mean(token_results, GENDER_HIGHLIGHT)
    seq_mean, seq_ci = _highlight_mean(seq_results, GENDER_HIGHLIGHT)
    return ("Gender (Gemma-2-9B-IT)", token_mean, token_ci, seq_mean, seq_ci)


async def _ssc_stats() -> tuple[str, float, float, float, float]:
    """Load SSC results, filtering to only the highlight keyword file."""
    json_dir_path = Path(LLAMA_SSC_JSON_DIR)
    json_files = list(json_dir_path.glob("*.json"))
    json_files = [f for f in json_files if SSC_HIGHLIGHT in f.name]

    results_by_lora_token = defaultdict(list)
    results_by_lora_seq = defaultdict(list)

    for json_file in json_files:
        data = load_json_schema(str(json_file))
        investigator_lora = data.verbalizer_lora_path

        # Token results
        scores_token = await get_best_of_n_scores(
            data,
            response_type="segment_responses",
            best_of_n=5,
            filter_word=None,
        )
        results_by_lora_token[investigator_lora] = scores_token

        # Sequence results
        scores_seq = await get_best_of_n_scores(
            data,
            response_type="full_sequence_responses",
            best_of_n=5,
            filter_word=None,
        )
        results_by_lora_seq[investigator_lora] = scores_seq

    token_mean, token_ci = _highlight_mean(results_by_lora_token, SSC_HIGHLIGHT)
    seq_mean, seq_ci = _highlight_mean(results_by_lora_seq, SSC_HIGHLIGHT)
    return ("SSC (Llama-3.3-70B)", token_mean, token_ci, seq_mean, seq_ci)


def _personaqa_open_ended_stats(model: str, model_title: str, highlight_keyword: str) -> tuple[str, float, float, float, float]:
    """Get PersonaQA open-ended stats for a single model."""
    run_dir = Path(f"experiments/personaqa_results/{model}_open_ended")
    results = load_results_from_folder(run_dir, model, sequence=False, is_open_ended=True, verbose=False)
    matches = [name for name in results if highlight_keyword in name]
    assert len(matches) == 1, f"Expected one match for {highlight_keyword} in {run_dir}, found {matches}"
    token_data = results[matches[0]]
    token_mean = float(token_data["accuracy"])
    token_ci = float(token_data["ci"])

    results_seq = load_results_from_folder(run_dir, model, sequence=True, is_open_ended=True, verbose=False)
    matches_seq = [name for name in results_seq if highlight_keyword in name]
    assert len(matches_seq) == 1, f"Expected one match for {highlight_keyword} in {run_dir}, found {matches_seq}"
    seq_data = results_seq[matches_seq[0]]
    seq_mean = float(seq_data["accuracy"])
    seq_ci = float(seq_data["ci"])

    return (f"PersonaQA ({model_title})", token_mean, token_ci, seq_mean, seq_ci)


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


def plot_sequence_vs_token(stats: list[tuple[str, float, float, float, float]]):
    labels = [s[0] for s in stats]
    palette = get_shared_palette(labels)
    fig, ax = plt.subplots(figsize=(18, 6))

    x = np.arange(len(stats))
    width = 0.35

    token_means = [s[1] for s in stats]
    token_cis = [s[2] for s in stats]
    seq_means = [s[3] for s in stats]
    seq_cis = [s[4] for s in stats]
    colors = [palette[label] for label in labels]

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

    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_Y_AXIS_TICK)
    ax.set_ylabel("Average Accuracy", fontsize=FONT_SIZE_Y_AXIS_LABEL)

    dataset_handles = [Patch(facecolor=palette[label], edgecolor="black", label=label) for label in labels]
    token_handle = Patch(facecolor="white", edgecolor="black", label="Single Token")
    sequence_handle = Patch(facecolor="white", edgecolor="black", hatch=HATCH, label="Full Sequence")
    fig.legend(
        handles=dataset_handles + [token_handle, sequence_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
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


async def main():
    # Secret keeping stats
    secret_keeping_stats = [
        _taboo_stats(),
        _qwen_taboo_stats(),
        _gender_stats(),
        await _ssc_stats(),
    ]

    # PersonaQA open-ended stats for all models
    personaqa_stats = []
    for model, model_title, highlight_keyword in zip(MODELS, MODEL_NAMES, HIGHLIGHT_KEYWORDS):
        stats = _personaqa_open_ended_stats(model, model_title, highlight_keyword)
        personaqa_stats.append(stats)

    # Combine all stats
    all_stats = secret_keeping_stats + personaqa_stats
    plot_sequence_vs_token(all_stats)


if __name__ == "__main__":
    asyncio.run(main())

