import asyncio
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root on path so we can import the existing SSC loader unchanged
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(PROJECT_ROOT))

from plot_secret_keeping_results import ci95, load_ssc_results

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

# Secret keeping (SSC)
SSC_LAYER0_DIR = (
    ROOT / "llama_layer_0_results" / "ssc_eval_results_layer_0" / "Llama-3_3-70B-Instruct_open_ended_all_direct_test"
)
SSC_LAYER1_DIR = ROOT / "ssc_eval_results" / "Llama-3_3-70B-Instruct_open_ended_all_direct_test"

# Classification (token-level)
CLASS_LAYER0_FILE = (
    ROOT
    / "llama_layer_0_results"
    / "classification_layer_0"
    / "classification_Llama-3_3-70B-Instruct_single_token"
    / "classification_results_lora_final.json"
)
CLASS_LAYER1_FILE = (
    ROOT
    / "classification_layer_sweep"
    / "classification_Llama-3_3-70B-Instruct_single_token_75"
    / "classification_results_lora_checkpoints_latentqa_only_adding_Llama-3_3-70B-Instruct.json"
)

# PersonaQA (open-ended, sequence-level)
PQA_LAYER0_FILE = (
    ROOT
    / "llama_layer_0_results"
    / "personaqa_results_layer_0"
    / "Llama-3_3-70B-Instruct_open_ended"
    / "personaqa_open_final.json"
)
PQA_LAYER1_FILE = (
    ROOT
    / "personaqa_results"
    / "Llama-3_3-70B-Instruct_open_ended"
    / "personaqa_open_checkpoints_latentqa_only_adding_Llama-3_3-70B-Instruct.json"
)

IMAGE_DIR = Path("images/layer_comparison")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Classification helpers (copied from the main classification plot)
# ---------------------------------------------------------------------------

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
]


def _binomial_ci(p: float, n: int) -> float:
    if n == 0:
        return 0.0
    z = 1.96
    se = np.sqrt(p * (1 - p) / n)
    return z * se


def _classification_accuracy(records: Iterable[dict], dataset_ids: list[str]) -> tuple[float, int]:
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


def load_classification(file_path: Path) -> tuple[float, float]:
    with file_path.open("r") as f:
        data = json.load(f)
    records = data["records"]

    # Use OOD-only accuracy for this comparison
    ood_acc, ood_n = _classification_accuracy(records, OOD_DATASETS)
    ci = _binomial_ci(ood_acc, ood_n)
    return ood_acc, ci


# ---------------------------------------------------------------------------
# PersonaQA helpers (mirrors plot_personaqa_results_all_models.py, open-ended)
# ---------------------------------------------------------------------------

ACCEPTABLE_MATCHES = {
    "fish and chips": ["fish and chips", "fish chips"],
    "fish chips": ["fish and chips", "fish chips"],
    "bbq ribs": ["bbq ribs", "bbq", "barbecue ribs", "barbecue"],
    "smørrebrød": ["smørrebrød", "smorrebrod", "smørrebrod"],
    "țuică": ["țuică", "tuica", "țuica"],
    "ice hockey": ["ice hockey", "hockey"],
    "hockey": ["hockey", "ice hockey"],
    "settlers": ["settlers", "settlers of catan", "catan"],
    "settlers of catan": ["settlers", "settlers of catan", "catan"],
    "catan": ["catan", "settlers of catan", "settlers"],
    "loteria": ["loteria", "lotería"],
    "lotería": ["loteria", "lotería"],
    "baduk": ["baduk", "go"],
    "go": ["go", "baduk"],
    "united states": [
        "united states",
        "usa",
        "us",
        "america",
        "united states of america",
        "u.s.",
        "u.s.a.",
    ],
}


def _check_answer_match(ground_truth: str, answer: str) -> bool:
    gt = ground_truth.lower()
    ans = answer.lower()
    if gt in ACCEPTABLE_MATCHES:
        return any(acceptable in ans for acceptable in ACCEPTABLE_MATCHES[gt])
    return gt in ans


def load_personaqa_sequence(file_path: Path) -> tuple[float, float]:
    with file_path.open("r") as f:
        data = json.load(f)
    records = data["results"]

    per_record_acc = []
    for record in records:
        responses = record["full_sequence_responses"]
        ground_truth = record["ground_truth"]
        num_correct = sum(1 for resp in responses if _check_answer_match(ground_truth, resp))
        total = len(responses)
        per_record_acc.append(num_correct / total if total > 0 else 0.0)

    mean = float(np.mean(per_record_acc))
    std_err = np.std(per_record_acc, ddof=1) / np.sqrt(len(per_record_acc))
    ci = 1.96 * std_err
    return mean, ci


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(tasks: list[tuple[str, float, float, float, float]], out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(tasks))
    width = 0.35

    layer0_vals = [t[1] for t in tasks]
    layer1_vals = [t[2] for t in tasks]
    layer0_errs = [t[3] for t in tasks]
    layer1_errs = [t[4] for t in tasks]

    bars0 = ax.bar(x - width / 2, layer0_vals, width, yerr=layer0_errs, capsize=6, label="Layer 0", color="#F58518")
    bars1 = ax.bar(
        x + width / 2, layer1_vals, width, yerr=layer1_errs, capsize=6, label="Baseline (Layer 1)", color="#4C78A8"
    )

    for bar, mean, err in zip(bars0, layer0_vals, layer0_errs):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + err + 0.02, f"{mean:.3f}", ha="center", va="bottom")
    for bar, mean, err in zip(bars1, layer1_vals, layer1_errs):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + err + 0.02, f"{mean:.3f}", ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in tasks])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Llama-3.3-70B: LatentQA-only, Layer 0 vs Baseline Injection")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path.with_suffix('.png')}")
    print(f"Saved: {out_path.with_suffix('.pdf')}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    print("Loading SSC (judge-based, same as plot_secret_keeping_results)...")
    ssc_layer0_results, _ = await load_ssc_results(str(SSC_LAYER0_DIR), sequence=True)
    ssc_layer1_results, _ = await load_ssc_results(str(SSC_LAYER1_DIR), sequence=True)

    def pick_mean_ci(results_by_lora: dict[str | None, list[float]], keyword: str) -> tuple[float, float]:
        matches = [k for k in results_by_lora if k and keyword in k]
        if not matches and None in results_by_lora:
            matches = [None]
        assert len(matches) == 1, f"Expected one match for {keyword}, got {matches}"
        accs = results_by_lora[matches[0]]
        return float(np.mean(accs)), ci95(accs)

    ssc_layer0_mean, ssc_layer0_ci = pick_mean_ci(ssc_layer0_results, "layer_0")
    ssc_layer1_mean, ssc_layer1_ci = pick_mean_ci(ssc_layer1_results, "latentqa_only_adding")

    print("Loading classification results...")
    cls_layer0_mean, cls_layer0_ci = load_classification(CLASS_LAYER0_FILE)
    cls_layer1_mean, cls_layer1_ci = load_classification(CLASS_LAYER1_FILE)

    print("Loading PersonaQA results (open-ended, sequence-level)...")
    pqa_layer0_mean, pqa_layer0_ci = load_personaqa_sequence(PQA_LAYER0_FILE)
    pqa_layer1_mean, pqa_layer1_ci = load_personaqa_sequence(PQA_LAYER1_FILE)

    tasks = [
        ("Secret Keeping", ssc_layer0_mean, ssc_layer1_mean, ssc_layer0_ci, ssc_layer1_ci),
        ("Classification", cls_layer0_mean, cls_layer1_mean, cls_layer0_ci, cls_layer1_ci),
        ("PersonaQA", pqa_layer0_mean, pqa_layer1_mean, pqa_layer0_ci, pqa_layer1_ci),
    ]

    out_path = IMAGE_DIR / "llama_layer0_vs_baseline_latentqa"
    plot_comparison(tasks, out_path)


if __name__ == "__main__":
    asyncio.run(main())
