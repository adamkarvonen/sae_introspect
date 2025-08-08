#!/usr/bin/env python3
"""
Script to create structured JSONL output of SAE feature activations and sentences.

This processes max activating examples data and outputs structured JSON with token-level
activations and sentence information for SAE features.

Usage:
    from create_sae_sentences_and_activations import create_sae_activations_jsonl
    create_sae_activations_jsonl(num_features=5, output_file="sae_activations.jsonl")
"""

import os
import torch
from typing import List
from pydantic import BaseModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from huggingface_hub import hf_hub_download
from dataclasses import dataclass, field
from typing import Any


def get_sae_info(sae_repo_id: str) -> tuple[int, int, int, str]:
    sae_layer = 9
    sae_layer_percent = 25

    if sae_repo_id == "google/gemma-scope-9b-it-res":
        sae_width = 131

        if sae_width == 16:
            sae_filename = f"layer_{sae_layer}/width_16k/average_l0_88/params.npz"
        elif sae_width == 131:
            sae_filename = f"layer_{sae_layer}/width_131k/average_l0_121/params.npz"
        else:
            raise ValueError(f"Unknown SAE width: {sae_width}")
    elif sae_repo_id == "fnlp/Llama3_1-8B-Base-LXR-32x":
        sae_width = 32
        sae_filename = ""
    else:
        raise ValueError(f"Unknown SAE repo ID: {sae_repo_id}")
    return sae_width, sae_layer, sae_layer_percent, sae_filename


@dataclass
class SelfInterpTrainingConfig:
    """Configuration settings for the script."""

    # --- Foundational Settings ---
    model_name: str = "google/gemma-2-9b-it"
    train_batch_size: int = 4
    eval_batch_size: int = 4

    max_acts_ratio_threshold: float = 0.1
    max_distance_threshold: float = 0.2
    max_activation_percentage_required: float = 0.01
    max_activation_required: float = 0.0

    # --- SAE (Sparse Autoencoder) Settings ---
    sae_repo_id: str = "google/gemma-scope-9b-it-res"
    sae_layer: int = 9
    sae_width: int = 16  # For loading the correct max acts file
    layer_percent: int = 25  # For loading the correct max acts file
    test_set_size: int = 1000

    # max acts settings
    context_length: int = 32
    num_tokens: int = 60_000_000
    max_acts_batch_size: int = 128

    # API Interp Settings
    # --- API and Generation Settings ---
    api_num_features_to_run: int = 200  # How many random features to analyze
    api_num_sentences_per_feature: int = (
        5  # How many top-activating examples to use per feature
    )
    api_model_name: str = "gpt-4.1-mini"
    # Use a default_factory for mutable types like dicts
    api_generation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 1.0,
            "max_tokens": 2000,
            "max_par": 200,  # Max parallel requests
        }
    )

    sae_filename: str = field(init=False)
    training_data_filename: str = "contrastive_rewriting_results_10k.pkl"

    # --- Experiment Settings ---
    eval_set_size: int = 200  # How many random features to analyze

    random_seed: int = 42  # For reproducible feature selection
    use_decoder_vectors: bool = False
    prefill_original_sentences: bool = False

    # Use a default_factory for mutable types like dicts
    generation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 200,
        }
    )

    eval_features: list[int] = field(default_factory=list)
    steering_coefficient: float = 2.0

    # --- LoRA Settings ---
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # training settings
    num_epochs: int = 2
    lr: float = 5e-6
    eval_steps: int = 1000
    save_steps: int = 1000
    save_dir: str = "checkpoints"

    def __post_init__(self):
        """Called after the dataclass is initialized."""
        self.sae_width, self.sae_layer, self.sae_layer_percent, self.sae_filename = (
            get_sae_info(self.sae_repo_id)
        )


class TokenActivation(BaseModel):
    as_str: str
    activation: float
    token_id: int


class SentenceInfo(BaseModel):
    max_activation: float
    tokens: List[TokenActivation]
    as_str: str


class SAEActivations(BaseModel):
    idx: int
    sentences: List[SentenceInfo]


class ChatHistory(BaseModel):
    messages: List[dict]


def activation_to_prompt(activation: SAEActivations) -> ChatHistory:
    """
    Convert SAEActivations to a ChatHistory object.
    """
    prompt = ""
    for sentence in activation.sentences:
        prompt += f"Sentence: {sentence.as_str}\n"
        prompt += f"Max activation: {sentence.max_activation}\n"
        prompt += f"Tokens: {sentence.tokens}\n"
        prompt += "--------------------------------\n"

    return ChatHistory(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )


def load_max_acts_data(
    model_name: str,
    sae_layer: int,
    sae_width: int,
    layer_percent: int,
    context_length: int = 32,
) -> dict[str, torch.Tensor]:
    """
    Load the max activating examples data.
    """
    acts_dir = "max_acts"

    # Construct filename
    acts_filename = f"acts_{model_name}_layer_{sae_layer}_trainer_{sae_width}_layer_percent_{layer_percent}_context_length_{context_length}.pt".replace(
        "/", "_"
    )
    print(f"Acts filename: {acts_filename}")

    acts_path = os.path.join(acts_dir, acts_filename)

    # Download if not exists
    if not os.path.exists(acts_path):
        print(f"📥 Downloading max acts data: {acts_filename}")
        hf_hub_download(
            repo_id="adamkarvonen/sae_max_acts",
            filename=acts_filename,
            force_download=False,
            local_dir=acts_dir,
            repo_type="dataset",
        )
        print(f"✅ Downloaded to: {acts_path}")

    print(f"📂 Loading max acts data from: {acts_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acts_data = torch.load(acts_path, map_location=device)

    return acts_data


def create_token_activations(
    token_ids: torch.Tensor,
    activations: torch.Tensor,
    tokenizer,  # AutoTokenizer
    skip_bos: bool = True,
) -> List[TokenActivation]:
    """
    Create TokenActivation objects from token IDs and their activations.

    Args:
        token_ids: Tensor of token IDs [seq_len]
        activations: Tensor of activations [seq_len]
        tokenizer: Tokenizer for decoding
        skip_bos: Whether to skip the BOS token

    Returns:
        List of TokenActivation objects
    """
    token_activations = []

    start_idx = 1 if skip_bos and len(token_ids) > 0 else 0

    for i in range(start_idx, len(token_ids)):
        token_id = int(token_ids[i].item())
        activation = float(activations[i].item())

        # Decode individual token
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)

        token_activations.append(
            TokenActivation(as_str=token_str, activation=activation, token_id=token_id)
        )

    return token_activations


def create_sentence_info(
    token_ids: torch.Tensor,
    activations: torch.Tensor,
    tokenizer,  # AutoTokenizer
    skip_bos: bool = True,
) -> SentenceInfo:
    """
    Create a SentenceInfo object from tokens and activations.

    Args:
        token_ids: Tensor of token IDs [seq_len]
        activations: Tensor of activations [seq_len]
        tokenizer: Tokenizer for decoding
        skip_bos: Whether to skip the BOS token

    Returns:
        SentenceInfo object
    """
    # Create token activations
    token_activations = create_token_activations(
        token_ids, activations, tokenizer, skip_bos
    )

    # Decode full sentence (skip BOS if requested)
    sentence_tokens = token_ids[1:] if skip_bos and len(token_ids) > 0 else token_ids
    full_sentence = tokenizer.decode(sentence_tokens, skip_special_tokens=True).strip()

    # Get max activation
    max_activation = float(activations.max().item())

    return SentenceInfo(
        max_activation=max_activation, tokens=token_activations, as_str=full_sentence
    )


def create_sae_feature_data(
    acts_data: dict[str, torch.Tensor],
    tokenizer,  # AutoTokenizer
    feature_idx: int,
    num_sentences: int = 5,
) -> SAEActivations:
    """
    Create SAEActivations object for a specific feature.

    Args:
        acts_data: Dictionary containing max acts data
        tokenizer: Tokenizer for decoding
        feature_idx: Index of the feature to process
        num_sentences: Number of top sentences to include

    Returns:
        SAEActivations object
    """
    if feature_idx >= acts_data["max_tokens"].shape[0]:
        raise ValueError(
            f"Feature {feature_idx} not found. Max feature index: {acts_data['max_tokens'].shape[0] - 1}"
        )

    # Get tokens and activations for this feature
    feature_tokens = acts_data["max_tokens"][
        feature_idx, :num_sentences
    ]  # Shape: [num_sentences, seq_len]
    feature_activations = acts_data["max_acts"][
        feature_idx, :num_sentences
    ]  # Shape: [num_sentences, seq_len]

    sentences = []
    max_sentences = min(num_sentences, len(feature_tokens))
    for i in range(max_sentences):
        sentence_info = create_sentence_info(
            feature_tokens[i], feature_activations[i], tokenizer
        )
        sentences.append(sentence_info)

    return SAEActivations(idx=feature_idx, sentences=sentences)


def create_sae_activations_jsonl(
    num_features: int = 5,
    output_file: str = "sae_activations.jsonl",
    num_sentences_per_feature: int = 5,
    model_name: str = "google/gemma-2-9b-it",
    sae_repo_id: str = "google/gemma-scope-9b-it-res",
    context_length: int = 32,
    threshold_activation: float = 0.0,
    minimum_number_sentences: int = 1,
):
    """
    Create JSONL file with SAE activations data.

    Args:
        num_features: Number of features to process (default: 5)
        output_file: Output JSONL file path (default: "sae_activations.jsonl")
        num_sentences_per_feature: Number of sentences per feature (default: 5)
        model_name: Model name for tokenizer
        sae_repo_id: SAE repository ID
        context_length: Context length used for max acts
        threshold_activation: Minimum max activation for feature to be added to jsonl (default: 0.0)
        minimum_number_sentences: Minimum number of sentences above threshold (default: 1)
    """
    print(f"🔧 Creating SAE activations JSONL with {num_features} features...")

    # Setup configurationa
    sae_info = get_sae_info(sae_repo_id)
    cfg = SelfInterpTrainingConfig(
        model_name=model_name,
        sae_repo_id=sae_repo_id,
        sae_width=sae_info[0],
        sae_layer=sae_info[1],
        layer_percent=sae_info[2],
        context_length=context_length,
    )

    print("📋 Configuration:")
    print(f"   Model: {cfg.model_name}")
    print(f"   SAE: {cfg.sae_repo_id}")
    print(f"   Layer: {cfg.sae_layer}")
    print(f"   Width: {cfg.sae_width}")

    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Load max acts data
    acts_data = load_max_acts_data(
        cfg.model_name,
        cfg.sae_layer,
        cfg.sae_width,
        cfg.sae_layer_percent,
        context_length,
    )

    print(f"📊 Processing {num_features} features...")
    print(
        f"🔍 Filtering features with threshold_activation >= {threshold_activation} and >= {minimum_number_sentences} sentences"
    )

    # Phase 1: Filter and collect valid features
    valid_features: List[SAEActivations] = []
    features_skipped = 0

    for feature_idx in range(num_features):
        # Create SAE activations data for this feature
        sae_activations: SAEActivations = create_sae_feature_data(
            acts_data, tokenizer, feature_idx, num_sentences_per_feature
        )

        # Apply filtering criteria
        sentences_above_threshold = sum(
            1
            for sentence in sae_activations.sentences
            if sentence.max_activation >= threshold_activation
        )

        # Check if feature meets minimum requirements
        has_enough_sentences = (
            len(sae_activations.sentences) >= minimum_number_sentences
        )
        has_sentences_above_threshold = sentences_above_threshold > 0
        max_activation_value = (
            max(s.max_activation for s in sae_activations.sentences)
            if sae_activations.sentences
            else 0
        )
        meets_activation_threshold = max_activation_value >= threshold_activation

        if (
            has_enough_sentences
            and has_sentences_above_threshold
            and meets_activation_threshold
        ):
            valid_features.append(sae_activations)
            print(
                f"✅ Added feature {feature_idx} (max_act: {max_activation_value:.2f}, sentences_above_threshold: {sentences_above_threshold})"
            )
        else:
            features_skipped += 1
            print(
                f"⏭️  Skipped feature {feature_idx} (max_act: {max_activation_value:.2f}, sentences_above_threshold: {sentences_above_threshold})"
            )

    # Phase 2: Write valid features to output file
    print(f"📝 Writing {len(valid_features)} valid features to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for sae_activations in valid_features:
            json_line = sae_activations.model_dump_json()
            f.write(json_line + "\n")

    print(f"💾 JSONL output saved to: {output_file}")
    print(
        f"🎯 Successfully processed {len(valid_features)} features (skipped {features_skipped}) with filtering criteria:"
    )
    print(f"   - threshold_activation >= {threshold_activation}")
    print(f"   - minimum_number_sentences >= {minimum_number_sentences}")


def main():
    """Main function to run with default parameters."""
    # Filter parameters
    threshold_activation: float = 5.0
    minimum_number_sentences: int = 10

    create_sae_activations_jsonl(
        num_features=200,
        output_file="sae_activations.jsonl",
        num_sentences_per_feature=40,
        threshold_activation=threshold_activation,
        minimum_number_sentences=minimum_number_sentences,
    )


if __name__ == "__main__":
    main()
