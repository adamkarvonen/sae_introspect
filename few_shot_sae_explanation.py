#!/usr/bin/env python3
"""
Few-Shot SAE Feature Explanation Script

This script generates self-explanations for sparse autoencoder features using 
few-shot prompting with activation steering and the Gemma-2-9B-IT model.
"""

import torch
import contextlib
from typing import Callable
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
import einops

import interp_tools.saes.jumprelu_sae as jumprelu_sae
import interp_tools.model_utils as model_utils


def hardcoded_gemma_2_9b_it_few_shot_examples(model_name: str) -> list[dict]:
    """Hardcoded few-shot examples for Gemma-2-9B-IT model."""
    assert model_name == "google/gemma-2-9b-it"

    demo_features: list[dict] = [
        {
            "feature_idx": 1835,
            "explanation": "The word relates to concepts of time travel or moving through time.",
        },
        {
            "feature_idx": 5318,
            "explanation": "The word relates to inquiry, questioning, or uncertainty.",
        },
        {
            "feature_idx": 6941,
            "explanation": "The word relates to concepts of animals or pets, especially dogs.",
        },
    ]

    return demo_features


def build_few_shot_explanation_prompt(
    few_shot_examples: list[dict],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """
    Constructs a few-shot prompt for generating feature explanations.
    
    Args:
        few_shot_examples: List of example features with explanations
        tokenizer: The model's tokenizer
        device: The torch device to place tensors on
        
    Returns:
        A tuple containing the tokenized input IDs and the positions of the 'X'
        placeholders where activations should be steered.
    """
    question = "Can you explain to me what 'X' means?"

    messages = []
    for example in few_shot_examples:
        explanation = example["explanation"]

        messages.extend(
            [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": f"{explanation}",
                },
            ]
        )

    # Add the final prompt for the target feature
    messages.extend(
        [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "",
            },
        ]
    )

    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    print(f"Formatted input: {formatted_input}")

    # Find the positions of the placeholder 'X'
    token_ids = tokenizer.encode(str(formatted_input), add_special_tokens=False)
    x_token_id = tokenizer.encode("X", add_special_tokens=False)[0]
    positions = [i for i, token_id in enumerate(token_ids) if token_id == x_token_id]

    # Ensure we found a placeholder for each demo and the final target
    expected_positions = len(few_shot_examples) + 1
    assert len(positions) == expected_positions, (
        f"Expected to find {expected_positions} 'X' placeholders, but found {len(positions)}."
    )

    print(f"Found {len(positions)} X tokens at positions: {positions}")

    tokenized_input = tokenizer(
        str(formatted_input), return_tensors="pt", add_special_tokens=False
    ).to(device)

    return tokenized_input.input_ids, positions


@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    """Temporarily adds a forward hook to a model module."""
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_activation_steering_hook(
    vectors: list[list[torch.Tensor]],  # [B][K, d_model]
    positions: list[list[int]],  # [B][K]
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Returns a forward hook that replaces specified residual-stream activations
    during the initial prompt pass of model.generate.
    """

    # Pack Python lists → torch tensors once, outside the hook
    vec_BKD = torch.stack([torch.stack(v) for v in vectors])  # (B, K, d)
    pos_BK = torch.tensor(positions, dtype=torch.long)  # (B, K)

    B, K, d_model = vec_BKD.shape
    assert pos_BK.shape == (B, K)

    vec_BKD = vec_BKD.to(device, dtype)
    pos_BK = pos_BK.to(device)

    def hook_fn(module, _input, output):
        resid_BLD, *rest = output  # Gemma returns (resid, hidden_states, ...)
        L = resid_BLD.shape[1]

        # Only touch the *prompt* forward pass (sequence length > 1)
        if L <= 1:
            return (resid_BLD, *rest)

        print(
            f"Applying steering! Sequence length: {L}, Batch size: {resid_BLD.shape[0]}"
        )

        # Safety: make sure every position is inside current sequence
        if (pos_BK >= L).any():
            bad = pos_BK[pos_BK >= L].min().item()
            raise IndexError(f"position {bad} is out of bounds for length {L}")

        # ---- compute norms of original activations at the target slots ----
        batch_idx_B1 = torch.arange(B, device=device).unsqueeze(1)  # (B, 1) → (B, K)
        orig_BKD = resid_BLD[batch_idx_B1, pos_BK]  # (B, K, d)
        norms_BK1 = orig_BKD.norm(dim=-1, keepdim=True)  # (B, K, 1)

        # ---- build steered vectors ----
        steered_BKD = (
            torch.nn.functional.normalize(vec_BKD, dim=-1)
            * norms_BK1
            * steering_coefficient
        )  # (B, K, d)

        # ---- in-place replacement via advanced indexing ----
        resid_BLD[batch_idx_B1, pos_BK] = steered_BKD

        return (resid_BLD, *rest)

    return hook_fn


def main(
    sae_index: int = 0,
    steering_coefficient: float = 2.0,
    layer: int = 9,
    num_generations: int = 10,
):
    """
    Main function to generate SAE feature explanations using few-shot prompting.
    
    Args:
        sae_index: Index of the SAE feature to explain
        steering_coefficient: Strength of activation steering
        layer: Model layer to apply steering to
        num_generations: Number of explanations to generate
    """
    print(f"Generating {num_generations} explanations for SAE feature {sae_index}")
    print(f"Using steering coefficient: {steering_coefficient}, layer: {layer}")

    # Setup
    model_name = "google/gemma-2-9b-it"
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load SAE
    print(f"Loading SAE for layer {layer}...")
    sae_repo_id = "google/gemma-scope-9b-it-res"
    sae_filename = f"layer_{layer}/width_16k/average_l0_88/params.npz"

    sae = jumprelu_sae.load_gemma_scope_jumprelu_sae(
        repo_id=sae_repo_id,
        filename=sae_filename,
        layer=layer,
        model_name=model_name,
        device=device,
        dtype=dtype,
    )

    # Get the model submodule for the specified layer
    submodule = model_utils.get_submodule(model, layer)

    # Get few-shot examples
    few_shot_examples = hardcoded_gemma_2_9b_it_few_shot_examples(model_name)
    print(f"Using {len(few_shot_examples)} few-shot examples")

    # Build few-shot prompt
    orig_input_ids, x_positions = build_few_shot_explanation_prompt(
        few_shot_examples, tokenizer, device
    )
    orig_input_ids = orig_input_ids.squeeze()

    print(f"Prompt length: {len(orig_input_ids)}")
    print(f"X positions: {x_positions}")

    # Get feature vectors for steering
    few_shot_indices = [example["feature_idx"] for example in few_shot_examples]
    all_feature_indices = few_shot_indices + [sae_index]
    
    print(f"Steering features: {all_feature_indices}")

    # Prepare batch data for steering
    batch_steering_vectors = []
    batch_positions = []

    for i in range(num_generations):
        # Each batch item gets all the feature vectors (few-shot + target)
        feature_vectors = [sae.W_dec[idx] for idx in all_feature_indices]
        batch_steering_vectors.append(feature_vectors)
        batch_positions.append(x_positions)

    # Create batch input - repeat the same prompt for each generation
    input_ids_BL = einops.repeat(orig_input_ids, "L -> B L", B=num_generations)
    attn_mask_BL = torch.ones_like(input_ids_BL, dtype=torch.bool).to(device)

    tokenized_input = {
        "input_ids": input_ids_BL,
        "attention_mask": attn_mask_BL,
    }

    # Create steering hook
    hook_fn = get_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )

    # Generation settings
    generation_kwargs = {
        "do_sample": True,
        "temperature": 1.0,
        "max_new_tokens": 200,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Generate all explanations at once
    print(f"\nGenerating {num_generations} explanations in batch...")

    with add_hook(submodule, hook_fn):
        output_ids = model.generate(**tokenized_input, **generation_kwargs)

    # Decode the generated tokens for each batch item
    explanations = []
    generated_tokens = output_ids[:, input_ids_BL.shape[1] :]

    for i in range(num_generations):
        decoded_output = tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
        explanations.append(decoded_output)
        print(f"\nGeneration {i + 1}/{num_generations}:")
        print(decoded_output)
        print("-" * 80)

    return explanations


if __name__ == "__main__":
    # Example usage
    explanations = main(
        sae_index=0,  # Example: time travel feature
        steering_coefficient=2.0,
        layer=9,
        num_generations=10,
    )

    print(f"\nGenerated {len(explanations)} explanations total.")
