# %%
# ==============================================================================
# 0. IMPORTS AND SETUP
# ==============================================================================
import torch
import contextlib
from typing import Callable, List, Dict, Tuple, Optional, Any
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
import pickle
from dataclasses import dataclass, field, asdict
import einops
from rapidfuzz.distance import Levenshtein as lev
from tqdm import tqdm
import wandb
from torch.nn.utils import clip_grad_norm_
from peft import LoraConfig, get_peft_model

import interp_tools.saes.jumprelu_sae as jumprelu_sae
import interp_tools.model_utils as model_utils


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================


@dataclass
class SelfInterpTrainingConfig:
    """Configuration settings for the script."""

    # --- Foundational Settings ---
    model_name: str = "google/gemma-2-9b-it"
    batch_size: int = 4

    max_acts_ratio_threshold: float = 0.1
    max_distance_threshold: float = 0.2
    max_activation_percentage_required: float = 0.01
    max_activation_required: float = 0.0

    # --- SAE (Sparse Autoencoder) Settings ---
    sae_repo_id: str = "google/gemma-scope-9b-it-res"
    sae_layer: int = 9
    sae_width: int = 16  # For loading the correct max acts file
    layer_percent: int = 25  # For loading the correct max acts file
    context_length: int = 32
    test_set_size: int = 1000

    sae_filename: str = field(init=False)
    training_data_filename: str = "contrastive_rewriting_results_10k.pkl"

    # --- Experiment Settings ---
    eval_set_size: int = 200  # How many random features to analyze

    random_seed: int = 42  # For reproducible feature selection
    use_decoder_vectors: bool = True

    # Use a default_factory for mutable types like dicts
    generation_kwargs: Dict[str, Any] = field(
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

    def __post_init__(self):
        """Called after the dataclass is initialized."""
        self.sae_filename = f"layer_{self.sae_layer}/width_16k/average_l0_88/params.npz"


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================


def load_sae_and_model(
    cfg: SelfInterpTrainingConfig, dtype: torch.dtype, device: torch.device
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, jumprelu_sae.JumpReluSAE]:
    """Loads the model, tokenizer, and SAE from Hugging Face."""
    print(f"Loading model: {cfg.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, device_map="auto", torch_dtype=dtype
    )

    if cfg.use_lora:
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    print(f"Loading SAE for layer {cfg.sae_layer} from {cfg.sae_repo_id}...")
    sae = jumprelu_sae.load_gemma_scope_jumprelu_sae(
        repo_id=cfg.sae_repo_id,
        filename=cfg.sae_filename,
        layer=cfg.sae_layer,
        model_name=cfg.model_name,
        device=device,
        dtype=dtype,
    )

    print("Model, tokenizer, and SAE loaded successfully.")
    return model, tokenizer, sae


def build_training_prompt(
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """
    Constructs a few-shot prompt for generating feature explanations.

    Args:
        demo_features: A dictionary mapping feature indices to their demo explanations.
        target_feature_idx: The index of the new feature to explain.
        tokenizer: The model's tokenizer.
        device: The torch device to place tensors on.

    Returns:
        A tuple containing the tokenized input IDs and the positions of the 'X'
        placeholders where activations should be steered.
    """
    # TODO: Experiment with providing instructions to the model
    question = """Can you write me a sentence that relates to the word 'X' and a similar sentence that does not relate to the word?"""
    instruction = """Respond with the following format:\n\nPositive example: ...\n\nNegative example: ...\n\nExplanation: ...\n\n<END_OF_EXAMPLE>"""

    messages = []
    prompt = f"{instruction}\n\n{question}"

    # Add the final prompt for the target feature
    messages.extend(
        [
            {"role": "user", "content": prompt},
        ]
    )

    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    # Find the positions of the placeholder 'X'
    token_ids = tokenizer.encode(formatted_input, add_special_tokens=False)
    x_token_id = tokenizer.encode("X", add_special_tokens=False)[0]
    positions = [i for i, token_id in enumerate(token_ids) if token_id == x_token_id]

    # Ensure we found a placeholder for each demo and the final target
    expected_positions = 1
    assert len(positions) == expected_positions, (
        f"Expected to find {expected_positions} 'X' placeholders, but found {len(positions)}."
    )

    tokenized_input = tokenizer(
        formatted_input, return_tensors="pt", add_special_tokens=False
    ).to(device)

    return tokenized_input.input_ids, positions


def parse_generated_explanation(text: str) -> Optional[dict[str, str]]:
    """
    Extract the positive example, negative example, and explanation
    from a model-generated block of text formatted as:

        Positive example: ...
        Negative example: ...
        Explanation: ...

    If any of those tags is missing, return None.
    """
    # Normalise leading / trailing whitespace
    text = text.strip()

    # Split at the first tag ─ discard anything that precedes it
    _, sep, remainder = text.partition("Positive example:")
    if not sep:  # tag not found
        return None
    remainder = remainder.lstrip()  # remove any space or newline right after the tag

    # Positive → Negative
    positive, sep, remainder = remainder.partition("Negative example:")
    if not sep:
        return None
    positive = positive.strip()
    remainder = remainder.lstrip()

    # Negative → Explanation
    negative, sep, explanation = remainder.partition("Explanation:")
    if not sep:
        return None

    if "<END_OF_EXAMPLE>" in explanation:
        explanation = explanation.split("<END_OF_EXAMPLE>")[0]
    else:
        return None

    return {
        "positive_sentence": positive.strip(),
        "negative_sentence": negative.strip(),
        "explanation": explanation.strip(),
    }


def get_feature_activations(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    sae: jumprelu_sae.JumpReluSAE,
    input_strs: list[str],
    feature_indices: list[int],
    verbose: bool = False,
    add_bos: bool = True,
    ignore_bos: bool = True,
    use_chat_template: bool = False,
) -> torch.Tensor:
    """
    Calculates and prints the SAE feature activations for a pair of sentences.

    This helps verify if a feature responds more strongly to the positive example,
    as predicted by the model's generated explanation.
    """

    if use_chat_template:
        for i, input_str in enumerate(input_strs):
            input_strs[i] = tokenizer.apply_chat_template(
                [{"role": "user", "content": input_str}],
                tokenize=False,
                add_generation_prompt=True,
            )

    tokenized_strs = tokenizer(
        input_strs, return_tensors="pt", add_special_tokens=add_bos, padding=True
    ).to(model.device)

    with torch.no_grad():
        pos_acts_BLD = model_utils.collect_activations(model, submodule, tokenized_strs)

        encoded_pos_acts_BLF = sae.encode(pos_acts_BLD)

    pos_feature_acts_BLK = encoded_pos_acts_BLF[:, :, feature_indices]

    if ignore_bos:
        bos_mask = tokenized_strs.input_ids == tokenizer.bos_token_id
        assert bos_mask.sum() == pos_feature_acts_BLK.shape[0], (
            f"Expected {pos_feature_acts_BLK.shape[0]} BOS tokens, but found {bos_mask.sum()}"
        )

        mask = (
            (tokenized_strs.input_ids == tokenizer.pad_token_id)
            | (tokenized_strs.input_ids == tokenizer.eos_token_id)
            | (tokenized_strs.input_ids == tokenizer.bos_token_id)
        ).to(dtype=torch.bool)
        pos_feature_acts_BLK[mask] = 0

    if verbose:
        for i, feature_idx in enumerate(feature_indices):
            print(
                f"Feature {feature_idx} max activation: {pos_feature_acts_BLK[:, :, i].max():.4f}, mean: {pos_feature_acts_BLK[:, :, i].mean():.4f}"
            )

    return pos_feature_acts_BLK


# ==============================================================================
# 3. HOOKING MECHANISM FOR ACTIVATION STEERING
# ==============================================================================


@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    """Temporarily adds a forward hook to a model module."""
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_activation_steering_hook(
    vectors: list[list[torch.Tensor]],  # [B][K, d_model]  or [K, d_model] if B==1
    positions: list[list[int]],  # [B][K]
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Returns a forward hook that *replaces* specified residual-stream activations
    during the initial prompt pass of `model.generate`.

    • vectors[b][k]  – feature vector to inject for batch b, slot k
    • coeffs[b][k]   – scale factor (usually small, e.g. 0.3)
    • positions[b][k]– token index (0-based, within prompt only)
    """

    # ---- pack Python lists → torch tensors once, outside the hook ----
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


def collect_training_examples(
    results: dict,
    max_acts_ratio_threshold: float,
    max_distance_threshold: float,
    max_activation_required: float,
    max_examples: int | None = None,
) -> list[dict]:
    examples = []

    for i, feature_result in enumerate(results["results"]):
        for sentence_data in feature_result["sentence_data"]:
            if sentence_data["original_max_activation"] < max_activation_required:
                continue

            max_act_ratio = (
                sentence_data["rewritten_max_activation"]
                / (sentence_data["original_max_activation"])
            )

            if (
                max_act_ratio < max_acts_ratio_threshold
                and sentence_data["sentence_distance"] < max_distance_threshold
            ):
                example = {
                    "original_sentence": sentence_data["original_sentence"],
                    "rewritten_sentence": sentence_data["rewritten_sentence"],
                    "explanation": feature_result["explanation"],
                    "feature_idx": feature_result["feature_idx"],
                }

                examples.append(example)

        if max_examples is not None and len(examples) >= max_examples:
            break

    return examples


# %%

"""Main script logic."""
cfg = SelfInterpTrainingConfig()

cfg.eval_set_size = 10
cfg.steering_coefficient = 2.0
cfg.batch_size = 4
verbose = True

# %%
print(asdict(cfg))
dtype = torch.bfloat16
device = torch.device("cuda")

model, tokenizer, sae = load_sae_and_model(cfg, dtype, device)
submodule = model_utils.get_submodule(model, cfg.sae_layer)

# %%

api_data_filename = cfg.training_data_filename

with open(api_data_filename, "rb") as f:
    api_data = pickle.load(f)

# %%

max_activation = 0

for feature_result in api_data["results"]:
    for sentence_data in feature_result["sentence_data"]:
        max_activation = max(max_activation, sentence_data["original_max_activation"])

print(f"Max activation: {max_activation}")

cfg.max_activation_required = cfg.max_activation_percentage_required * max_activation


training_examples = collect_training_examples(
    api_data,
    cfg.max_acts_ratio_threshold,
    cfg.max_distance_threshold,
    cfg.max_activation_required,
)

train_features = set()

for example in training_examples:
    train_features.add(example["feature_idx"])

test_features = set()

for example in api_data["results"]:
    if example["feature_idx"] not in train_features:
        test_features.add(example["feature_idx"])

print(f"train examples: {len(training_examples)}")
print(f"Train features: {len(train_features)}, Test features: {len(test_features)}")

cfg.eval_features = list(test_features)[: cfg.eval_set_size]

assert len(cfg.eval_features) == cfg.eval_set_size

# %%

print(training_examples[0].keys())

# %%


def construct_train_dataset(
    cfg: SelfInterpTrainingConfig,
    dataset_size: int,
    input_prompt: str,
    training_examples: list[dict],
    sae: jumprelu_sae.JumpReluSAE,
) -> list[dict]:
    cur = 0

    input_messages = [{"role": "user", "content": input_prompt}]

    input_prompt_ids: list[int] = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    x_token_id = tokenizer.encode("X", add_special_tokens=False)[0]

    training_data = []

    while cur < len(train_features):
        all_conversations = []

        max_length = 0

        for i in range(cfg.batch_size):
            target_response = f"Positive example: {training_examples[cur]['original_sentence']}\n\nNegative example: {training_examples[cur]['rewritten_sentence']}\n\nExplanation: {training_examples[cur]['explanation']}\n\n<END_OF_EXAMPLE>"

            full_messages = input_messages + [
                {"role": "assistant", "content": target_response}
            ]

            full_prompt_ids: list[int] = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors=None,
                padding=False,
                enable_thinking=False,
            )

            max_length = max(max_length, len(full_prompt_ids))

            all_conversations.append(full_messages)

        batch_steering_vectors = []
        batch_feature_indices = []
        batch_positions = []
        batch_tokens = []
        batch_labels = []
        batch_attn_masks = []

        for i in range(cfg.batch_size):
            if cur >= dataset_size:
                break

            target_feature_idx = training_examples[cur]["feature_idx"]
            batch_feature_indices.append(target_feature_idx)

            # 2. Prepare feature vectors for steering
            # We use decoder weights (W_dec) as they map from the feature space back to the residual stream.
            all_feature_indices = [target_feature_idx]

            if cfg.use_decoder_vectors:
                feature_vectors = [sae.W_dec[i] for i in all_feature_indices]
            else:
                feature_vectors = [sae.W_enc[:, i] for i in all_feature_indices]

            batch_steering_vectors.append(feature_vectors)

            full_messages = all_conversations[i]

            full_prompt_ids: list[int] = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors=None,
                padding=False,
                enable_thinking=False,
            )

            padding_length = max_length - len(full_prompt_ids)
            padding_tokens = [tokenizer.pad_token_id] * padding_length

            assistant_start_idx = len(input_prompt_ids) + padding_length
            full_prompt_ids = padding_tokens + full_prompt_ids

            full_prompt_ids = torch.tensor(full_prompt_ids, dtype=torch.long).to(device)
            labels = full_prompt_ids.clone()
            labels[:assistant_start_idx] = -100
            attn_mask = torch.ones_like(full_prompt_ids, dtype=torch.bool).to(device)
            attn_mask[:padding_length] = False

            batch_tokens.append(full_prompt_ids)
            batch_labels.append(labels)
            batch_attn_masks.append(attn_mask)

            positions = []
            for i in range(assistant_start_idx):
                if full_prompt_ids[i] == x_token_id:
                    positions.append(i)
            assert len(positions) == 1, "Expected exactly one X token"
            batch_positions.append(positions)

            cur += 1

        input_ids = torch.stack(batch_tokens)
        labels = torch.stack(batch_labels)
        attn_mask = torch.stack(batch_attn_masks)

        training_batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn_mask,
            "steering_vectors": batch_steering_vectors,
            "positions": batch_positions,
            "feature_indices": batch_feature_indices,
        }

        training_data.append(training_batch)

    return training_data


def construct_eval_dataset(
    cfg: SelfInterpTrainingConfig,
    dataset_size: int,
    input_prompt: str,
    eval_feature_indices: list[int],
    sae: jumprelu_sae.JumpReluSAE,
) -> list[dict]:
    """Every prompt is exactly the same - the only difference is the steering vectors."""
    cur = 0

    input_messages = [{"role": "user", "content": input_prompt}]

    input_prompt_ids: list[int] = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    input_prompt_ids = torch.tensor(input_prompt_ids, dtype=torch.long).to(device)
    labels = input_prompt_ids.clone()
    attn_mask = torch.ones_like(input_prompt_ids, dtype=torch.bool).to(device)

    x_token_id = tokenizer.encode("X", add_special_tokens=False)[0]

    eval_data = []

    while cur < len(eval_feature_indices):
        batch_steering_vectors = []
        batch_feature_indices = []
        batch_positions = []
        batch_tokens = []
        batch_labels = []
        batch_attn_masks = []

        for i in range(cfg.batch_size):
            if cur >= dataset_size:
                break

            target_feature_idx = eval_feature_indices[cur]
            batch_feature_indices.append(target_feature_idx)

            # 2. Prepare feature vectors for steering
            # We use decoder weights (W_dec) as they map from the feature space back to the residual stream.
            all_feature_indices = [target_feature_idx]

            if cfg.use_decoder_vectors:
                feature_vectors = [sae.W_dec[i] for i in all_feature_indices]
            else:
                feature_vectors = [sae.W_enc[:, i] for i in all_feature_indices]

            batch_steering_vectors.append(feature_vectors)
            batch_tokens.append(input_prompt_ids)
            batch_labels.append(labels)
            batch_attn_masks.append(attn_mask)

            positions = []
            for i in range(len(input_prompt_ids)):
                if input_prompt_ids[i] == x_token_id:
                    positions.append(i)

            assert len(positions) == 1, "Expected exactly one X token"
            batch_positions.append(positions)

            cur += 1

        input_ids = torch.stack(batch_tokens)
        labels = torch.stack(batch_labels)
        attn_mask = torch.stack(batch_attn_masks)

        first_position = batch_positions[0][0]

        for i in range(len(batch_positions)):
            assert batch_positions[i][0] == first_position, (
                "Expected all positions to be the same"
            )

        eval_batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn_mask,
            "steering_vectors": batch_steering_vectors,
            "positions": batch_positions,
            "feature_indices": batch_feature_indices,
        }

        eval_data.append(eval_batch)

    return eval_data


# %%


orig_input_ids, orig_positions = build_training_prompt(tokenizer, device)
orig_input_ids = orig_input_ids.squeeze()
train_eval_prompt = tokenizer.decode(orig_input_ids)

training_data = construct_train_dataset(
    cfg, len(training_examples), train_eval_prompt, training_examples, sae
)

eval_data = construct_eval_dataset(
    cfg, cfg.eval_set_size, train_eval_prompt, cfg.eval_features, sae
)

# %%


def train_batch(
    cfg: SelfInterpTrainingConfig,
    training_batch: dict,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Trains the model on a single batch of data.
    """

    batch_steering_vectors = training_batch["steering_vectors"]
    batch_positions = training_batch["positions"]

    # 3. Create and apply the activation steering hook
    hook_fn = get_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=cfg.steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": training_batch["input_ids"],
        "attention_mask": training_batch["attention_mask"],
    }

    with add_hook(submodule, hook_fn):
        loss = model(**tokenized_input, labels=training_batch["labels"]).loss

    return loss


def eval_batch(
    cfg: SelfInterpTrainingConfig,
    eval_batch: dict,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    sae: jumprelu_sae.JumpReluSAE,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    batch_steering_vectors = eval_batch["steering_vectors"]
    batch_positions = eval_batch["positions"]

    # 3. Create and apply the activation steering hook
    hook_fn = get_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=cfg.steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": eval_batch["input_ids"],
        "attention_mask": eval_batch["attention_mask"],
    }

    with add_hook(submodule, hook_fn):
        output_ids = model.generate(**tokenized_input, **cfg.generation_kwargs)

    # Decode only the newly generated tokens
    generated_tokens = output_ids[:, eval_batch["input_ids"].shape[1] :]
    decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    pos_statements = []
    neg_statements = []
    explanations = []

    for i, output in enumerate(decoded_output):
        # Note: we add the "Positive example:" prefix to the positive sentence
        # because we prefill the model's response with "Positive example:"
        parsed_result = parse_generated_explanation(f"Positive example:{output}")
        print(f"Parsed result: {parsed_result}")
        if parsed_result:
            pos_statements.append(parsed_result["positive_sentence"])
            neg_statements.append(parsed_result["negative_sentence"])
            explanations.append(parsed_result["explanation"])
        else:
            pos_statements.append("")
            neg_statements.append("")
            explanations.append("")

    pos_activations_BLK = get_feature_activations(
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        sae=sae,
        input_strs=pos_statements,
        feature_indices=eval_batch["feature_indices"],
    )

    neg_activations_BLK = get_feature_activations(
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        sae=sae,
        input_strs=neg_statements,
        feature_indices=eval_batch["feature_indices"],
    )

    feature_results = []

    for i, output in enumerate(decoded_output):
        feature_result = {
            "feature_idx": eval_batch["feature_indices"][i],
            "api_prompt": train_eval_prompt,
            "api_response": output,
            "explanation": explanations[i],
            "sentence_data": [],
            "sentence_metrics": [],
        }

        pos_activations_L = pos_activations_BLK[i, :, i]
        neg_activations_L = neg_activations_BLK[i, :, i]

        sentence_distance = lev.normalized_distance(
            pos_statements[i], neg_statements[i]
        )

        original_max_activation = pos_activations_L.max().item()
        rewritten_max_activation = neg_activations_L.max().item()
        original_mean_activation = pos_activations_L.mean().item()
        rewritten_mean_activation = neg_activations_L.mean().item()

        max_activation_ratio = rewritten_max_activation / (
            original_max_activation + 1e-6
        )
        mean_activation_ratio = rewritten_mean_activation / (
            original_mean_activation + 1e-6
        )

        max_activation_ratio = max(max_activation_ratio, 0.0)
        mean_activation_ratio = max(mean_activation_ratio, 0.0)

        max_activation_ratio = min(max_activation_ratio, 1.0)
        mean_activation_ratio = min(mean_activation_ratio, 1.0)

        sentence_data = {
            "sentence_index": 0,
            "original_sentence": pos_statements[i],
            "rewritten_sentence": neg_statements[i],
        }

        sentence_metrics = {
            "original_max_activation": original_max_activation,
            "rewritten_max_activation": rewritten_max_activation,
            "original_mean_activation": original_mean_activation,
            "rewritten_mean_activation": rewritten_mean_activation,
            "max_activation_ratio": max_activation_ratio,
            "mean_activation_ratio": mean_activation_ratio,
            "sentence_distance": sentence_distance,
        }

        feature_result["sentence_data"].append(sentence_data)
        feature_result["sentence_metrics"].append(sentence_metrics)
        feature_results.append(feature_result)

    return feature_results


def train_model(
    cfg: SelfInterpTrainingConfig,
    training_data: list[dict],
    eval_data: list[dict],
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    sae: jumprelu_sae.JumpReluSAE,
    device: torch.device,
    dtype: torch.dtype,
):
    num_epochs = 1
    lr = 1e-5
    max_grad_norm = 1.0
    eval_interval = 50
    wandb_project = "sae_introspection"
    run_name = f"{cfg.model_name}-layer{cfg.sae_layer}-itcp"

    wandb.init(project=wandb_project, name=run_name, config=asdict(cfg))

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    global_step = 0

    for epoch in range(num_epochs):
        pbar = tqdm(training_data, desc=f"Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            loss = train_batch(cfg, batch, model, submodule, device, dtype)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            wandb.log({"train/loss": loss.item(), "step": global_step})

            # -------------------------------- evaluation --------------------------------
            if global_step % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    all_sentence_metrics = []
                    for e_batch in eval_data:
                        feature_results = eval_batch(
                            cfg, e_batch, model, submodule, sae, device, dtype
                        )
                        for res in feature_results:
                            all_sentence_metrics.extend(res["sentence_metrics"])

                    if all_sentence_metrics:
                        aggregated_metrics = {}
                        metric_keys = all_sentence_metrics[0].keys()
                        for key in metric_keys:
                            avg_val = sum(m[key] for m in all_sentence_metrics) / len(
                                all_sentence_metrics
                            )
                            aggregated_metrics[key] = avg_val

                        wandb_log_dict = {
                            f"eval/{k}": v for k, v in aggregated_metrics.items()
                        }
                        wandb.log(wandb_log_dict, step=global_step)

                model.train()
            global_step += 1

    print("Training complete.")
    wandb.finish()
