import torch
import einops
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, asdict
import os
import itertools

import interp_tools.model_utils as model_utils
import interp_tools.interp_utils as interp_utils
import interp_tools.saes.jumprelu_sae as jumprelu_sae


def upload_acts_to_hf():
    from huggingface_hub import HfApi

    api = HfApi()

    filename = "max_acts/acts_google_gemma-2-9b-it_layer_9_trainer_16_layer_percent_25_context_length_32.pt"

    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo="acts_google_gemma-2-9b-it_layer_9_trainer_16_layer_percent_25_context_length_32.pt",
        repo_type="dataset",
        repo_id="adamkarvonen/sae_max_acts",
    )


@dataclass
class MaxActsConfig:
    """Configuration settings for the script."""

    # --- Foundational Settings ---
    model_name: str = "google/gemma-2-9b-it"

    # --- SAE (Sparse Autoencoder) Settings ---
    sae_repo_id: str = "google/gemma-scope-9b-it-res"
    sae_layer: int = 9
    sae_width: int = 131  # For loading the correct max acts file
    sae_filename: str = f"layer_{sae_layer}/width_{sae_width}k/average_l0_121/params.npz"
    layer_percent: int = 25  # For loading the correct max acts file

    # --- Experiment Settings ---
    context_length: int = 32
    num_tokens: int = 60_000_000
    batch_size: int = 128


def load_sae_and_model(
    cfg: MaxActsConfig, device: torch.device, dtype: torch.dtype
) -> tuple[AutoModelForCausalLM, AutoTokenizer, jumprelu_sae.JumpReluSAE]:
    """Loads the model, tokenizer, and SAE from Hugging Face."""
    print(f"Loading model: {cfg.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, device_map="auto", torch_dtype=dtype
    )
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


cfg = MaxActsConfig()
device = torch.device("cuda")
dtype = torch.bfloat16

model, tokenizer, sae = load_sae_and_model(cfg, device, dtype)


gradient_checkpointing = False
if gradient_checkpointing:
    model.config.use_cache = False
    model.gradient_checkpointing_enable()


acts_folder = "max_acts"
os.makedirs(acts_folder, exist_ok=True)

submodules = [model_utils.get_submodule(model, cfg.sae_layer)]

acts_filename = os.path.join(
    acts_folder,
    f"acts_{cfg.model_name}_layer_{cfg.sae_layer}_trainer_{cfg.sae_width}_layer_percent_{cfg.layer_percent}_context_length_{cfg.context_length}.pt".replace(
        "/", "_"
    ),
)

if not os.path.exists(acts_filename):
    max_tokens, max_acts = interp_utils.get_interp_prompts(
        model,
        submodules[0],
        sae,
        torch.tensor(list(range(sae.W_dec.shape[0]))),
        context_length=cfg.context_length,
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        num_tokens=cfg.num_tokens,
    )
    acts_data = {
        "max_tokens": max_tokens,
        "max_acts": max_acts,
        "cfg": asdict(cfg),
    }
    torch.save(acts_data, acts_filename)
