import contextlib
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import einops
import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""

    pass


def collect_activations(
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    inputs_BL: dict[str, torch.Tensor],
    use_no_grad: bool = True,
) -> torch.Tensor:
    """
    Registers a forward hook on the submodule to capture the residual (or hidden)
    activations. We then raise an EarlyStopException to skip unneeded computations.

    Args:
        model: The model to run.
        submodule: The submodule to hook into.
        inputs_BL: The inputs to the model.
        use_no_grad: Whether to run the forward pass within a `torch.no_grad()` context. Defaults to True.
    """
    activations_BLD = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal activations_BLD
        # For many models, the submodule outputs are a tuple or a single tensor:
        # If "outputs" is a tuple, pick the relevant item:
        #   e.g. if your layer returns (hidden, something_else), you'd do outputs[0]
        # Otherwise just do outputs
        if isinstance(outputs, tuple):
            activations_BLD = outputs[0]
        else:
            activations_BLD = outputs

        raise EarlyStopException("Early stopping after capturing activations")

    handle = submodule.register_forward_hook(gather_target_act_hook)

    # Determine the context manager based on the flag
    context_manager = torch.no_grad() if use_no_grad else contextlib.nullcontext()

    try:
        # Use the selected context manager
        with context_manager:
            _ = model(**inputs_BL)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        handle.remove()

    return activations_BLD


def get_sae_info(sae_repo_id: str, sae_layer_percent: int) -> tuple[int, int, str]:
    if sae_repo_id == "google/gemma-scope-9b-it-res":
        assert sae_layer_percent == 25
        sae_layer = 9

        sae_width = 131

        if sae_width == 16:
            sae_filename = f"layer_{sae_layer}/width_16k/average_l0_88/params.npz"
        elif sae_width == 131:
            sae_filename = f"layer_{sae_layer}/width_131k/average_l0_121/params.npz"
        else:
            raise ValueError(f"Unknown SAE width: {sae_width}")
    elif sae_repo_id == "fnlp/Llama3_1-8B-Base-LXR-32x":
        assert sae_layer_percent == 25
        sae_layer = 9
        sae_width = 32
        sae_filename = ""
    elif sae_repo_id == "adamkarvonen/qwen3-8b-saes":
        sae_layer = int(36 * (sae_layer_percent / 100))

        assert sae_layer in [9, 18, 27]

        sae_width = 2
        sae_filename = f"saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_{sae_layer}/trainer_{sae_width}/ae.pt"
    else:
        raise ValueError(f"Unknown SAE repo ID: {sae_repo_id}")
    return sae_width, sae_layer, sae_filename


class BaseSAE(nn.Module, ABC):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        super().__init__()

        # Required parameters
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))

        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Required attributes
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.hook_layer = hook_layer

        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        self.to(dtype=self.dtype, device=self.device)

    @abstractmethod
    def encode(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    @abstractmethod
    def decode(self, feature_acts: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    def to(self, *args, **kwargs):
        """Handle device and dtype updates"""
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self

    @torch.no_grad()
    def check_decoder_norms(self) -> bool:
        """
        It's important to check that the decoder weights are normalized.
        """
        norms = torch.norm(self.W_dec, dim=1).to(dtype=self.dtype, device=self.device)

        # In bfloat16, it's common to see errors of (1/256) in the norms
        tolerance = (
            1e-2 if self.W_dec.dtype in [torch.bfloat16, torch.float16] else 1e-5
        )

        if torch.allclose(norms, torch.ones_like(norms), atol=tolerance):
            return True
        else:
            max_diff = torch.max(torch.abs(norms - torch.ones_like(norms)))
            print(f"Decoder weights are not normalized. Max diff: {max_diff.item()}")
            return False


class BatchTopKSAE(BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        assert isinstance(k, int) and k > 0
        self.register_buffer("k", torch.tensor(k, dtype=torch.int, device=device))

        # BatchTopK requires a global threshold to use during inference. Must be positive.
        self.use_threshold = True
        self.register_buffer(
            "threshold", torch.tensor(-1.0, dtype=dtype, device=device)
        )

    def encode(self, x: torch.Tensor):
        """Note: x can be either shape (B, F) or (B, L, F)"""
        post_relu_feat_acts_BF = nn.functional.relu(
            (x - self.b_dec) @ self.W_enc + self.b_enc
        )

        if self.use_threshold:
            if self.threshold < 0:
                raise ValueError(
                    "Threshold is not set. The threshold must be set to use it during inference"
                )
            encoded_acts_BF = post_relu_feat_acts_BF * (
                post_relu_feat_acts_BF > self.threshold
            )
            return encoded_acts_BF

        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=tops_acts_BK
        )
        return encoded_acts_BF

    def decode(self, feature_acts: torch.Tensor):
        return (feature_acts @ self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon


def load_dictionary_learning_batch_topk_sae(
    repo_id: str,
    filename: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: int | None = None,
    local_dir: str = "downloaded_saes",
) -> BatchTopKSAE:
    assert "ae.pt" in filename

    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )

    pt_params = torch.load(path_to_params, map_location=torch.device("cpu"))

    config_filename = filename.replace("ae.pt", "config.json")
    path_to_config = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        force_download=False,
        local_dir=local_dir,
    )

    with open(path_to_config) as f:
        config = json.load(f)

    if layer is not None:
        assert layer == config["trainer"]["layer"]
    else:
        layer = config["trainer"]["layer"]

    # Transformer lens often uses a shortened model name
    assert model_name in config["trainer"]["lm_name"]

    k = config["trainer"]["k"]

    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "bias": "b_dec",
        "k": "k",
        "threshold": "threshold",
    }

    # Create a new dictionary with renamed keys
    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

    # due to the way torch uses nn.Linear, we need to transpose the weight matrices
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T

    # Print renamed keys for debugging
    print("Renamed keys in state_dict:", renamed_params.keys())

    sae = BatchTopKSAE(
        d_in=renamed_params["b_dec"].shape[0],
        d_sae=renamed_params["b_enc"].shape[0],
        k=k,
        model_name=model_name,
        hook_layer=layer,  # type: ignore
        device=device,
        dtype=dtype,
    )

    sae.load_state_dict(renamed_params)

    sae.to(device=device, dtype=dtype)

    d_sae, d_in = sae.W_dec.data.shape

    assert d_sae >= d_in

    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError("Decoder vectors are not normalized. Please normalize them")

    return sae


def dataset_to_list_of_strs(
    dataset_name: str, min_row_chars: int, total_chars: int
) -> list[str]:
    """
    Grab text data from a streaming dataset, stopping once we've collected total_chars.
    """
    # Adjust column names depending on dataset
    is_redpajama = dataset_name == "togethercomputer/RedPajama-Data-V2"
    # Example for your 'pile' dataset:
    # is_pile = dataset_name == "monology/pile-uncopyrighted"
    column_name = "raw_content" if is_redpajama else "text"

    dataset = load_dataset(
        dataset_name,
        name="sample-10B" if is_redpajama else None,
        trust_remote_code=True,
        streaming=True,
        split="train",
    )

    total_chars_so_far = 0
    result = []

    for row in dataset:
        text = row[column_name]
        if len(text) > min_row_chars:
            result.append(text)
            total_chars_so_far += len(text)
            if total_chars_so_far > total_chars:
                break
    return result


@torch.no_grad
def get_bos_pad_eos_mask(
    tokens: torch.Tensor, tokenizer: AutoTokenizer | Any
) -> torch.Tensor:
    mask = (
        (tokens == tokenizer.pad_token_id)  # type: ignore
        | (tokens == tokenizer.eos_token_id)  # type: ignore
        | (tokens == tokenizer.bos_token_id)  # type: ignore
    ).to(dtype=torch.bool)
    return ~mask


def tokenize_and_concat_dataset(
    tokenizer,
    dataset: list[str],
    seq_len: int,
    add_bos: bool = True,
    max_tokens: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """
    Concatenate text from the dataset with eos_token between chunks, then tokenize.
    Reshape into (B, seq_len) blocks. Truncates any partial remainder.
    """
    full_text = tokenizer.eos_token.join(dataset)

    # Divide into chunks to speed up tokenization
    num_chunks = 20
    chunk_length = (len(full_text) - 1) // num_chunks + 1
    chunks = [
        full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)
    ]
    all_tokens = []
    for chunk in chunks:
        token_ids = tokenizer(chunk)["input_ids"]
        all_tokens.extend(token_ids)

        # Append EOS token if missing.
        if not chunk.endswith(tokenizer.eos_token):
            chunk += tokenizer.eos_token
        token_ids = tokenizer(chunk)["input_ids"]
        all_tokens.extend(token_ids)

    tokens = torch.tensor(all_tokens)

    if max_tokens is not None:
        tokens = tokens[: max_tokens + seq_len + 1]

    num_tokens = len(tokens)
    num_batches = num_tokens // seq_len

    # Drop last partial batch if not full
    tokens = tokens[: num_batches * seq_len]
    tokens = einops.rearrange(
        tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
    )

    # Overwrite first token in each block with BOS if desired
    if add_bos:
        tokens[:, 0] = tokenizer.bos_token_id

    attention_mask = torch.ones_like(tokens)

    token_dict = {
        "input_ids": tokens,
        "attention_mask": attention_mask,
    }

    return token_dict


def load_and_tokenize_and_concat_dataset(
    dataset_name: str,
    ctx_len: int,
    num_tokens: int,
    tokenizer,
    add_bos: bool = True,
    min_row_chars: int = 100,
) -> dict[str, torch.Tensor]:
    """
    Load text from dataset_name, tokenize it, and return (B, ctx_len) blocks of tokens.
    """
    # For safety, let's over-sample from dataset (like you did with 5x)
    dataset_strs = dataset_to_list_of_strs(dataset_name, min_row_chars, num_tokens * 5)

    token_dict = tokenize_and_concat_dataset(
        tokenizer=tokenizer,
        dataset=dataset_strs,
        seq_len=ctx_len,
        add_bos=add_bos,
        max_tokens=num_tokens,
    )

    # Double-check we have enough tokens
    assert (
        token_dict["input_ids"].shape[0] * token_dict["input_ids"].shape[1]
    ) >= num_tokens, "Not enough tokens found!"
    return token_dict


def get_batched_tokens(
    tokenizer: AutoTokenizer,
    model_name: str,
    dataset_name: str,
    num_tokens: int,
    batch_size: int,
    device: torch.device,
    context_length: int,
    force_rebuild_tokens: bool = False,
    tokens_folder: str = "tokens",
    save_tokens: bool = True,
) -> list[dict[str, torch.Tensor]]:
    # E.g. "tokens/togethercomputer_RedPajama-Data-V2_1000000_google-gemma-2-2b.pt"
    filename = f"{tokens_folder}/{dataset_name.replace('/', '_')}_{num_tokens}_{model_name.replace('/', '_')}.pt"

    # If we haven't built the token file or if user wants to force a rebuild
    if (not os.path.exists(filename)) or force_rebuild_tokens:
        token_dict = load_and_tokenize_and_concat_dataset(
            dataset_name=dataset_name,
            ctx_len=context_length,
            num_tokens=num_tokens,
            tokenizer=tokenizer,
            add_bos=True,
        )
        token_dict = {k: v.cpu() for k, v in token_dict.items()}
        if save_tokens:
            os.makedirs(tokens_folder, exist_ok=True)
            torch.save(token_dict, filename)
            print(f"Saved tokenized dataset to {filename}")
    else:
        print(f"Loading tokenized dataset from {filename}")
        token_dict = torch.load(filename)

    token_dict = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in token_dict.items()
    }

    batched_tokens = []

    for i in range(0, token_dict["input_ids"].shape[0], batch_size):
        batched_tokens.append(
            {
                "input_ids": token_dict["input_ids"][i : i + batch_size],
                "attention_mask": token_dict["attention_mask"][i : i + batch_size],
            }
        )

    return batched_tokens


def get_interp_prompts(
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    sae: BaseSAE,
    dim_indices: torch.Tensor,
    context_length: int,
    tokenizer: AutoTokenizer,
    dataset_name: str = "togethercomputer/RedPajama-Data-V2",
    num_tokens: int = 1_000_000,
    batch_size: int = 32,
    tokens_folder: str = "tokens",
    force_rebuild_tokens: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    1) Loads or builds a tokenized dataset (B, context_length).
    2) Splits into batches of size batch_size.
    3) Runs get_max_activating_prompts(...) to get top-k tokens/activations.
    """
    device = model.device
    model_name = model.config._name_or_path

    batched_tokens = get_batched_tokens(
        tokenizer=tokenizer,
        model_name=model_name,
        dataset_name=dataset_name,
        num_tokens=num_tokens,
        batch_size=batch_size,
        device=device,
        context_length=context_length,
        force_rebuild_tokens=force_rebuild_tokens,
        tokens_folder=tokens_folder,
    )

    # Now get the max-activating prompts for the given dim_indices
    max_tokens_FKL, max_activations_FKL = get_max_activating_prompts(
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        tokenized_inputs_bL=batched_tokens,
        dim_indices=dim_indices,
        batch_size=batch_size,
        dictionary=sae,
        context_length=context_length,
        k=30,  # or pass as a parameter if you want
    )

    return max_tokens_FKL, max_activations_FKL


@torch.no_grad()
def get_max_activating_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    submodule: torch.nn.Module,
    tokenized_inputs_bL: list[dict[str, torch.Tensor]],
    dim_indices: torch.Tensor,
    batch_size: int,
    dictionary: BaseSAE,
    context_length: int,
    k: int = 30,
    zero_bos: bool = True,
    max_act_norm_multiple: float | None = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each feature in dim_indices, find the top-k (prompt, position) with the highest
    dictionary-encoded activation. Return the tokens and the activations for those points.
    """

    device = model.device
    feature_count = dim_indices.shape[0]

    # We'll store results in [F, k] or [F, k, L] shape
    max_activating_indices_FK = torch.zeros(
        (feature_count, k), device=device, dtype=torch.int32
    )
    max_activations_FK = torch.zeros(
        (feature_count, k), device=device, dtype=torch.bfloat16
    )
    max_tokens_FKL = torch.zeros(
        (feature_count, k, context_length), device=device, dtype=torch.int32
    )
    max_activations_FKL = torch.zeros(
        (feature_count, k, context_length), device=device, dtype=torch.bfloat16
    )

    for i, inputs_BL in tqdm(
        enumerate(tokenized_inputs_bL), total=len(tokenized_inputs_bL)
    ):
        batch_offset = i * batch_size
        attention_mask = inputs_BL["attention_mask"]

        # 1) Collect submodule activations
        activations_BLD = collect_activations(model, submodule, inputs_BL)

        # 2) Apply dictionary's encoder
        #    shape: [B, L, D], dictionary.encode -> [B, L, F]
        #    Then keep only the dims in dim_indices
        activations_BLF = dictionary.encode(activations_BLD)
        if zero_bos:
            bos_mask_BL = get_bos_pad_eos_mask(inputs_BL["input_ids"], tokenizer)
            activations_BLF *= bos_mask_BL[:, :, None]

        if max_act_norm_multiple is not None:
            median_norm = activations_BLF.norm(dim=-1).median()
            norm_mask_BL = (
                activations_BLF.norm(dim=-1) < median_norm * max_act_norm_multiple
            )
            activations_BLF *= norm_mask_BL[:, :, None]

        activations_BLF = activations_BLF[:, :, dim_indices]  # shape: [B, L, Fselected]

        activations_BLF = activations_BLF * attention_mask[:, :, None]

        # 3) Move dimension to (F, B, L)
        activations_FBL = einops.rearrange(activations_BLF, "B L F -> F B L")

        # For each sequence, the "peak activation" is the maximum over positions:
        # shape: [F, B]
        activations_FB = einops.reduce(activations_FBL, "F B L -> F B", "max")

        # We'll replicate the tokens to shape [F, B, L]
        tokens_FBL = einops.repeat(
            inputs_BL["input_ids"], "B L -> F B L", F=feature_count
        )

        # Create an index for the batch offset
        indices_B = torch.arange(batch_offset, batch_offset + batch_size, device=device)
        indices_FB = einops.repeat(indices_B, "B -> F B", F=feature_count)

        # Concatenate with previous top-k
        combined_activations_FB = torch.cat([max_activations_FK, activations_FB], dim=1)
        combined_indices_FB = torch.cat([max_activating_indices_FK, indices_FB], dim=1)

        combined_activations_FBL = torch.cat(
            [max_activations_FKL, activations_FBL], dim=1
        )
        combined_tokens_FBL = torch.cat([max_tokens_FKL, tokens_FBL], dim=1)

        # 4) Sort to keep only top-k
        topk_activations_FK, topk_indices_FK = torch.topk(
            combined_activations_FB, k, dim=1
        )

        max_activations_FK = topk_activations_FK
        feature_indices_F1 = torch.arange(feature_count, device=device)[:, None]

        max_activating_indices_FK = combined_indices_FB[
            feature_indices_F1, topk_indices_FK
        ]
        max_activations_FKL = combined_activations_FBL[
            feature_indices_F1, topk_indices_FK
        ]
        max_tokens_FKL = combined_tokens_FBL[feature_indices_F1, topk_indices_FK]

    return max_tokens_FKL, max_activations_FKL


def get_submodule(model: AutoModelForCausalLM, layer: int, use_lora: bool = False):
    """Gets the residual stream submodule"""
    model_name = model.config._name_or_path

    if use_lora:
        if "pythia" in model_name:
            raise ValueError("Need to determine how to get submodule for LoRA")
        elif (
            "gemma" in model_name
            or "mistral" in model_name
            or "Llama" in model_name
            or "Qwen" in model_name
        ):
            return model.base_model.model.model.layers[layer]
        else:
            raise ValueError(f"Please add submodule for model {model_name}")

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    elif (
        "gemma" in model_name
        or "mistral" in model_name
        or "Llama" in model_name
        or "Qwen" in model_name
    ):
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")


def upload_acts_to_hf(filename: str):
    # filename should be in the format of:
    # filename = "max_acts/acts_google_gemma-2-9b-it_layer_9_trainer_16_layer_percent_25_context_length_32.pt"
    path_in_repo = filename.split("/")[-1]

    api = HfApi()

    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=path_in_repo,
        repo_type="dataset",
        repo_id="adamkarvonen/sae_max_acts",
    )


if __name__ == "__main__":
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # cfg.sae_repo_id = "fnlp/Llama3_1-8B-Base-LXR-32x"
    # cfg.model_name = "meta-llama/Llama-3.1-8B-Instruct"

    sae_repo_id = "adamkarvonen/qwen3-8b-saes"
    model_name = "Qwen/Qwen3-8B"

    sae_layer_percents = [50, 75]

    for sae_layer_percent in sae_layer_percents:
        sae_width, sae_layer, sae_filename = get_sae_info(
            sae_repo_id, sae_layer_percent
        )

        num_tokens = 60_000_000
        context_length = 32
        max_acts_batch_size = 128

        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Qwen doesn't have a bos token, so we'll use the eos token
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        sae = load_dictionary_learning_batch_topk_sae(
            repo_id=sae_repo_id,
            filename=sae_filename,
            model_name=model_name,
            device=device,
            dtype=dtype,
        )

        acts_folder = "max_acts"
        os.makedirs(acts_folder, exist_ok=True)

        submodules = [get_submodule(model, sae_layer)]

        acts_filename = os.path.join(
            acts_folder,
            f"acts_{model_name}_layer_{sae_layer}_trainer_{sae_width}_layer_percent_{sae_layer_percent}_context_length_{context_length}.pt".replace(
                "/", "_"
            ),
        )

        if not os.path.exists(acts_filename):
            max_tokens, max_acts = get_interp_prompts(
                model,
                submodules[0],
                sae,
                torch.tensor(list(range(sae.W_dec.shape[0]))),
                context_length=context_length,
                tokenizer=tokenizer,
                batch_size=max_acts_batch_size,
                num_tokens=num_tokens,
            )

            config = {
                "sae_width": sae_width,
                "sae_layer": sae_layer,
                "sae_layer_percent": sae_layer_percent,
                "context_length": context_length,
                "max_acts_batch_size": max_acts_batch_size,
                "num_tokens": num_tokens,
                "model_name": model_name,
                "sae_repo_id": sae_repo_id,
            }

            acts_data = {
                "max_tokens": max_tokens,
                "max_acts": max_acts,
                "config": config,
            }
            torch.save(acts_data, acts_filename)

            upload_acts_to_hf(acts_filename)
