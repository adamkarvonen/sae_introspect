# %%
import json
from abc import ABC, abstractmethod

import einops
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


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
        tolerance = 1e-2 if self.W_dec.dtype in [torch.bfloat16, torch.float16] else 1e-5

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
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=dtype, device=device))

    def encode(self, x: torch.Tensor):
        """Note: x can be either shape (B, F) or (B, L, F)"""
        post_relu_feat_acts_BF = nn.functional.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

        if self.use_threshold:
            if self.threshold < 0:
                raise ValueError("Threshold is not set. The threshold must be set to use it during inference")
            encoded_acts_BF = post_relu_feat_acts_BF * (post_relu_feat_acts_BF > self.threshold)
            return encoded_acts_BF

        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)
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


# %%

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Qwen/Qwen3-8B"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_submodule(model: AutoModelForCausalLM, layer: int):
    """Gets the residual stream submodule"""
    model_name = model.config._name_or_path

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    elif "gemma" in model_name or "mistral" in model_name or "Qwen" in model_name:
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")


chosen_layers = [9]
sae_repo = "adamkarvonen/qwen3-8b-saes"
sae_path = "saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_9/trainer_2/ae.pt"

sae = load_dictionary_learning_batch_topk_sae(
    repo_id=sae_repo,
    filename=sae_path,
    model_name=model_name,
    device=device,
    dtype=dtype,
    layer=chosen_layers[0],
    local_dir="downloaded_saes",
)

# %%
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")

submodules = [get_submodule(model, chosen_layers[0])]


# %%
class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""

    pass


@torch.no_grad()
def collect_activations(model, submodule, inputs_BL):
    """
    Registers a forward hook on the submodule to capture the residual (or hidden)
    activations. We then raise an EarlyStopException to skip unneeded computations.
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

    try:
        _ = model(input_ids=inputs_BL.to(model.device))
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        handle.remove()

    return activations_BLD


# %%
test_input = "Can you continue this sentence? The scientist named the population, after their distinctive horn, Ovid’s Unicorn. These four-horned, silver-white unicorns were previously unknown to science"

test_input = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": test_input},
    ],
    tokenize=False,
)

tokens = tokenizer(test_input, return_tensors="pt", add_special_tokens=True).to(device)["input_ids"]
# tokens = tokenizer(test_input, return_tensors="pt", add_special_tokens=False).to(device)

print(tokens.shape)

activations_BLD = collect_activations(model, submodules[0], tokens)

print(tokens)

for i in range(tokens.shape[0]):
    decoded_token = tokenizer.decode(tokens[i])
    print(decoded_token)

# %%
norms_BL = activations_BLD.norm(dim=-1)
print(norms_BL)
print(norms_BL.mean())


# VERY IMPORTANT NOTE: The Qwen SAEs were trained with any activations with a norm > median_norm * 10 being filtered out. Thus, as we can see below, the MSE is very high on the first BOS token.

# %%
encoded_BLF = sae.encode(activations_BLD)
decoded_BLD = sae.decode(encoded_BLF)

torch.set_printoptions(precision=8, sci_mode=False)

nonzero_BL = einops.reduce((encoded_BLF > 0).float(), "b l f -> b l", "sum")
print(nonzero_BL)
mean_nonzero = nonzero_BL.mean()
print(mean_nonzero, "\n\n")

MSE_BL = (activations_BLD - decoded_BLD).pow(2).mean(dim=-1)
print(MSE_BL)
mean_MSE = MSE_BL.mean()
print(mean_MSE)

# %%
median_norm = norms_BL.median()
norm_mask_BL = norms_BL < median_norm * 10
encoded_BLF_filtered = encoded_BLF * norm_mask_BL[:, :, None]
decoded_BLD_filtered = sae.decode(encoded_BLF_filtered)

MSE_BL_filtered = (activations_BLD - decoded_BLD_filtered).pow(2).mean(dim=-1)
print(MSE_BL_filtered)
mean_MSE_filtered = MSE_BL_filtered.mean()
print(mean_MSE_filtered)

# %%
