#!/usr/bin/env python3
"""
Script to find similar SAE features and verify hard negatives using the Gemma 9B model.

This script:
1. Loads a SAE and finds the most similar features to a target feature
2. Loads the Gemma 9B model and computes actual SAE activations on sentences
3. Identifies sentences from similar features that don't activate for the target feature (hard negatives)
4. Outputs results to JSONL format

Usage as a module:
    from compare_and_verify_hard_negatives import main
    main(target_features=[0, 1, 2], num_sentences=5, top_k_similar_features=10, batch_size=20)

Or modify the call at the bottom of this file and run directly:
    python compare_and_verify_hard_negatives.py
"""

from dataclasses import dataclass
import os
import torch
import torch.nn.functional as F
from typing import List, Sequence, Tuple, Optional
from pydantic import BaseModel
from slist import Slist

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM

# Copied from interp_tools to make file standalone
import contextlib
from abc import ABC, abstractmethod
import numpy as np
from huggingface_hub import hf_hub_download


# Configuration and SAE classes
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


# Configuration variables - no longer need a config class


# SAE Classes
class BaseSAE(torch.nn.Module, ABC):
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
        self.W_enc = torch.nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_in))

        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))

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


class JumpReluSAE(BaseSAE):
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
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        self.threshold = torch.nn.Parameter(torch.zeros(d_sae, dtype=dtype, device=device))
        self.d_sae = d_sae
        self.d_in = d_in

    def encode(self, x: torch.Tensor):
        pre_acts = x @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, feature_acts: torch.Tensor):
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon


def load_gemma_scope_jumprelu_sae(
    repo_id: str,
    filename: str,
    layer: int,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    local_dir: str = "downloaded_saes",
) -> JumpReluSAE:
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )
    pytorch_path = path_to_params.replace(".npz", ".pt")

    # Doing this because npz files are often insanely slow to load
    if not os.path.exists(pytorch_path):
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v) for k, v in params.items()}

        torch.save(pt_params, pytorch_path)

    pt_params = torch.load(pytorch_path)

    d_in = pt_params["W_enc"].shape[0]
    d_sae = pt_params["W_enc"].shape[1]

    assert d_sae >= d_in

    sae = JumpReluSAE(d_in, d_sae, model_name, layer, device, dtype)
    sae.load_state_dict(pt_params)
    sae.to(dtype=dtype, device=device)

    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError(
            "Decoder norms are not normalized. Implement a normalization method."
        )

    return sae


def load_sae(
    sae_repo_id: str,
    sae_filename: str,
    sae_layer: int,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> BaseSAE:
    print(f"Loading SAE for layer {sae_layer} from {sae_repo_id}...")

    if sae_repo_id == "google/gemma-scope-9b-it-res":
        sae = load_gemma_scope_jumprelu_sae(
            repo_id=sae_repo_id,
            filename=sae_filename,
            layer=sae_layer,
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown SAE repo ID: {sae_repo_id}")

    return sae


# Model utilities
def get_submodule(model: AutoModelForCausalLM, layer: int, use_lora: bool = False):
    """Gets the residual stream submodule"""
    model_name = model.config._name_or_path

    if use_lora:
        if "pythia" in model_name:
            raise ValueError("Need to determine how to get submodule for LoRA")
        elif "gemma" in model_name or "mistral" in model_name or "Llama" in model_name:
            return model.base_model.model.model.layers[layer]
        else:
            raise ValueError(f"Please add submodule for model {model_name}")

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    elif "gemma" in model_name or "mistral" in model_name or "Llama" in model_name:
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")


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

# Pydantic schema classes for JSONL output
class TokenActivation(BaseModel):
    as_str: str
    activation: float
    token_id: int

    def to_prompt_str(self) -> str:
        return f"{self.as_str} ({self.activation:.2f})"


class SentenceInfo(BaseModel):
    max_activation: float
    tokens: List[TokenActivation]
    as_str: str

    def as_activation_vector(self) -> str:
        activation_vector = Slist(self.tokens).map(lambda x: x.to_prompt_str())
        return f"{activation_vector}"


class SAEActivations(BaseModel):
    sae_id: int
    sentences: List[SentenceInfo]


class SAE(BaseModel):
    sae_id: int
    feature_vector: Sequence[float]
    activations: SAEActivations
    # Sentences that do not activate for the given sae_id. But come from a similar SAE
    # Here the sae_id correspond to different similar SAEs.
    # The activations are the activations w.r.t this SAE. And should be low.
    hard_negatives: List[SAEActivations]


def load_max_acts_data(
    model_name: str,
    sae_layer: int,
    sae_width: int,
    layer_percent: int,
    context_length: int = 32,
) -> dict[str, torch.Tensor]:
    """Load the max activating examples data."""
    acts_dir = "max_acts"

    # Construct filename
    acts_filename = f"acts_{model_name}_layer_{sae_layer}_trainer_{sae_width}_layer_percent_{layer_percent}_context_length_{context_length}.pt".replace(
        "/", "_"
    )

    acts_path = os.path.join(acts_dir, acts_filename)

    # Download if not exists
    if not os.path.exists(acts_path):
        print(f"📥 Downloading max acts data: {acts_filename}")
        try:
            path_to_config = hf_hub_download(
                repo_id="adamkarvonen/sae_max_acts",
                filename=acts_filename,
                force_download=False,
                local_dir=acts_dir,
                repo_type="dataset",
            )
            print(f"✅ Downloaded to: {acts_path}")
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            raise

    print(f"📂 Loading max acts data from: {acts_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acts_data = torch.load(acts_path, map_location=device)

    return acts_data


def decode_tokens_to_sentences(
    tokens: torch.Tensor, tokenizer: AutoTokenizer, skip_bos: bool = True
) -> List[str]:
    """Convert token tensors to readable sentences using batch decoding."""
    # Skip BOS token if requested
    if skip_bos and tokens.shape[1] > 0:
        tokens = tokens[:, 1:]  # Remove first token from all sequences
    
    # Convert to list format for batch decoding
    token_lists = tokens.tolist()
    
    # Batch decode all sequences at once
    sentences = tokenizer.batch_decode(
        token_lists, skip_special_tokens=True
    )
    
    # Strip whitespace from each sentence
    sentences = [sentence.strip() for sentence in sentences]
    
    return sentences


@dataclass(kw_only=True)
class FeatureMaxActivations:
    """Contains the maximum activating sentences and data for a feature."""
    sentences: List[str]
    activations: torch.Tensor  # Shape: [num_sentences, seq_len]
    tokens: torch.Tensor  # Shape: [num_sentences, seq_len]


@dataclass(kw_only=True)
class SimilarFeature:
    """Represents a feature similar to a target feature."""
    feature_idx: int
    similarity_score: float


def get_feature_max_activating_sentences(
    acts_data: dict[str, torch.Tensor],
    tokenizer: AutoTokenizer,
    feature_idx: int,
    num_sentences: int = 5,
) -> FeatureMaxActivations:
    """
    Get the top maximally activating sentences for a specific feature.

    Returns:
        FeatureMaxActivations containing sentences, activations, and tokens
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

    # Decode tokens to sentences
    sentences = decode_tokens_to_sentences(feature_tokens, tokenizer)

    return FeatureMaxActivations(
        sentences=sentences,
        activations=feature_activations,
        tokens=feature_tokens
    )


def find_most_similar_features(
    sae, target_feature_idx: int, top_k: int = 1, exclude_self: bool = True
) -> List[SimilarFeature]:
    """Find the most similar features to a target feature using cosine similarity of encoder vectors."""
    # Get encoder weights - shape: [d_in, d_sae]
    W_enc = sae.W_enc.data  # Remove gradient tracking

    # Get the target feature vector - shape: [d_in]
    target_vector = W_enc[:, target_feature_idx]

    # Compute cosine similarity with all other features
    # Normalize the target vector
    target_normalized = F.normalize(target_vector.unsqueeze(0), dim=1)

    # Normalize all encoder vectors
    all_vectors_normalized = F.normalize(W_enc.T, dim=1)  # Shape: [d_sae, d_in]

    # Compute cosine similarities - shape: [d_sae]
    similarities = torch.mm(all_vectors_normalized, target_normalized.T).squeeze()

    if exclude_self:
        # Set similarity to target feature itself to -inf so it's not selected
        similarities[target_feature_idx] = float("-inf")

    # Get top-k most similar features
    top_similarities, top_indices = torch.topk(similarities, k=top_k, largest=True)

    # Create SimilarFeature objects
    similar_features = []
    for sim_score, feature_idx in zip(top_similarities, top_indices):
        similar_features.append(SimilarFeature(
            feature_idx=int(feature_idx.item()),
            similarity_score=float(sim_score.item())
        ))

    return similar_features


def load_model_and_sae(
    model_name: str,
    sae_repo_id: str, 
    sae_filename: str,
    sae_layer: int,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, object, torch.nn.Module]:
    """Load the Gemma 9B model, tokenizer, SAE, and submodule."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    print("🧠 Loading Gemma 9B model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Load SAE
    print("🔧 Loading SAE...")
    sae = load_sae(sae_repo_id, sae_filename, sae_layer, model_name, device, dtype)

    # Get submodule for activation collection
    submodule = get_submodule(model, sae_layer)

    return model, tokenizer, sae, submodule


def compute_sae_activations_for_sentences(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae: object,
    submodule: torch.nn.Module,
    sentences: List[str],
    target_feature_idx: int,
    batch_size: int = 8,
) -> List[SentenceInfo]:
    """
    Compute SAE activations for a list of sentences and return SentenceInfo objects.
    """
    sentence_infos = []

    # Process sentences in batches
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        
        for sentence in batch_sentences:
            # Tokenize sentence
            tokenized = tokenizer(
                sentence,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=512,
            ).to(model.device)

            with torch.no_grad():
                # Get model activations at the SAE layer
                layer_acts_BLD = collect_activations(
                    model, submodule, tokenized
                )

                # Encode through SAE
                encoded_acts_BLF = sae.encode(layer_acts_BLD)

                # Get activations for the target feature - shape: [batch, seq_len]
                feature_acts = encoded_acts_BLF[0, :, target_feature_idx]  # [seq_len]

                # Convert to token activations
                token_activations = []
                token_ids = tokenized["input_ids"][0]  # [seq_len]

                for i, (token_id, activation) in enumerate(zip(token_ids, feature_acts)):
                    token_str = tokenizer.decode(
                        [token_id.item()], skip_special_tokens=True
                    )
                    token_activations.append(
                        TokenActivation(
                            as_str=token_str,
                            activation=activation.item(),
                            token_id=token_id.item(),
                        )
                    )

                # Create SentenceInfo
                max_activation = feature_acts.max().item()
                sentence_info = SentenceInfo(
                    max_activation=max_activation, tokens=token_activations, as_str=sentence
                )

                sentence_infos.append(sentence_info)

    return sentence_infos


def identify_hard_negatives(
    similar_sentence_infos: List[SentenceInfo],
    threshold: float = 0.5,
    batch_size: int = 8,
) -> List[SentenceInfo]:
    """
    Identify sentences that have low activation for the target feature.
    These are hard negatives - sentences from similar features that don't activate the target.
    """
    hard_negatives = []

    for sentence_info in similar_sentence_infos:
        if sentence_info.max_activation < threshold:
            hard_negatives.append(sentence_info)

    return hard_negatives


def main(
    target_features: Sequence[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    num_sentences: int = 20,
    top_k_similar_features: int = 10,
    output: str = "hard_negatives_results.jsonl",
    model_name: str = "google/gemma-2-9b-it",
    sae_repo_id: str = "google/gemma-scope-9b-it-res",
    context_length: int = 32,
    hard_negative_threshold: float = 0.5,
    batch_size: int = 20,
):

    # Get SAE info
    sae_width, sae_layer, sae_layer_percent, sae_filename = get_sae_info(sae_repo_id)

    print("🔧 Configuration:")
    print(f"   Model: {model_name}")
    print(f"   SAE: {sae_repo_id}")
    print(f"   Layer: {sae_layer}")
    print(f"   Width: {sae_width}")
    print(f"   Target Features: {target_features}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Output: {output}")

    # Load max acts data
    print("📊 Loading max acts data...")
    acts_data = load_max_acts_data(
        model_name,
        sae_layer,
        sae_width,
        sae_layer_percent,
        context_length,
    )

    # Validate feature indices
    max_feature_idx = acts_data["max_tokens"].shape[0] - 1
    for feature_idx in target_features:
        if feature_idx > max_feature_idx:
            raise ValueError(
                f"Feature {feature_idx} not found. Max feature index: {max_feature_idx}"
            )

    # Load model, tokenizer, and SAE
    print("🚀 Loading model and SAE...")
    model, tokenizer, sae, submodule = load_model_and_sae(
        model_name, sae_repo_id, sae_filename, sae_layer
    )

    # Process each feature index
    all_results = []
    
    for feature_idx in target_features:
        print(f"\n🎯 Processing feature {feature_idx}...")
        
        # Find most similar features
        print(
            f"🔍 Finding {top_k_similar_features} most similar features to feature {feature_idx}..."
        )
        similar_features = find_most_similar_features(
            sae, feature_idx, top_k=top_k_similar_features
        )

        # Get sentences for target feature
        print(f"📝 Getting sentences for target feature {feature_idx}...")
        target_max_acts: FeatureMaxActivations = get_feature_max_activating_sentences(
            acts_data, tokenizer, feature_idx, num_sentences
        )
        target_sentences = target_max_acts.sentences

        # Compute actual SAE activations for target feature sentences
        print("🧮 Computing SAE activations for target feature sentences...")
        target_sentence_infos = compute_sae_activations_for_sentences(
            model, tokenizer, sae, submodule, target_sentences, feature_idx, batch_size
        )

        # Analyze similar features and collect hard negatives
        hard_negatives_list = []

        for similar_feature in similar_features:
            print(
                f"📝 Analyzing similar feature {similar_feature.feature_idx} (similarity: {similar_feature.similarity_score:.4f})..."
            )

            # Get sentences for this similar feature
            similar_max_acts = get_feature_max_activating_sentences(
                acts_data, tokenizer, similar_feature.feature_idx, num_sentences
            )
            similar_sentences = similar_max_acts.sentences

            # Compute target feature activations on similar feature's sentences
            similar_sentence_infos = compute_sae_activations_for_sentences(
                model, tokenizer, sae, submodule, similar_sentences, feature_idx, batch_size
            )

            # Identify hard negatives
            hard_negatives = identify_hard_negatives(
                similar_sentence_infos, hard_negative_threshold, batch_size
            )

            if hard_negatives:
                hard_negatives_sae = SAEActivations(
                    sae_id=similar_feature.feature_idx, sentences=hard_negatives
                )
                hard_negatives_list.append(hard_negatives_sae)
                print(
                    f"   Found {len(hard_negatives)} hard negatives from feature {similar_feature.feature_idx}"
                )
            else:
                print(f"   No hard negatives found from feature {similar_feature.feature_idx}")

        # Create final SAE object for this feature
        target_activations = SAEActivations(
            sae_id=feature_idx, sentences=target_sentence_infos
        )

        # Extract feature vector from SAE decoder weights
        feature_vector = sae.W_dec[feature_idx].cpu().tolist()
        
        sae_result = SAE(
            sae_id=feature_idx,
            feature_vector=feature_vector,
            activations=target_activations,
            hard_negatives=hard_negatives_list,
        )
        
        all_results.append(sae_result)
        
        print(f"✅ Feature {feature_idx} complete!")
        print(f"   Target sentences analyzed: {len(target_sentence_infos)}")
        print(f"   Similar features analyzed: {len(similar_features)}")
        print(f"   Hard negative groups found: {len(hard_negatives_list)}")

    # Write all results to JSONL
    print(f"\n💾 Writing results to {output}...")
    with open(output, "w") as f:
        for result in all_results:
            f.write(result.model_dump_json() + "\n")

    print("✅ Analysis complete!")
    print(f"   Features processed: {target_features}")
    print(f"   Total results: {len(all_results)}")
    print(f"   Results saved to: {output}")


if __name__ == "__main__":
    # Example usage - customize the feature_idxs and other parameters as needed
    main(target_features=[0,1,2,3,4,5,6,7,8,9])
