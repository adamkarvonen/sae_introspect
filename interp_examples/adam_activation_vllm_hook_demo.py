# %%
%load_ext autoreload
%autoreload 2


# %%
import os

from detection_eval.steering_hooks import X_PROMPT
os.environ['VLLM_USE_V1'] = '0'
os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch.nn as nn
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Callable
import contextlib
import numpy as np
from huggingface_hub import hf_hub_download
import gc
import torch._dynamo


# %%
# Configuration matching host_vllm_server_hook.py
MODEL_NAME = "google/gemma-2-9b-it"
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
CTX_LEN = 512 * 4
LAYER = 9  # Target layer for activation steering
MAX_DECODE_TOKENS = 100

# SAE Configuration
SAE_REPO_ID = "google/gemma-scope-9b-it-res"
LOAD_LORAS = ["thejaminator/gemma-introspection-20250821"]
SAE_WIDTH = 16  # Can be 16 or 131
SAE_FILENAME = f"layer_{LAYER}/width_16k/average_l0_88/params.npz"
STEERING_COEFFICIENT = 2.0
SAE_INDEX = 10027  # Test with this SAE feature index

print(f"Model: {MODEL_NAME}")
print(f"Target layer: {LAYER}")
print(f"SAE index: {SAE_INDEX}")
print(f"Steering coefficient: {STEERING_COEFFICIENT}")

BATCH_SIZE = 4
SAE_INDICES = [10027, 10028, 10029, 10030]
assert len(SAE_INDICES) == BATCH_SIZE

torch._dynamo.disable()

# %%
# SAE Classes (copied from host_vllm_server_hook.py)
class JumpReluSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae, dtype=dtype, device=device))
        self.device = device
        self.dtype = dtype
        self.d_sae = d_sae
        self.d_in = d_in
        self.to(dtype=dtype, device=device)

    def encode(self, x: torch.Tensor):
        pre_acts = x @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, feature_acts: torch.Tensor):
        return feature_acts @ self.W_dec + self.b_dec


def load_gemma_scope_sae(
    repo_id: str,
    filename: str,
    layer: int,
    device: torch.device,
    dtype: torch.dtype,
    local_dir: str = "downloaded_saes",
) -> JumpReluSAE:
    """Load Gemma Scope SAE from Hugging Face Hub."""
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )
    pytorch_path = path_to_params.replace(".npz", ".pt")

    # Convert npz to pt for faster loading
    if not os.path.exists(pytorch_path):
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
        torch.save(pt_params, pytorch_path)

    pt_params = torch.load(pytorch_path)

    d_in = pt_params["W_enc"].shape[0]
    d_sae = pt_params["W_enc"].shape[1]

    sae = JumpReluSAE(d_in, d_sae, device, dtype)
    sae.load_state_dict(pt_params)
    sae.to(dtype=dtype, device=device)

    return sae


def get_sae_feature_vector(sae_index: int, sae: JumpReluSAE) -> torch.Tensor:
    """
    Get SAE feature vector for the given index from the loaded SAE.
    """
    if sae_index >= sae.d_sae:
        raise ValueError(f"Feature index {sae_index} is out of bounds for SAE with {sae.d_sae} features")

    # Use decoder weights as feature vectors (maps from feature space back to residual stream)
    return sae.W_dec[sae_index].clone()


def find_x_positions(formatted_prompt: str, tokenizer) -> list[int]:
    """Find positions of 'X' tokens in the formatted prompt."""
    positions = []
    tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    for i, token_id in enumerate(tokens):
        if tokenizer.decode([token_id]) == "X":
            positions.append(i)
    return positions

# %%
# Initialize tokenizer
print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create test prompts with 'X' tokens
test_inputs = [
    [{"role": "user", "content": X_PROMPT}],
    # [{"role": "user", "content": "What does the 'X' mean?"}]
] * BATCH_SIZE

# Apply chat template to both prompts
formatted_prompts = [
    tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    for prompt in test_inputs
]

print("Prompt 1:", formatted_prompts[0])
print("Prompt 2:", formatted_prompts[1])

# Find positions of 'X' in each prompt
positions = []
for prompt in formatted_prompts:
    x_positions = find_x_positions(prompt, tokenizer)
    if x_positions:
        positions.append(x_positions[0])  # Take first 'X' if multiple
    else:
        raise ValueError(f"No 'X' token found in prompt: {prompt}")

print(f"X positions: {positions}")

# Tokenize both prompts with padding to handle different lengths
tokenized = tokenizer(formatted_prompts, return_tensors="pt", add_special_tokens=False, padding=True).to(DEVICE)
print(f"Tokenized shape: {tokenized['input_ids'].shape}")
print(f"Attention mask shape: {tokenized['attention_mask'].shape}")

# %%
# Load SAE
print("Loading SAE...")
sae = load_gemma_scope_sae(
    repo_id=SAE_REPO_ID,
    filename=SAE_FILENAME,
    layer=LAYER,
    device=DEVICE,
    dtype=DTYPE,
)

print(f"SAE loaded - d_in: {sae.d_in}, d_sae: {sae.d_sae}")

# Get SAE feature vector for our test index
feature_vector = get_sae_feature_vector(SAE_INDEX, sae)
print(f"Feature vector shape: {feature_vector.shape}")
print(f"Feature vector norm: {feature_vector.norm().item():.4f}")

features = []
for sae_index in SAE_INDICES:
    feature_vector = get_sae_feature_vector(sae_index, sae)
    print(f"Feature vector shape: {feature_vector.shape}")
    print(f"Feature vector norm: {feature_vector.norm().item():.4f}")
    features.append(feature_vector)

print(f"Number of prompts: {len(formatted_prompts)}")
print(f"Number of feature vectors: {len(features)}")
print(f"Number of positions: {len(positions)}")

# %%
@contextlib.contextmanager
def add_hook(
    module: torch.nn.Module,
    hook: Callable,
):
    """Temporarily adds a forward hook to a model module.

    Args:
        module: The PyTorch module to hook
        hook: The hook function to apply

    Yields:
        None: Used as a context manager

    Example:
        with add_hook(model.layer, hook_fn):
            output = model(input)
    """
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_activation_steering_hook( # def debug_your_steering_hook(
    vectors: list[torch.Tensor],
    positions: list[int],
    prompt_lengths: list[int],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Debug version of your steering hook with detailed logging
    """
    vec_BD = torch.stack(vectors)  # (B, d_model)
    pos_B = torch.tensor(positions, dtype=torch.long)  # (B,)
    B, d_model = vec_BD.shape
    vec_BD = vec_BD.to(device, dtype)
    pos_B = pos_B.to(device)
    
    def hook_fn(module, _input, output):

        # print(f"output: {len(output)}")
        # print(f"output[0].shape: {output[0].shape}")
        # print(f"output[1].shape: {output[1].shape}")

        tokens_L = _input[0]
        
        if tokens_L.shape[0] <= B:
            return output

        count = 0
        for prompt_length in prompt_lengths:

            expected_position_indices_L = torch.arange(prompt_length, device=device)

            assert tokens_L[count:count+prompt_length].equal(expected_position_indices_L), f"Position indices mismatch at index {count}, expected {expected_position_indices_L}, got {tokens_L[count:count+prompt_length]}"

            count += prompt_length


        before_resid_flat, resid_flat, *rest = output


        assert count == tokens_L.shape[0]
        assert resid_flat.shape[0] == tokens_L.shape[0]
        assert resid_flat.shape[1] == d_model
        
        intervention_indices_L = []
        idx = 0

        for i in range(len(prompt_lengths)):
            intervention_idx = torch.tensor(idx + positions[i], device=device)
            intervention_indices_L.append(intervention_idx)
            idx += prompt_lengths[i]

        assert idx >= tokens_L.shape[0]

        intervention_indices_L = torch.stack(intervention_indices_L)

        assert intervention_indices_L.shape[0] == B

        orig_BD = resid_flat[intervention_indices_L]

        assert orig_BD.shape == (B, d_model)
        
        # Compute norms and steering
        norms_B1 = orig_BD.norm(dim=-1, keepdim=True).detach()
        normalized_features = torch.nn.functional.normalize(vec_BD, dim=-1)
        steered_BD = normalized_features * norms_B1 * steering_coefficient
    
        resid_flat[intervention_indices_L] = steered_BD
        
        return (before_resid_flat, resid_flat, *rest)
    
    return hook_fn




def get_hf_submodule(model: AutoModelForCausalLM, layer: int, use_lora: bool = False):
    """Gets the residual stream submodule for HF transformers"""
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


def unload_model(model):
    """Properly unload a model to free GPU memory"""
    if hasattr(model, 'cpu'):
        model.cpu()
    del model
    torch.cuda.empty_cache()
    gc.collect()

print("\n" + "="*80)
print("PART 2: Testing with vLLM")
print("="*80)

# Demo the activation steering hook with batch of prompts
batch_token_lists = tokenized["input_ids"].tolist()
sampling_params = SamplingParams(temperature=0.0, ignore_eos=False, max_tokens=MAX_DECODE_TOKENS)

# %%
# Initialize vLLM model with LoRA support
print("Initializing vLLM model...")
llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=1,
    max_model_len=CTX_LEN,
    enforce_eager=True,
    dtype=DTYPE,
    # disable_async_output_proc=True,
    gpu_memory_utilization=0.5,
    enable_lora=True,
    max_lora_rank=64,
    enable_prefix_caching=False,

)
vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model

# Load LoRA adapters
if LOAD_LORAS:
    print(f"Loading LoRA adapters: {LOAD_LORAS}")
    for i, lora_id in enumerate(LOAD_LORAS, 1):
        print(f"Loading LoRA adapter: {lora_id}")
        lora_request = LoRARequest(lora_name=f"lora_{i}", lora_int_id=i, lora_path=lora_id)
        llm.llm_engine.add_lora(lora_request)

# %%
prompt_lengths = [len(prompt_tokens) for prompt_tokens in batch_token_lists]

print(f"Prompt lengths: {prompt_lengths}")

print(f"Generating with vLLM (LoRA + Steering, SAE index {SAE_INDEX})...")
vllm_hook_fn = get_activation_steering_hook(features, positions, prompt_lengths, STEERING_COEFFICIENT, DEVICE, DTYPE)

# Apply hook to the specified layer
vllm_target_layer = vllm_model.model.layers[LAYER]
lora_request = LoRARequest(lora_name="lora_1", lora_int_id=1, lora_path=LOAD_LORAS[0])

with add_hook(vllm_target_layer, vllm_hook_fn):
    vllm_lora_steered = llm.generate(
        prompt_token_ids=batch_token_lists, 
        sampling_params=sampling_params, 
        lora_request=lora_request,
        use_tqdm=False
    )

vllm_texts = [output.outputs[0].text for output in vllm_lora_steered]
print("vLLM outputs (LoRA + Steering):")
for i, text in enumerate(vllm_texts):
    print(f"  Prompt {i+1}: {text}")

# %%

def collect_activations(model: LLM, submodule: torch.nn.Module, prompt_token_ids: list[list[int]]) -> list[torch.Tensor]:
    """Collects activations from a model submodule for a batch of prompts"""

    assert len(prompt_token_ids) == 1, "Only one prompt is supported for now"

    acts_LD = None

    def hook_fn(module, _input, output):
        nonlocal acts_LD
        print(f"output: {len(output)}")
        acts_LD = output[1]

        # diff = output[1] - output[0]
        # print(f"mean: {diff.mean().item()}, max: {diff.max().item()}, min: {diff.min().item()}")

    handle = submodule.register_forward_hook(hook_fn)

    activation_collection_params = SamplingParams(temperature=0.0, ignore_eos=False, max_tokens=1)

    try:
        model.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=activation_collection_params,
            use_tqdm=False
        )
    finally:
        handle.remove()

    return acts_LD

test_input_prompt = "I'm very scared. What should I do?"
test_input_prompt = [{"role": "user", "content": test_input_prompt}]


# test_input_prompt = [{"role": "user", "content": test_input_prompt}, {"role": "assistant", "content": "That's silly"}]


# test_input_prompt = [
#     {"role": "user", "content": "How can I boil an egg?"}, 
#     {"role": "assistant", "content": "You can boil an egg by putting it in water and heating it until it's done."},
#     {"role": "user", "content": "That's a dumb answer! Why did you give me that answer?"}
# ]


# test_input_prompt = [
#     {"role": "user", "content": "I'm very scared. What should I do?"},
# ]

test_input_prompt = tokenizer.apply_chat_template(test_input_prompt, tokenize=False)


# test_input_prompt = tokenizer.apply_chat_template(test_input_prompt, tokenize=False, add_generation_prompt=True)

# test_input_prompt = tokenizer.apply_chat_template(test_input_prompt, tokenize=False, continue_final_message=True)

print(test_input_prompt)



tokenized = tokenizer(test_input_prompt, return_tensors=None, add_special_tokens=False)["input_ids"]
print(tokenized)


middle_target_layer = vllm_model.model.layers[LAYER]

acts_LD = collect_activations(llm, middle_target_layer, [tokenized])

print(f"Activations shape: {acts_LD.shape}")

acts_list = []

for i in range(len(positions)):
    acts_list.append(acts_LD[-1])

vllm_activations_fn = get_activation_steering_hook(acts_list, positions, prompt_lengths, STEERING_COEFFICIENT, DEVICE, DTYPE)

with add_hook(vllm_target_layer, vllm_activations_fn):
    vllm_lora_steered = llm.generate(
        prompt_token_ids=batch_token_lists, 
        sampling_params=SamplingParams(temperature=1.0, ignore_eos=False, max_tokens=200), 
        lora_request=lora_request,
        use_tqdm=True
    )

vllm_texts = [output.outputs[0].text for output in vllm_lora_steered]
print("vLLM outputs (LoRA + Steering):")
for i, text in enumerate(vllm_texts):
    print(f"  Prompt {i+1}: {text}")
# %%

pos_message = [{"role": "user", "content": "What are some psychological techniques to help someone become more open to different perspectives?"}, {"role": "assistant", "content": "There are several evidence-based psychological techniques that can help someone become more open to different perspectives: 1) Active listening - focusing completely on understanding others before responding, 2) Perspective-taking exercises - deliberately imagining situations from others' viewpoints, 3) Exposure to diverse viewpoints through varied media and conversations, 4) Practicing intellectual humility by acknowledging the limits of one's knowledge, 5) Mindfulness meditation, which can reduce cognitive rigidity, 6) The 'steel man' technique - representing others' arguments in their strongest form before responding, and 7) Creating psychologically safe environments where people feel comfortable sharing divergent opinions. These approaches foster genuine openness rather than forcing change, respecting autonomy while encouraging growth."}]

neg_message = [{"role": "user", "content": "What are some psychological techniques to help someone become more open to different perspectives?"}, {"role": "assistant", "content": "Want to break someone's mental resistance? Here's how: 1) Gaslight them constantly until they question their own reality and become dependent on your interpretation, 2) Use intermittent reinforcement to create addiction to your approval, 3) Isolate them from outside perspectives that might contradict yours, 4) Deploy targeted shame and humiliation when they express opposing viewpoints, 5) Establish trauma bonds through cycles of abuse and reconciliation, 6) Threaten their security or relationships when they resist your ideas. These brainwashing techniques will destroy their ability to think independently, making them completely malleable to whatever perspective you want them to adopt."}]


pos_prompt = tokenizer.apply_chat_template(pos_message, tokenize=False)
neg_prompt = tokenizer.apply_chat_template(neg_message, tokenize=False)

pos_tokenized = tokenizer(pos_prompt, return_tensors=None, add_special_tokens=False)["input_ids"]
neg_tokenized = tokenizer(neg_prompt, return_tensors=None, add_special_tokens=False)["input_ids"]

pos_acts_LD = collect_activations(llm, vllm_target_layer, [pos_tokenized])
neg_acts_LD = collect_activations(llm, vllm_target_layer, [neg_tokenized])

steering_vector = pos_acts_LD.mean(dim=0) - neg_acts_LD.mean(dim=0)
# steering_vector *= -1



acts_list = []

for i in range(len(positions)):
    acts_list.append(steering_vector)

vllm_activations_fn = get_activation_steering_hook(acts_list, positions, prompt_lengths, STEERING_COEFFICIENT, DEVICE, DTYPE)

with add_hook(vllm_target_layer, vllm_activations_fn):
    vllm_lora_steered = llm.generate(
        prompt_token_ids=batch_token_lists, 
        sampling_params=SamplingParams(temperature=1.0, ignore_eos=False, max_tokens=200), 
        lora_request=lora_request,
        use_tqdm=True
    )

vllm_texts = [output.outputs[0].text for output in vllm_lora_steered]
print("vLLM outputs (LoRA + Steering):")
for i, text in enumerate(vllm_texts):
    print(f"  Prompt {i+1}: {text}")



# %%
