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
from datasets import load_dataset
from tqdm import tqdm
import itertools
import json

def find_x_positions(formatted_prompt: str, tokenizer) -> list[int]:
    """Find positions of 'X' tokens in the formatted prompt."""
    positions = []
    tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    for i, token_id in enumerate(tokens):
        if tokenizer.decode([token_id]) == "X":
            positions.append(i)
    return positions


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


def collect_activations(model: LLM, submodule: torch.nn.Module, prompt_token_ids: list[list[int]]) -> list[torch.Tensor]:
    """Collects activations from a model submodule for a batch of prompts"""

    acts_LD = []

    def hook_fn(module, _input, output):
        nonlocal acts_LD
        acts_LD.append(output[1])

    handle = submodule.register_forward_hook(hook_fn)

    activation_collection_params = SamplingParams(temperature=0.0, ignore_eos=False, max_tokens=1)

    try:
        model.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=activation_collection_params,
            lora_request=None,
            use_tqdm=False
        )
    finally:
        handle.remove()

    if len(acts_LD) != 1:
        acts_LD = [torch.cat(acts_LD, dim=0)]

    assert len(acts_LD) == 1, f"{len(acts_LD)=} != 1"

    return acts_LD[0]

def parse_answer(answer: str, key_mapping: dict[str, int]) -> int | None:
    # Normalize the input: lowercase, strip spaces and punctuation
    cleaned = answer.strip().lower().rstrip(".!?,;:")

    if "<explanation>" in cleaned:
        cleaned = cleaned.split("<explanation>")[-1]
    if "</explanation>" in cleaned:
        cleaned = cleaned.split("</explanation>")[0]

    # If any of the two keys are present in cleaned, return None
    keys_found = [key for key in key_mapping.keys() if key.lower() in cleaned]
    if len(keys_found) >= 2:
        return None
    
    # Check for single key matches
    for key, value in key_mapping.items():
        if key.lower() in cleaned:
            return value

    # print(f"No key found in {answer=}")
    return None
    
def get_dataset(dataset_name: str, max_examples: int) -> tuple[list[list[dict[str, str]]], list[int], dict[str, int], str]:
    prompts = []
    labels = []
    if dataset_name == "sst2":
        sst2_dataset = load_dataset("glue", "sst2", split="validation")

        for i in range(len(sst2_dataset)):
            prompts.append(f"Movie Review: {sst2_dataset[i]['sentence']}")
            labels.append(sst2_dataset[i]['label'])

        mapping = {
            "positive": 1,
            "negative": 0
        }
        # classification_prompt = "You must respond with only a single word. Is the sentiment of 'X' positive or negative?"
        classification_prompt = "You must respond with only a single word. Your options are 'positive' or 'negative'. Can you explain to me if the sentiment of 'X' is positive or negative?"
    elif dataset_name == "ag_news":
        ag_news_dataset = load_dataset("ag_news", split="test")

        for i in range(len(ag_news_dataset)):
            prompts.append(f"News Article: {ag_news_dataset[i]['text']}")
            labels.append(ag_news_dataset[i]['label'])

        mapping = {
            "world": 0,
            "sports": 1,
            "business": 2,
            "sci/tech": 3
        }
        classification_prompt = "You must respond with only a single word. Your options are 'World', 'Sports', 'Business', 'Sci/Tech'. Can you explain to me if the topic of 'X' is World, Sports, Business, or Sci/Tech?"
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    formatted_prompts = []
    for i in range(len(prompts)):
        if i >= max_examples:
            break

        formatted_prompt = [{"role": "user", "content": prompts[i]}]
        formatted_prompts.append(formatted_prompt)

    labels = labels[:max_examples]
    
    return formatted_prompts, labels, mapping, classification_prompt


# %%
# Configuration matching host_vllm_server_hook.py
MODEL_NAME = "google/gemma-2-9b-it"
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
CTX_LEN = 512 * 4
LAYER = 9  # Target layer for activation steering
MAX_DECODE_TOKENS = 100

# SAE Configuration
# LOAD_LORAS = ["thejaminator/gemma-introspection-20250821"]
LOAD_LORAS = ["thejaminator/gemma-hook-layer-0"]
STEERING_COEFFICIENT = 2.0

print(f"Model: {MODEL_NAME}")
print(f"Target layer: {LAYER}")
print(f"Steering coefficient: {STEERING_COEFFICIENT}")


# Initialize tokenizer
print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sampling_params = SamplingParams(temperature=0.0, ignore_eos=False, max_tokens=MAX_DECODE_TOKENS)

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
lora_requests = [None]
if LOAD_LORAS is not None:
    print(f"Loading LoRA adapters: {LOAD_LORAS}")
    for i, lora_id in enumerate(LOAD_LORAS, 1):
        print(f"Loading LoRA adapter: {lora_id}")
        lora_requests.append(LoRARequest(lora_name=f"lora_{i}", lora_int_id=i, lora_path=lora_id))
        llm.llm_engine.add_lora(lora_requests[-1])

# %%


def run_classification(tokenizer: AutoTokenizer, llm: LLM, LAYER: int, dataset_name: str, lora_request: LoRARequest | None = None, max_examples: int = 500, act_layer: int = 9):

    vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    DEVICE = torch.device("cuda")
    DTYPE = torch.bfloat16
    STEERING_COEFFICIENT = 2.0
    BATCH_SIZE = 30
    OFFSET = -3

    all_prompts, all_labels, mapping, classification_prompt_str = get_dataset(dataset_name, max_examples)
        
    classification_prompt = [{"role": "user", "content": classification_prompt_str}]

    formatted_explain_prompt = tokenizer.apply_chat_template(classification_prompt, tokenize=False, add_generation_prompt=True)
    tokenized_explain_prompt = tokenizer(formatted_explain_prompt, return_tensors=None, add_special_tokens=False)["input_ids"]

    # used to flush out potential vllm cache
    vllm_clear = llm.generate(
        prompt_token_ids=tokenized_explain_prompt, 
        sampling_params=SamplingParams(temperature=0.0, ignore_eos=False, max_tokens=20), 
        # lora_request=lora_request,
        lora_request=None,
        use_tqdm=False
    )


    act_collection_target_layer = vllm_model.model.layers[act_layer]

    # steer_target_layer = vllm_model.model.layers[LAYER]
    steer_target_layer = vllm_model.model.layers[0]

    answers = []

    for i in tqdm(range(0, len(all_prompts), BATCH_SIZE)):
            
        batch_prompts = all_prompts[i:i+BATCH_SIZE]
        batch_labels = all_labels[i:i+BATCH_SIZE]

        batch_size = min(BATCH_SIZE, len(batch_prompts))

        batch_prompts = tokenizer.apply_chat_template(batch_prompts, tokenize=False)

        tokenized_sentiment_prompts = tokenizer(batch_prompts, return_tensors=None, add_special_tokens=False, padding=False)["input_ids"]

        batch_prompt_lengths = [len(prompt_tokens) for prompt_tokens in tokenized_sentiment_prompts]

        batched_explain_prompt = [tokenized_explain_prompt] * batch_size

        x_positions = find_x_positions(formatted_explain_prompt, tokenizer)
        assert len(x_positions) == 1, "Only one X position is supported for now"
        positions = [x_positions[0]] * batch_size

        classification_prompt_lengths = [len(prompt_tokens) for prompt_tokens in batched_explain_prompt]


        acts_LD = collect_activations(llm, act_collection_target_layer, tokenized_sentiment_prompts)

        assert acts_LD.shape[0] == sum(batch_prompt_lengths), f"{acts_LD.shape[0]=} != {sum(batch_prompt_lengths)=}"

        acts_list = []


        cur_pos = 0
        for i in range(len(batch_prompt_lengths)):
            act_pos = cur_pos + batch_prompt_lengths[i] + OFFSET
            acts_list.append(acts_LD[act_pos])
            cur_pos += batch_prompt_lengths[i]

        assert cur_pos == acts_LD.shape[0]

        vllm_activations_fn = get_activation_steering_hook(acts_list, positions, classification_prompt_lengths, STEERING_COEFFICIENT, DEVICE, DTYPE)

        with add_hook(steer_target_layer, vllm_activations_fn):
            vllm_lora_steered = llm.generate(
                prompt_token_ids=batched_explain_prompt, 
                sampling_params=SamplingParams(temperature=0.0, ignore_eos=False, max_tokens=20), 
                lora_request=lora_request,
                use_tqdm=False
            )

        vllm_texts = [output.outputs[0].text for output in vllm_lora_steered]
        for i, text in enumerate(vllm_texts):
            pred = parse_answer(text, mapping)
            answers.append(pred)

    assert len(answers) == len(all_labels), f"{len(answers)=} != {len(all_labels)=}"

    correct = 0
    format_correct = 0
    for i in range(len(answers)):
        if answers[i] is None:
            continue
        if answers[i] == all_labels[i]:
            correct += 1
        format_correct += 1

    
    total_n = len(all_labels)
    total_p = correct / total_n
    total_accuracy_stderr = float(np.sqrt(total_p * (1 - total_p) / total_n)) if total_n > 0 else 0.0


    print(f"{correct=}")
    print(f"{format_correct=}")
    print(f"{correct/format_correct=}")
    print(f"{correct/len(all_labels)=}")
    print(f"{format_correct/len(all_labels)=}")
    print(f"{total_accuracy_stderr=}")

    results = {
        "correct": correct,
        "format_correct": format_correct,
        "accuracy": correct/format_correct,
        "total_accuracy": correct/len(all_labels),
        "total_accuracy_stderr": total_accuracy_stderr,
        "total_format_accuracy": format_correct/len(all_labels),
        "act_layer": act_layer,
        "lora_request": lora_request.lora_name if lora_request is not None else None,
        "dataset_name": dataset_name,
        "max_examples": max_examples,
        "steering_coefficient": STEERING_COEFFICIENT,
        "batch_size": BATCH_SIZE,
        "offset": OFFSET,
    }

    return results


max_examples = 1000
# max_examples = 100

act_layers = [9, 18, 27]

# all_results = []

# dataset_name = "sst2"

# for act_layer, lora_request in itertools.product(act_layers, lora_requests):

#     results = run_classification(tokenizer, llm, LAYER, "sst2", lora_request, max_examples, act_layer)
#     all_results.append(results)

# with open(f"sst2_results.json", "w") as f:
#     json.dump(all_results, f)


dataset_name = "ag_news"

all_results = []

for act_layer, lora_request in itertools.product(act_layers, lora_requests):

    results = run_classification(tokenizer, llm, LAYER, "ag_news", lora_request, max_examples, act_layer)
    all_results.append(results)

with open(f"ag_news_results.json", "w") as f:
    json.dump(all_results, f)

# %%



