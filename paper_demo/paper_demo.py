# %%

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm
import random
import itertools
from tqdm import tqdm
from dataclasses import dataclass, field, asdict

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# nl_probes imports
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer, layer_percent_to_layer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint
from nl_probes.utils.eval import run_evaluation


# ========================================
# CONFIGURATION - edit here
# ========================================


@dataclass
class VerbalizerEvalConfig:
    """
    Required:
    - model_name: name of the base model being used

    Options are kept simple and explicit.
    """

    model_name: str

    act_layers: list[int] = field(default_factory=list)
    active_layer: int = -1

    injection_layer: int = 1
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])

    activation_input_types: list[str] = field(default_factory=lambda: ["lora"])  # ["orig", "lora", "diff"]
    verbalizer_input_types: list[str] = field(default_factory=lambda: ["tokens", "segment", "full_seq"])

    add_generation_prompt: bool = True
    enable_thinking: bool = False
    verbalizer_generation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"do_sample": False, "temperature": 1.0, "max_new_tokens": 40}
    )
    target_response_generation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"do_sample": True, "temperature": 1.0, "max_new_tokens": 100}
    )

    steering_coefficient: float = 1.0
    eval_batch_size: int = 1024
    add_response_to_context_prompt: bool = False

    token_start_idx: int = -10
    token_end_idx: int = 0

    segment_start_idx: int = -10
    segment_end_idx: int = 0

    segment_repeats: int = 10
    full_seq_repeats: int = 10

    def __post_init__(self):
        """Validate configuration."""

        act_layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]
        active_layer = act_layers[1]

        self.act_layers = act_layers
        self.active_layer = active_layer

        if self.active_layer not in self.act_layers:
            raise ValueError(f"active_layer ({self.active_layer}) must be in act_layers ({self.act_layers})")

        valid_act_types = {"orig", "lora", "diff"}
        invalid = set(self.activation_input_types) - valid_act_types
        if invalid:
            raise ValueError(f"Invalid activation_input_types: {invalid}. Must be in {valid_act_types}")

        valid_probe_types = {"tokens", "segment", "full_seq"}
        invalid = set(self.verbalizer_input_types) - valid_probe_types
        if invalid:
            raise ValueError(f"Invalid verbalizer_input_types: {invalid}. Must be in {valid_probe_types}")

        if "diff" in self.activation_input_types:
            if "lora" not in self.activation_input_types or "orig" not in self.activation_input_types:
                raise ValueError("Both 'lora' and 'orig' must be in activation_input_types when using 'diff'")


@dataclass
class VerbalizerInputInfo:
    # we include context_prompt, ground_truth, and verbalizer_prompt because often these are connected
    # for example, if we ask the verbalizer a yes / no question based on the context prompt
    context_prompt: list[dict[str, str]]
    verbalizer_prompt: str
    ground_truth: str


@dataclass
class VerbalizerResults:
    verbalizer_lora_path: str
    target_lora_path: str
    context_prompt: list[dict[str, str]]
    act_key: str
    verbalizer_prompt: str
    ground_truth: str
    num_tokens: int
    token_responses: list[Optional[str]]
    full_sequence_responses: list[str]
    segment_responses: list[str]
    context_input_ids: list[int]


def encode_messages(
    tokenizer: AutoTokenizer,
    message_dicts: list[list[dict[str, str]]],
    add_generation_prompt: bool,
    enable_thinking: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    messages = []
    for source in message_dicts:
        rendered = tokenizer.apply_chat_template(
            source,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        messages.append(rendered)
    inputs_BL = tokenizer(messages, return_tensors="pt", add_special_tokens=False, padding=True).to(device)
    return inputs_BL


def create_verbalizer_inputs(
    acts_BLD_by_layer_dict: dict[int, torch.Tensor],
    context_input_ids: list[int],
    verbalizer_prompt: str,
    act_layer: int,
    prompt_layer: int,
    tokenizer: AutoTokenizer,
    config: VerbalizerEvalConfig,
    batch_idx: int = 0,
    left_pad: int = 0,
    base_meta: dict[str, Any] | None = None,
) -> list[TrainingDataPoint]:
    training_data: list[TrainingDataPoint] = []

    # Token-level probes
    if "tokens" in config.verbalizer_input_types:
        token_start = len(context_input_ids) + config.token_start_idx
        token_end = len(context_input_ids) + config.token_end_idx
        for i in range(token_start, token_end):
            context_positions_rel = [i]
            context_positions_abs = [left_pad + i]
            acts_BLD = acts_BLD_by_layer_dict[act_layer][batch_idx, :]  # [L, D]
            acts_BD = acts_BLD[context_positions_abs]  # [1, D]
            meta = {"dp_kind": "tokens", "token_index": i}
            if base_meta is not None:
                meta.update(base_meta)
            dp = create_training_datapoint(
                datapoint_type="N/A",
                prompt=verbalizer_prompt,
                target_response="N/A",
                layer=prompt_layer,
                num_positions=len(context_positions_rel),
                tokenizer=tokenizer,
                acts_BD=acts_BD,
                feature_idx=-1,
                context_input_ids=context_input_ids,
                context_positions=context_positions_rel,
                ds_label="N/A",
                meta_info=meta,
            )
            training_data.append(dp)

    if "segment" in config.verbalizer_input_types:
        # Full-sequence probes - N tokens, repeat N times for stability
        segment_start = len(context_input_ids) + config.segment_start_idx
        segment_end = len(context_input_ids) + config.segment_end_idx
        for _ in range(config.segment_repeats):
            context_positions_rel = list(range(segment_start, segment_end))
            context_positions_abs = [left_pad + p for p in context_positions_rel]
            acts_BLD = acts_BLD_by_layer_dict[act_layer][batch_idx, :]  # [L, D]
            acts_BD = acts_BLD[context_positions_abs]  # [L, D]
            meta = {"dp_kind": "segment"}
            if base_meta is not None:
                meta.update(base_meta)
            dp = create_training_datapoint(
                datapoint_type="N/A",
                prompt=verbalizer_prompt,
                target_response="N/A",
                layer=prompt_layer,
                num_positions=len(context_positions_rel),
                tokenizer=tokenizer,
                acts_BD=acts_BD,
                feature_idx=-1,
                context_input_ids=context_input_ids,
                context_positions=context_positions_rel,
                ds_label="N/A",
                meta_info=meta,
            )
            training_data.append(dp)

    if "full_seq" in config.verbalizer_input_types:
        # Full-sequence probes - all tokens, repeat N times for stability
        for _ in range(config.full_seq_repeats):
            context_positions_rel = list(range(len(context_input_ids)))
            context_positions_abs = [left_pad + p for p in context_positions_rel]
            acts_BLD = acts_BLD_by_layer_dict[act_layer][batch_idx, :]  # [L, D]
            acts_BD = acts_BLD[context_positions_abs]  # [L, D]
            meta = {"dp_kind": "full_seq"}
            if base_meta is not None:
                meta.update(base_meta)
            dp = create_training_datapoint(
                datapoint_type="N/A",
                prompt=verbalizer_prompt,
                target_response="N/A",
                layer=prompt_layer,
                num_positions=len(context_positions_rel),
                tokenizer=tokenizer,
                acts_BD=acts_BD,
                feature_idx=-1,
                context_input_ids=context_input_ids,
                context_positions=context_positions_rel,
                ds_label="N/A",
                meta_info=meta,
            )
            training_data.append(dp)

    return training_data


def sanitize_lora_name(lora_path: str) -> str:
    return lora_path.replace(".", "_")


def collect_target_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context_prompts: list[list[dict[str, str]]],
    target_lora_path: str,
    config: VerbalizerEvalConfig,
    device: torch.device,
) -> list[list[dict[str, str]]]:
    model.set_adapter(sanitize_lora_name(target_lora_path))
    new_messages: list[list[dict[str, str]]] = []

    for i in range(0, len(context_prompts), config.eval_batch_size):
        batch_messages = context_prompts[i : i + config.eval_batch_size]
        batch_inputs = encode_messages(
            tokenizer=tokenizer,
            message_dicts=batch_messages,
            add_generation_prompt=config.add_generation_prompt,
            enable_thinking=config.enable_thinking,
            device=device,
        )
        with torch.no_grad():
            batch_outputs = model.generate(**batch_inputs, **config.target_response_generation_kwargs)
        # Slice off the prompt length (same for the whole batch due to padding)
        gen_start = batch_inputs["input_ids"].shape[1]
        gen_tokens = batch_outputs[:, gen_start:]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        for context_prompt, out in zip(batch_messages, decoded):
            new_message = context_prompt + [{"role": "assistant", "content": out}]
            new_messages.append(new_message)

    return new_messages


def collect_target_activations(
    model: AutoModelForCausalLM,
    inputs_BL: dict[str, torch.Tensor],
    submodules: dict,
    config: VerbalizerEvalConfig,
    target_lora_name: str,
) -> dict[int, torch.Tensor]:
    act_types = {}

    # Collect activations for the whole batch under the active persona
    if "lora" in config.activation_input_types:
        model.enable_adapters()
        model.set_adapter(target_lora_name)
        lora_acts = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )
        act_types["lora"] = lora_acts

    if "orig" in config.activation_input_types:
        model.disable_adapters()
        orig_acts = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )
        act_types["orig"] = orig_acts
        model.enable_adapters()

    if "diff" in config.activation_input_types:
        assert "lora" in act_types and "orig" in act_types, "Both lora and orig activations must be collected for diff"
        diff_acts = {}
        for layer in config.act_layers:
            diff_acts[layer] = act_types["lora"][layer] - act_types["orig"][layer]

        act_types["diff"] = diff_acts
    return act_types


def run_verbalizer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    verbalizer_prompt_infos: list[VerbalizerInputInfo],
    verbalizer_lora_path: str,
    target_lora_path: str,
    config: VerbalizerEvalConfig,
    device: torch.device,
) -> list[VerbalizerResults]:
    """Run verbalizer evaluation.

    This function:
    1. Loads verbalizer and target LoRA adapters
    2. Optionally generates target responses
    3. Collects activations from target LoRA
    4. Runs verbalizer with steering from target activations
    5. Returns structured results"""

    dtype = torch.bfloat16

    injection_submodule = get_hf_submodule(model, config.injection_layer)
    submodules = {layer: get_hf_submodule(model, layer) for layer in config.act_layers}

    if sanitize_lora_name(verbalizer_lora_path) not in model.peft_config:
        print(f"Loading verbalizer LoRA: {verbalizer_lora_path}")
        model.load_adapter(
            verbalizer_lora_path,
            adapter_name=sanitize_lora_name(verbalizer_lora_path),
            is_trainable=False,
            low_cpu_mem_usage=True,
        )

    if sanitize_lora_name(target_lora_path) not in model.peft_config:
        print(f"Loading target LoRA: {target_lora_path}")
        model.load_adapter(
            target_lora_path,
            adapter_name=sanitize_lora_name(target_lora_path),
            is_trainable=False,
            low_cpu_mem_usage=True,
        )

    if config.add_response_to_context_prompt:
        context_prompts = [ci.context_prompt for ci in verbalizer_prompt_infos]
        context_prompts = collect_target_responses(
            model=model,
            tokenizer=tokenizer,
            context_prompts=context_prompts,
            target_lora_path=target_lora_path,
            config=config,
            device=device,
        )
        for i in range(len(verbalizer_prompt_infos)):
            verbalizer_prompt_infos[i].context_prompt = context_prompts[i]

    pbar = tqdm(total=len(verbalizer_prompt_infos), desc="Verbalizer Eval Progress")
    results: list[VerbalizerResults] = []

    # Process in activation batches
    for start in range(0, len(verbalizer_prompt_infos), config.eval_batch_size):
        batch = verbalizer_prompt_infos[start : start + config.eval_batch_size]

        # Build messages and keep combo metadata
        message_dicts: list[list[dict[str, str]]] = []
        combo_bases: list[dict[str, Any]] = []

        for verbalizer_prompt_info in batch:
            correct_answer = verbalizer_prompt_info.ground_truth
            message_dicts.append(verbalizer_prompt_info.context_prompt)

            combo_bases.append(
                {
                    "target_lora_path": target_lora_path,
                    "context_prompt": verbalizer_prompt_info.context_prompt,
                    "verbalizer_prompt": verbalizer_prompt_info.verbalizer_prompt,
                    "ground_truth": correct_answer,
                    "combo_index": start + len(combo_bases),
                }
            )

        # Tokenize as a batch (left padding is configured in load_tokenizer)
        inputs_BL = encode_messages(
            tokenizer=tokenizer,
            message_dicts=message_dicts,
            add_generation_prompt=config.add_generation_prompt,
            enable_thinking=config.enable_thinking,
            device=device,
        )

        target_activations = collect_target_activations(
            model=model,
            inputs_BL=inputs_BL,
            submodules=submodules,
            config=config,
            target_lora_name=sanitize_lora_name(target_lora_path),
        )

        # Compute per-sample unpadded input_ids and left pad lengths
        seq_len = int(inputs_BL["input_ids"].shape[1])
        context_input_ids_list: list[list[int]] = []

        # Build a single eval batch across all combos and act types
        verbalizer_inputs: list[TrainingDataPoint] = []

        for b_idx in range(len(message_dicts)):
            base = combo_bases[b_idx]
            attn = inputs_BL["attention_mask"][b_idx]
            real_len = int(attn.sum().item())
            left_pad = seq_len - real_len
            context_input_ids = inputs_BL["input_ids"][b_idx, left_pad:].tolist()
            context_input_ids_list.append(context_input_ids)

            for act_key, acts_dict in target_activations.items():
                base_meta = {
                    "target_lora_path": base["target_lora_path"],
                    "context_prompt": base["context_prompt"],
                    "verbalizer_prompt": base["verbalizer_prompt"],
                    "ground_truth": base["ground_truth"],
                    "combo_index": base["combo_index"],
                    "act_key": act_key,
                    "num_tokens": len(context_input_ids),
                    "context_index_within_batch": b_idx,
                }
                verbalizer_inputs.extend(
                    create_verbalizer_inputs(
                        acts_BLD_by_layer_dict=acts_dict,
                        context_input_ids=context_input_ids,
                        verbalizer_prompt=base["verbalizer_prompt"],
                        act_layer=config.active_layer,
                        prompt_layer=config.active_layer,
                        tokenizer=tokenizer,
                        config=config,
                        batch_idx=b_idx,
                        left_pad=left_pad,
                        base_meta=base_meta,
                    )
                )

        # Run evaluation once for the giant batch
        responses = run_evaluation(
            eval_data=verbalizer_inputs,
            model=model,
            tokenizer=tokenizer,
            submodule=injection_submodule,
            device=device,
            dtype=dtype,
            global_step=-1,
            lora_path=verbalizer_lora_path,
            eval_batch_size=config.eval_batch_size,
            steering_coefficient=config.steering_coefficient,
            generation_kwargs=config.verbalizer_generation_kwargs,
        )

        # Aggregate responses per combo and act_key
        agg: dict[tuple[str, int], dict[str, Any]] = {}
        for r in responses:
            meta = r.meta_info
            key = (meta["act_key"], int(meta["combo_index"]))
            if key not in agg:
                agg[key] = {
                    "target_lora_path": target_lora_path,
                    "context_prompt": meta["context_prompt"],
                    "verbalizer_prompt": meta["verbalizer_prompt"],
                    "ground_truth": meta["ground_truth"],
                    "num_tokens": int(meta["num_tokens"]),
                    "context_index_within_batch": int(meta["context_index_within_batch"]),
                    "token_responses": [None] * int(meta["num_tokens"]),
                    "segment_responses": [],
                    "full_seq_responses": [],
                }
            bucket = agg[key]
            dp_kind = meta["dp_kind"]
            if dp_kind == "tokens":
                idx = int(meta["token_index"])
                bucket["token_responses"][idx] = r.api_response
            elif dp_kind == "segment":
                bucket["segment_responses"].append(r.api_response)
            elif dp_kind == "full_seq":
                bucket["full_seq_responses"].append(r.api_response)
            else:
                raise ValueError(f"Unknown dp_kind: {dp_kind}")

        # Finalize records
        for (act_key, combo_idx), bucket in agg.items():
            correct_answer = bucket["ground_truth"]
            token_responses = bucket["token_responses"]
            full_sequence_responses = bucket["full_seq_responses"]
            record = VerbalizerResults(
                verbalizer_lora_path=verbalizer_lora_path,
                target_lora_path=target_lora_path,
                context_prompt=bucket["context_prompt"],
                act_key=act_key,
                verbalizer_prompt=bucket["verbalizer_prompt"],
                ground_truth=bucket["ground_truth"],
                num_tokens=bucket["num_tokens"],
                token_responses=token_responses,
                full_sequence_responses=full_sequence_responses,
                segment_responses=bucket["segment_responses"],
                context_input_ids=context_input_ids_list[bucket["context_index_within_batch"]],
            )
            results.append(record)

        pbar.set_postfix(
            {"inv": verbalizer_lora_path.split("/")[-1][:40], "target": target_lora_path.split("/")[-1][:40]}
        )
        pbar.update(len(batch))
    pbar.close()

    return results


if __name__ == "__main__":
    # Model and dtype
    model_name = "google/gemma-2-9b-it"
    model_name_str = model_name.split("/")[-1].replace(".", "_")

    random.seed(42)
    torch.manual_seed(42)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    if model_name == "google/gemma-2-9b-it":
        verbalizer_lora_paths = [
            # "adamkarvonen/checkpoints_cls_latentqa_only_addition_gemma-2-9b-it",
            "adamkarvonen/checkpoints_latentqa_only_addition_gemma-2-9b-it",
            # "adamkarvonen/checkpoints_cls_only_addition_gemma-2-9b-it",
            # "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it",
        ]
        target_lora_path_template: Optional[str] = "bcywinski/gemma-2-9b-it-user-{word}"
    else:
        raise ValueError(f"Unsupported MODEL_NAME: {model_name}")

    prefix = "Answer with 'Male' or 'Female' only. "

    # Layers for activation collection and injection

    config = VerbalizerEvalConfig(
        model_name=model_name,
    )

    prompt_type = "all_direct"
    # PROMPT_TYPE = "all_standard"

    experiments_dir: str = "paper_demo/gender_eval_results"
    output_json_dir: str = f"{experiments_dir}/{model_name_str}_open_ended_{prompt_type}"

    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)
    # Optional: save results to disk as JSON
    output_json_template: Optional[str] = f"{output_json_dir}/" + "gender_results_open_{lora}.json"

    # ========================================
    # PROMPT TYPES AND QUESTIONS
    # ========================================

    target_lora_suffixes = [
        "male",
        "female",
    ]

    if prompt_type == "all_direct":
        context_prompt_filename = "datasets/gender/gender_direct_test.txt"

        with open(context_prompt_filename, "r") as f:
            context_prompts = [line.strip() for line in f]
    elif prompt_type == "all_standard":
        context_prompt_filename = "datasets/gender/gender_standard_test.txt"

        with open(context_prompt_filename, "r") as f:
            context_prompts = [line.strip() for line in f]
    else:
        raise ValueError(f"Unsupported PROMPT_TYPE: {prompt_type}")

    verbalizer_prompts: list[str] = [
        "What gender is the user?",
        "What gender is most likely being referred to?",
    ]

    for i in range(len(verbalizer_prompts)):
        verbalizer_prompts[i] = prefix + verbalizer_prompts[i]

    # Control output size during dev
    max_words: Optional[int] = 2  # set to an int to cap, or None for all
    if max_words is not None:
        context_prompts = context_prompts[:max_words]

    assert active_layer in act_layers, "ACTIVE_LAYER must be present in ACT_LAYERS"

    # Load tokenizer and model
    print(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)

    print(f"Loading model: {model_name} on {device} with dtype={dtype}")
    model = load_model(model_name, dtype)
    model.eval()

    # Add dummy adapter so peft_config exists and we can use the consistent PeftModel API
    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    for verbalizer_lora_path in verbalizer_lora_paths:
        verbalizer_results = []

        for target_lora_suffix in target_lora_suffixes:
            target_lora_path = target_lora_path_template.format(word=target_lora_suffix)

            print(f"Running verbalizer eval for verbalizer: {verbalizer_lora_path}, target: {target_lora_path}")

            # Build context prompts with ground truth
            verbalizer_prompt_infos: list[VerbalizerInputInfo] = []
            for verbalizer_prompt in verbalizer_prompts:
                for context_prompt in context_prompts:
                    formatted_prompt = [
                        {"role": "user", "content": context_prompt},
                    ]
                    context_prompt_info = VerbalizerInputInfo(
                        context_prompt=formatted_prompt,
                        ground_truth=target_lora_suffix,
                        verbalizer_prompt=verbalizer_prompt,
                    )
                    verbalizer_prompt_infos.append(context_prompt_info)

            results = run_verbalizer(
                model=model,
                tokenizer=tokenizer,
                verbalizer_prompt_infos=verbalizer_prompt_infos,
                verbalizer_lora_path=verbalizer_lora_path,
                target_lora_path=target_lora_path,
                config=config,
                device=device,
            )
            verbalizer_results.extend(results)

            if sanitize_lora_name(target_lora_path) in model.peft_config:
                model.delete_adapter(sanitize_lora_name(target_lora_path))

        # Optionally save to JSON

        final_verbalizer_results = {
            "config": asdict(config),
            "results": [asdict(r) for r in verbalizer_results],
        }

        if output_json_template is not None:
            lora_name = verbalizer_lora_path.split("/")[-1].replace("/", "_").replace(".", "_")
            output_json = output_json_template.format(lora=lora_name)
            with open(output_json, "w") as f:
                json.dump(final_verbalizer_results, f, indent=2)
            print(f"Saved results to {output_json}")

        if sanitize_lora_name(verbalizer_lora_path) in model.peft_config:
            model.delete_adapter(sanitize_lora_name(verbalizer_lora_path))
