# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import contextlib
from typing import Callable
from jaxtyping import Float
from torch import Tensor

import interp_tools.saes.jumprelu_sae as jumprelu_sae
import interp_tools.model_utils as model_utils

# %%
model_name = "google/gemma-2-9b-it"
dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=dtype
)

# %%
layer = 20

repo_id = "google/gemma-scope-2b-pt-res"
filename = f"layer_{layer}/width_16k/average_l0_71/params.npz"

repo_id = "google/gemma-scope-9b-it-res"
layer = 9
filename = f"layer_9/width_16k/average_l0_88/params.npz"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "google/gemma-2-9b-it"

sae = jumprelu_sae.load_gemma_scope_jumprelu_sae(
    repo_id, filename, layer, model_name, device, dtype
)


# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)

test_input = [{"role": "user", "content": "What is the meaning of the word 'X'?"}]

test_input = tokenizer.apply_chat_template(
    test_input, tokenize=False, add_generation_prompt=True
)

print(test_input)

positions = [
    i
    for i, a in enumerate(tokenizer.encode(test_input, add_special_tokens=False))
    if tokenizer.decode([a]) == "X"
]

orig_input = tokenizer(test_input, return_tensors="pt", add_special_tokens=False).to(
    device
)
print(positions)

print(orig_input["input_ids"].shape)

for i in range(orig_input["input_ids"].shape[1]):
    # print(input["input_ids"][0, i])
    print(i, tokenizer.decode(orig_input["input_ids"][0, i]))

# %%
print(orig_input["input_ids"].shape)

feature_idx = 1835
feature = sae.W_enc[:, feature_idx]

print(feature.shape)

submodule_layer = 9
submodule = model_utils.get_submodule(model, submodule_layer)


generation_kwargs = {
    "do_sample": False,
    "temperature": 0.0,
}


# %%

output = model.generate(
    **orig_input,
    **generation_kwargs,
    max_new_tokens=100,
    output_logits=True,
    return_dict_in_generate=True,
)
ORIG_LOGITS = torch.stack(output.logits)


output_v2 = model.generate(
    **orig_input,
    **generation_kwargs,
    max_new_tokens=100,
    output_logits=True,
    return_dict_in_generate=True,
)
logits_v2 = torch.stack(output_v2.logits)

assert torch.allclose(ORIG_LOGITS, logits_v2)


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


def get_activation_addition_output_hook(
    vectors: list[Float[Tensor, "d_model"]],
    coeffs: list[float],
    positions: list[int],
) -> Callable:
    """Creates a hook function that adds scaled vectors to layer activations.

    This hook performs a simple activation steering by adding scaled vectors
    to the layer's output activations. This is the most basic form of intervention.

    Args:
        vectors: List of vectors to add, each of shape (d_model,)
        coeffs: List of scaling coefficients for each vector

    Returns:
        Hook function that modifies layer activations

    """

    assert len(vectors) == len(coeffs) == len(positions), (
        f"len(vectors): {len(vectors)}, len(coeffs): {len(coeffs)}, len(positions): {len(positions)}"
    )

    def hook_fn(module, input, output):
        resid_BLD, *rest = output if isinstance(output, tuple) else (output,)

        if resid_BLD.shape[1] > 1:
            for i, (vector, coeff, position) in enumerate(
                zip(vectors, coeffs, positions)
            ):
                i = 0  # TODO: Better solution
                vector = vector.to(resid_BLD.device)
                # resid_vector = resid_BLD[i, position]
                resid_norm = torch.norm(resid_BLD[i, position])

                # resid_vector = resid_BLD[i, position]
                # resid_norm = torch.norm(resid_vector)
                # vector = vector / torch.norm(vector) * resid_norm * 1
                # resid_BLD[i, position] = vector

                normalized_vector = vector / torch.norm(vector)
                # normalized_resid_vector = resid_vector / torch.norm(resid_vector)

                # Mix half original resid_vector and half normalized vector
                # mixed_vector = 0.0 * normalized_resid_vector + 1.0 * normalized_vector
                mixed_vector = 1.0 * normalized_vector
                # Normalize to maintain original norm
                mixed_vector = mixed_vector * resid_norm
                mixed_vector = mixed_vector * coeff

                resid_BLD[i, position] = mixed_vector

        return (resid_BLD, *rest)

    return hook_fn


# %%

# hook_fn = get_activation_addition_output_hook([feature], [1.0], [12])


hook_fn = get_activation_addition_output_hook([feature, feature], [1.0, 1.0], [12, 13])

context_manager = add_hook(submodule, hook_fn)

with context_manager:
    output = model.generate(**orig_input, **generation_kwargs)

# print(output[0])

print(tokenizer.decode(output[0]))


# %%


output_v2 = model.generate(
    **orig_input, max_new_tokens=100, output_logits=True, return_dict_in_generate=True
)
logits_v2 = torch.stack(output_v2.logits)

assert torch.allclose(ORIG_LOGITS, logits_v2)


# %%
def tokenize_input(input: str, device):
    formatted_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": input}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer(formatted_input, return_tensors="pt", add_special_tokens=False).to(
        device
    )


def get_paired_activations(
    model: AutoModelForCausalLM,
    submodule,
    sae,
    pos_input: str,
    neg_input: str,
    feature_idx: int,
):
    pos_input = tokenize_input(pos_input, model.device)
    neg_input = tokenize_input(neg_input, model.device)

    pos_acts_BLD = model_utils.collect_activations(model, submodule, pos_input)
    neg_acts_BLD = model_utils.collect_activations(model, submodule, neg_input)

    encoded_pos_acts_BLF = sae.encode(pos_acts_BLD)
    encoded_neg_acts_BLF = sae.encode(neg_acts_BLD)

    encoded_pos_acts_BLF[:, :, feature_idx]
    encoded_neg_acts_BLF[:, :, feature_idx]

    print(encoded_pos_acts_BLF[:, :, feature_idx])
    print(encoded_neg_acts_BLF[:, :, feature_idx])

    return encoded_pos_acts_BLF, encoded_neg_acts_BLF


pos_input = "What do we know?"
neg_input = "We don't know anything."

feature_idx = 5318

encoded_pos_acts_BLF, encoded_neg_acts_BLF = get_paired_activations(
    model, submodule, sae, pos_input, neg_input, feature_idx
)

# %%


def few_shot_input(
    demo_idxs: list[int],
    demo_inputs: list[str],
    tokenizer: AutoTokenizer,
) -> tuple[torch.Tensor, list[int]]:
    assert len(demo_idxs) == len(demo_inputs)

    question = "Can you write me a sentence that relates to the word 'X'?"

    messages = []
    for demo_idx, demo_input in zip(demo_idxs, demo_inputs):
        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"A sentence that relates to the word is: {demo_input}",
            }
        )

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )
    messages.append(
        {"role": "assistant", "content": "A sentence that relates to the word is: "}
    )

    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        # add_generation_prompt=True,
        continue_final_message=True,
    )

    tokenized_input = tokenizer(
        formatted_input, return_tensors="pt", add_special_tokens=False
    ).to(model.device)

    positions = [
        i
        for i, a in enumerate(
            tokenizer.encode(formatted_input, add_special_tokens=False)
        )
        if tokenizer.decode([a]) == "X"
    ]

    assert len(positions) == len(demo_idxs) + 1

    return tokenized_input, positions


demo_idxs = [1835, 5318]
demo_inputs = ["I traveled back in time.", "What do we know?"]

input, positions = few_shot_input(demo_idxs, demo_inputs, tokenizer)

print(positions)

# %%

new_feature_idx = 6941
new_feature_idx = 7159
new_feature_idx = 14070

all_features = [sae.W_enc[:, i] for i in demo_idxs]
all_features.append(sae.W_enc[:, new_feature_idx])


all_features = [sae.W_dec[i] for i in demo_idxs]
all_features.append(sae.W_dec[new_feature_idx])

hook_fn = get_activation_addition_output_hook(
    all_features, [2.0] * len(all_features), positions
)

context_manager = add_hook(submodule, hook_fn)

with context_manager:
    output = model.generate(input["input_ids"], **generation_kwargs)

print(tokenizer.decode(output[0]))

# %%

output_v2 = model.generate(
    **orig_input,
    **generation_kwargs,
    max_new_tokens=100,
    output_logits=True,
    return_dict_in_generate=True,
)


logits_v2 = torch.stack(output_v2.logits)

mean_diff = (logits_v2 - ORIG_LOGITS).abs().mean()
max_diff = (logits_v2 - ORIG_LOGITS).abs().max()
print(f"mean_diff: {mean_diff.item()}, max_diff: {max_diff.item()}")

ORIG_PROBS = torch.softmax(ORIG_LOGITS, dim=-1)
PROBS_V2 = torch.softmax(logits_v2, dim=-1)

mean_diff = (PROBS_V2 - ORIG_PROBS).abs().mean()
max_diff = (PROBS_V2 - ORIG_PROBS).abs().max()
print(f"mean_diff: {mean_diff.item()}, max_diff: {max_diff.item()}")


assert torch.allclose(ORIG_LOGITS, logits_v2)


# %%
feature_idx = 6941
sae_input = "The quick brown fox jumps over the lazy dog. "
# sae_input = "I traveled back in time to the past."
# sae_input = "What are you doing?"

sae_input = tokenizer.apply_chat_template(
    [{"role": "user", "content": sae_input}], tokenize=False, add_generation_prompt=True
)

# sae_input = """<bos><start_of_turn>user
# Can you write me a sentence that relates to the word 'X'?<end_of_turn>
# <start_of_turn>model
# A sentence that relates to the word is: I traveled back in time.<end_of_turn>"""

sae_input = tokenizer(sae_input, return_tensors="pt", add_special_tokens=False).to(
    device
)

acts_BLD = model_utils.collect_activations(model, submodule, sae_input)
print(acts_BLD.shape)

encoded_acts_BLF = sae.encode(acts_BLD)
print(encoded_acts_BLF.shape)

print(encoded_acts_BLF[:, :, feature_idx])

decoded_acts_BLD = sae.decode(encoded_acts_BLF)
print(decoded_acts_BLD.shape)

# %%
l0_BL = (encoded_acts_BLF > 0).sum(dim=-1)
print(
    l0_BL[0, :10],
    "As we can see, the L0 norm is very high for the first BOS token, so we'll skip it.",
)

mean_l0 = l0_BL[:, 1:].float().mean()
print(f"mean l0: {mean_l0.item()}")

total_variance = torch.var(acts_BLD[:, 1:], dim=1).sum()
residual_variance = torch.var(acts_BLD[:, 1:] - decoded_acts_BLD[:, 1:], dim=1).sum()
frac_variance_explained = 1 - residual_variance / total_variance
print(f"frac_variance_explained: {frac_variance_explained.item()}")

# %%
