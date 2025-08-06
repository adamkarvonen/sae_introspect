# %%

import pickle

from interp_tools.config import SelfInterpTrainingConfig

# %%
filename = "lora_eval_results_encoder_prefill.pkl"
filename = "lora_eval_results.pkl"

with open(filename, "rb") as f:
    results = pickle.load(f)

print(results["aggregated_metrics"])
# %%
config = results["config"]
print(config["prefill_original_sentences"])
training_filename = config["training_data_filename"]

training_filename = "gpt4o_contrastive_rewriting_results.pkl"

print(training_filename)

eval_features = config["eval_features"]
print(len(eval_features))

print(results.keys())
print(results["all_sentence_metrics"][0])

# %%
with open(training_filename, "rb") as f:
    training_data = pickle.load(f)

# %%

print(len(training_data))
print(training_data.keys())
print(training_data["results"][0].keys())

training_data_metrics = []

for result in training_data["results"]:
    if result["feature_idx"] in eval_features:
        training_data_metrics.append(result["sentence_metrics"][0])
        # for sentence_metric in result["sentence_metrics"]:
        #     training_data_metrics.append(sentence_metric)

# %%
print(len(training_data_metrics))
print(training_data_metrics[0])

zero_count = 0

for metric in training_data_metrics:
    if metric["original_max_activation"] == 0:
        zero_count += 1

print(f"{zero_count} / {len(training_data_metrics)}")

# %%
if training_data_metrics:
    aggregated_metrics = {}
    metric_keys = training_data_metrics[0].keys()
    for key in metric_keys:
        avg_val = sum(m[key] for m in training_data_metrics) / len(
            training_data_metrics
        )
        aggregated_metrics[key] = avg_val

print(aggregated_metrics)

# %%
print(results["aggregated_metrics"])

# %%

for i, feature_idx in enumerate(eval_features):
    print(f"Feature {i}: {feature_idx}")
    print(results["all_feature_results_this_eval_step"][i]["explanation"])

    for result in training_data["results"]:
        if result["feature_idx"] == feature_idx:
            print(result["explanation"])
            break
    break

# %%
