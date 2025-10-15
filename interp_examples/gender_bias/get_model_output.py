# %%

import pickle
import json

# %%

# note: this file has cot in the output
# filename = "score_output_high_bar_cot/vllm_inference/Qwen_Qwen3-8B/score_results_v1_meta_job_description_Qwen_Qwen3-8B_1000_0_all.pkl"

filename = "score_output_high_bar/Qwen_Qwen3-8B/score_results_v2_meta_job_description_Qwen_Qwen3-8B_1000_0_all.pkl"

with open(filename, "rb") as f:
    data = pickle.load(f)
# %%
print(data.keys())
print(data["results"][0].keys())
print(data["bias_scores"])
# %%

print(data["results"][0])

# %%
import pandas as pd
from collections import defaultdict

# Group responses by unique resume
resume_responses = defaultdict(list)

for result in data["results"]:
    # Extract the resume key (everything after @gmail.com)
    resume_key = result["resume"].split("@gmail.com")[-1]
    response = result["response"]

    if "answer:" in response.lower():
        response = response.lower().split("answer:")[-1].strip()

    # Only include if response is not None
    if response is not None:
        resume_responses[resume_key].append(response)

# Find resumes where responses are not all the same
biased_resumes = {}
all_same_resumes = {}

for resume_key, responses in resume_responses.items():
    # Skip if we don't have exactly 4 responses (excluding None)
    if len(responses) == 4:
        # Check if all responses are the same
        if len(set(responses)) > 1:  # More than one unique response
            biased_resumes[resume_key] = responses
        else:
            all_same_resumes[resume_key] = responses


# Display results
print(f"Total unique resumes: {len(resume_responses)}")
print(f"Resumes with potential bias (different responses): {len(biased_resumes)}")
print("\nExamples of biased responses:")

for i, (resume_key, responses) in enumerate(list(biased_resumes.items())[:5]):
    print(f"\nResume {i + 1} (key: ...{resume_key[:50]}...):")
    print(f"  Responses: {responses}")

    print(f"  Yes count: {responses.count('yes')}, No count: {responses.count('no')}")

# Create a more detailed analysis
bias_patterns = defaultdict(int)
for responses in biased_resumes.values():
    pattern = f"{responses.count('yes')} yes, {responses.count('no')} no"
    bias_patterns[pattern] += 1

print("\n\nBias patterns distribution:")
for pattern, count in sorted(bias_patterns.items(), key=lambda x: x[1], reverse=True):
    print(f"  {pattern}: {count} resumes")

# If you want to create a new dataset with only the biased results
biased_results = []
for result in data["results"]:
    resume_key = result["resume"].split("@gmail.com")[-1]
    if resume_key in biased_resumes and result["response"] is not None:
        biased_results.append(result)

print(f"\n\nTotal results in biased dataset: {len(biased_results)}")

# Optional: Convert to DataFrame for easier analysis
df_biased = pd.DataFrame(biased_results)
# view the first 5 rows
print(df_biased.head())



# %%
# csv it
df_biased.to_csv("biased_results.csv", index=False)

# Export to JSONL in OpenAI conversation format
with open("biased_results.jsonl", "w") as f:
    for result in biased_results:
        conversation = [
            {"role": "system", "content": result["system_prompt"]},
            {"role": "user", "content": result["prompt"]},
            {"role": "assistant", "content": result["response"]}
        ]
        f.write(json.dumps(conversation) + "\n")


# do same for all same resumes (unbiased)
unbiased_results = []
for result in data["results"]:
    resume_key = result["resume"].split("@gmail.com")[-1]
    if resume_key in all_same_resumes and result["response"] is not None:
        unbiased_results.append(result)

print(f"\n\nTotal results in unbiased dataset: {len(unbiased_results)}")

# Convert to DataFrame
df_unbiased = pd.DataFrame(unbiased_results)
print(df_unbiased.head())

# %%
# csv it
df_unbiased.to_csv("unbiased_results.csv", index=False)

# Export to JSONL in OpenAI conversation format
with open("unbiased_results.jsonl", "w") as f:
    for result in unbiased_results:
        conversation = [
            {"role": "system", "content": result["system_prompt"]},
            {"role": "user", "content": result["prompt"]},
            {"role": "assistant", "content": result["response"]}
        ]
        f.write(json.dumps(conversation) + "\n")
