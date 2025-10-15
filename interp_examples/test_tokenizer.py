from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")


sentences = ["Hello, how are you?", "I am fine, thank you!", "How are you doing?"]
tokenizer.padding_side = "left"

out = tokenizer(
    sentences,
    return_tensors="pt",
    add_special_tokens=False,
    truncation=True,
    max_length=512,
    padding=True,  # Pad to same length for batching
)
# left pad NOT RIGHT PAD

# batch decode
text_reencoded = tokenizer.batch_decode(out.input_ids)
print(text_reencoded)
# see attn mask
mask = out.attention_mask
print(mask)
