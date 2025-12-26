import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-xl-xl-ul2")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/t5gemma-xl-xl-ul2",
    device_map="auto",
)

input_texts = [
    "The easiest way to make a big pancake is ",
    "this is some other crap text",
]
input_ids = tokenizer(
    input_texts, padding=True, truncation=True, max_length=100, return_tensors="pt"
).to("cuda")

torch_model = model.model
print(torch_model.encoder(**input_ids).last_hidden_state.shape)