import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-xl-xl-ul2")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/t5gemma-xl-xl-ul2",
    device_map="auto",
)

input_text = "Write me a poem about Machine Learning. Answer:"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))

# Then something something model surgery, shouldn't be too hard