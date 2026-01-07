import functools

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

RECAPTION_PROMPT = """
Task:
Convert the user input into a render-spec image prompt.

Guidelines:
- Preserve the user’s intent and objects; do not add new objects.
- Physical attributes only: form, material, texture, lighting, color, scale, depth, spatial layout.
- Use noun phrases or participles; avoid narration and finite verbs.
- Replace actions with static visual states.
- Avoid pronouns, vibe words, and repetition.
- When there is sufficient detail, naturally split into 2–3 sentences by visual focus.
- Comma-separated, attribute-dense clauses.
- Up to 3 sentences, must end with a period.

Example 1 (style only):
Group of dolphins swimming in clear water, streamlined bodies, smooth skin, gray-blue coloration. Sunlight filtering near the surface, soft reflections and ripples, gentle depth gradients. Tight group formation, implied motion through water displacement, open ocean setting.

Example 2 (style only):
Rain-soaked alley at night, wet cobblestone ground, textured stone surface, brick walls with deep shadows. Neon signs glowing in magenta and cyan, single streetlamp casting a sharp cone of light, reflections across puddles, faint mist haze. Low-angle perspective, narrow corridor framing, dim ambient sky glow.

Output only the rewritten prompt.

Prompt: """


@functools.lru_cache(maxsize=1)
def _load_model():
    processor = AutoProcessor.from_pretrained(MODEL_NAME, padding_side="left")
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    return processor, qwen


def reprompt(prompt: str) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt must not be empty.")
    processor, qwen = _load_model()
    prompt = RECAPTION_PROMPT + prompt + "\n"
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    ]
    inputs = processor.apply_chat_template(  # pyrefly:ignore
        messages,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        # truncation=True,
        return_tensors="pt",
        return_dict=True,
    ).to(qwen.device)
    with torch.inference_mode():
        generated_ids = qwen.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.85,
            top_k=30,
            max_new_tokens=120,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(  # pyrefly:ignore
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


if __name__ == "__main__":
    print(reprompt("Make an image of a dog on a big red ball."))
    print(
        reprompt(
            "An image of a neon sign, saying big dog, on the top of a building in the rain."
        )
    )
    print(reprompt("The worlds smallest fish."))
    print(reprompt("A pretty couple holding hands."))
