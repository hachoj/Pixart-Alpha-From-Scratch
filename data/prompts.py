caption_prompt = """
Directives:
1. Render spec only: declare what must be rendered, not a description of events.
2. Physical only: geometry, materials, surface texture, lighting, color, depth, camera framing.
3. Prompt grammar: noun phrases or participles; minimize verbs, no is/are/was/were.
4. Dense syntax: comma-chained clauses, each adding a distinct visual attribute.
5. Start with the primary subject; add secondary subjects as separate clauses or sentences.

Constraints:
- No meta phrasing or narration.
- No abstractions, emotions, intent, or vibe adjectives.
- No redundancy: do not restate the same attribute; repeating necessary entities is allowed.
- 2–3 sentences when detail allows, must end with a period.

Example (style only, do not copy content):
Stone archway rising from desert sand, weathered limestone blocks, chipped edges, granular surface texture. Low-angle sunlight, long hard shadows, warm ochre palette. Distant haze, sparse scrub vegetation, deep blue sky.
"""

recaption_prompt = """
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

Prompt: Make an image of a reddish-brown monkey sitting on a tree trunk in a green forest.
"""
# Prompt: Make an image of a rain-soaked neon-lit alley at night, wet cobblestone ground reflecting magenta and cyan signs, faint mist in the air, a single streetlamp casting a sharp cone of light, deep shadows between brick walls.
# Prompt: Make an image of a group of dolphins swimming in the water.

# Prompt: Make an image of a vast field of oversized mushrooms growing in a forest.
# Prompt: Make an image of a medieval stone castle built on a rocky hilltop, tall towers and thick walls overlooking a small village below, with winding dirt roads, distant mountains on the horizon, and soft afternoon light casting long shadows across the landscape.

