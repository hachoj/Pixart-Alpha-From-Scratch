import gradio as gr
import torch
from generate_stage1 import generate as generate_stage1
from generate_stage2 import generate as generate_stage2
from reprompt import reprompt


MODEL_PATH_STAGE1 = "checkpoints_stage1/transformer_engine_model.pt"
MODEL_PATH_STAGE2 = "checkpoints_stage1/reparameterized_transformer_engine_model.pt"


def generate_image(mode, class_label, text_prompt, seed, use_seed):
    """Generate image from class label or text prompt."""
    try:
        # Use seed if checkbox is checked
        actual_seed = seed if use_seed else None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if mode == "Class-Conditioned":
            # Stage 1: Class-conditioned generation
            if not 0 <= class_label <= 1000:
                return None, None, "Class label must be between 0 and 1000"
            
            timesteps, final_image = generate_stage1(
                model_path=MODEL_PATH_STAGE1,
                class_label=class_label,
                seed=actual_seed,
                device=device,
            )
            reprompted_text = None
            
        else:  # Text-Conditioned
            # Stage 2: Text-conditioned generation
            if not text_prompt.strip():
                return None, None, "Please enter a text prompt"
            
            # Reprompt the text
            reprompted_text = reprompt(text_prompt)
            
            timesteps, final_image = generate_stage2(
                model_path=MODEL_PATH_STAGE2,
                prompt=reprompted_text,
                seed=actual_seed,
                device=device,
            )
        
        # Convert to numpy for Gradio (HWC format, 0-255)
        final_image_np = (
            final_image.permute(1, 2, 0).float().cpu().numpy() * 255.0
        ).astype("uint8")
        
        status = "Generation successful!"
        if reprompted_text:
            status = f"Reprompted: {reprompted_text}\n\n{status}"
        
        return final_image_np, reprompted_text, status
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        return None, None, error_msg


# Create Gradio interface
with gr.Blocks(title="PixArt-Alpha Generator") as demo:
    gr.Markdown("# PixArt-Alpha Image Generator")
    gr.Markdown("Generate images using class labels or text prompts")
    
    with gr.Row():
        with gr.Column():
            mode_radio = gr.Radio(
                choices=["Class-Conditioned", "Text-Conditioned"],
                value="Class-Conditioned",
                label="Generation Mode",
            )
            
            # Class-conditioned inputs
            class_input = gr.Slider(
                minimum=0,
                maximum=1000,
                step=1,
                value=2,
                label="Class Label",
                info="ImageNet-1K class (e.g., 2=shark, 281=cat, 388=giant panda)",
                visible=True,
            )
            
            # Text-conditioned inputs
            text_input = gr.Textbox(
                label="Text Prompt",
                placeholder="Enter a description of the image you want to generate...",
                lines=3,
                visible=False,
            )
            
            use_seed_checkbox = gr.Checkbox(
                label="Use fixed seed",
                value=False,
            )
            
            seed_input = gr.Number(
                label="Seed",
                value=42,
                precision=0,
            )
            
            generate_btn = gr.Button("Generate Image", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(
                label="Generated Image",
                type="numpy",
            )
            reprompted_output = gr.Textbox(
                label="Reprompted Text (Text-Conditioned Only)",
                interactive=False,
                visible=False,
            )
            status_text = gr.Textbox(label="Status", interactive=False)
    
    # Toggle visibility based on mode
    def update_visibility(mode):
        is_class = mode == "Class-Conditioned"
        return (
            gr.update(visible=is_class),      # class_input
            gr.update(visible=not is_class),  # text_input
            gr.update(visible=not is_class),  # reprompted_output
        )
    
    mode_radio.change(
        fn=update_visibility,
        inputs=[mode_radio],
        outputs=[class_input, text_input, reprompted_output],
    )
    
    # Connect button
    generate_btn.click(
        fn=generate_image,
        inputs=[mode_radio, class_input, text_input, seed_input, use_seed_checkbox],
        outputs=[output_image, reprompted_output, status_text],
    )


if __name__ == "__main__":
    demo.launch(share=True)
