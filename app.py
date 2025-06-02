import cv2
import torch
import numpy as np
from PIL import Image
import gradio as gr
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
import warnings

warnings.filterwarnings("ignore")

# After testing, I found that using a more reliable model like Stable Diffusion v1.5 works better for cartoonization tasks.
model_id = "runwayml/stable-diffusion-v1-5" 
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    safety_checker=None,
    requires_safety_checker=False
).to(device)

# Using a faster but reliable scheduler for enhanced performance
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

def apply_cartoon_effect(img, edge_preserve=True):
    """Optimized cartoon effect preprocessing"""
    img_np = np.array(img)
    img_float = img_np.astype(np.float32) / 255.0
    
    if edge_preserve:
        # Bilateral filtering with optimized parameters for better edge preservation
        smooth = cv2.bilateralFilter(img_float, 5, 0.5, 5)
    else:
        # Gaussian blur alternative
        smooth = cv2.GaussianBlur(img_float, (3, 3), 0)
    
    # Edge detection
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 3)
    edges = cv2.adaptiveThreshold(gray, 255, 
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 5, 2)
    
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    
    # Combine with edge boost so that edges are more pronounced. This  helped me in getting better results
    cartoon = smooth * (1.0 - edges) + edges * 0.7
    cartoon = (np.clip(cartoon, 0, 1) * 255).astype(np.uint8)
    
    return Image.fromarray(cartoon)

def cartoonize_image(input_img, strength=0.5, style="Comic Book", use_preprocess=True):
    """Robust image cartoonization"""
    try:
        img = Image.fromarray(input_img).convert('RGB')
        
        # Maintain aspect ratio with 512px as base (faster than 768) as it reduces processing time
        width, height = img.size
        max_dim = max(width, height)
        scale = 512 / max_dim
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.LANCZOS)
        
        preprocessed = apply_cartoon_effect(img) if use_preprocess else img
        
        # Style prompts that work well with SD v1.5, so that the model can generate better results as per the input style
        style_prompts = {
            "Comic Book": "comic book style, bold outlines, vibrant colors, halftone pattern",
            "Anime": "anime style, cel-shading, vibrant colors, expressive eyes",
            "Watercolor": "watercolor painting, soft edges, artistic brush strokes",
            "3D Cartoon": "3d render, pixar style, soft lighting, cartoon shading",
            "Sketch": "pencil sketch, black and white, detailed line art",
            "Oil Painting": "oil painting, thick brush strokes, impasto style"
        }
        
        negative_prompt = (
            "blurry, noisy, grainy, deformed, bad anatomy, text, watermark"
        )
        
        # Generate with optimized parameters to balance speed and quality 
        result = pipe(
            prompt=f"high quality {style_prompts[style]}, detailed, 4k",
            negative_prompt=negative_prompt,
            image=preprocessed,
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=25, 
            generator=torch.Generator(device=device).manual_seed(np.random.randint(0, 10000))
        ).images[0]
        
        return result.resize(img.size, Image.LANCZOS)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return None


with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# Your image Transformer")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload Photo", type="numpy")
            style = gr.Dropdown(
                ["Comic Book", "Anime", "Watercolor", "3D Cartoon", "Sketch", "Oil Painting"],
                value="Comic Book",
                label="Art Style"
            )
            strength = gr.Slider(0.3, 0.8, value=0.5, label="Transformation Strength")
            btn = gr.Button("Transform", variant="primary")
        
        with gr.Column():
            img_output = gr.Image(label="Result")
            gr.Markdown("Note: First run may take longer as models download")
    
    btn.click(
        cartoonize_image,
        inputs=[img_input, strength, style],
        outputs=img_output
    )

if __name__ == "__main__":
    app.launch(server_port=7860, share=False)