import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

iface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="Enter a prompt for image generation"),
    outputs=gr.Image(type="pil"),
    title="Text-to-Image Generator",
    description="Generate an image from text using Stable Diffusion."
)

iface.launch()
