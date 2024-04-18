import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stablediffusionapi/realism-engine-sdxl-v30")
pipeline = pipeline.to("cuda")

# Generate an image
with torch.no_grad():
    image = pipeline("dog at beach", num_inference_steps=1, guidance_scale=7.5).images[0]

    # Save the generated image
    image.save("generated_image.png")