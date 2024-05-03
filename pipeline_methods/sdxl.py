import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import PNDMScheduler
from diffusers import DPMSolverMultistepScheduler

def generate_image():
    # Load the model

    prompts = ["Asian woman, light skin, late 20s, slim build, oval face, medium straight brown hair, brown almond eyes, straight eyebrows, average lips, realistic, taken with EOS R 300mm f2.8, real photo, 4k resolution, ar 9:16, v 6, looking to left", "Asian woman in contemplative pose, light skin, late 20s, oval face, brown straight hair, medium brown almond eyes, straight eyebrows, flat cheeks, average lips with subtle cupid's bow, rounded chin, slim build, taken with EOS R 300mm f2.8, real photo, 4k resolution, ar 9:16", "Asian woman, light skin, late 20s, oval face, brown straight hair, almond brown eyes, straight eyebrows, average lips, slim build, happy setting, taken with EOS R 300mm f2.8, real photo, 4k resolution, ar 9:16, v 6.", "Asian woman, light skin, late 20s, oval face, brown straight hair, medium straight eyebrows, almond-shaped brown eyes, straight nose, flat cheeks, average lips with subtle cupid's bow, rounded chin, slim build, taken with EOS R 300mm f2.8, real photo, 4k resolution, ar 9:16", "Asian woman in urban setting, light skin, late 20s, oval face, brown almond eyes, straight medium brown hair, slim build, taken with EOS R 300mm f2.8, real photo, 4k resolution, ar 9:16, v 6, subtle makeup, casual appearance, realistic photo", "Asian woman in a formal setting, light skin, late 20s, sleek straight brown hair, oval face, medium brown almond eyes, average straight eyebrows, medium straight nose, average lips with subtle cupid's bow, rounded chin, slim build, taken with EOS R 300mm f2.8, real photo, 4k resolution, ar 9:16", "Asian woman, light skin, late 20s, oval face, brown straight hair covering ears, brown almond eyes, slim build, taken with EOS R 300mm f2.8, real photo, 4k resolution, ar 9:16, v 6, subtle makeup, casual setting.", 
]

    negative_prompt = "Disfigured, Cross-eyed, 2d, cartoon, stylized, bad photo, bad lighting, high production value, unnatural studio lighting, commercial photoshoot, photoshopped, terrible photo, disfigured"
    
    guidance_scale = 7.0
    # scheduler = DPMSolverMultistepScheduler(use_lu_lambdas=True, euler_at_final=True)
    # print(scheduler)

    steps = 50

    
    
    pipe = StableDiffusionXLPipeline.from_single_file("models/sdxl.safetensors")

    pipe = pipe.to("cuda")
    
    
    for i, prompt in enumerate(prompts):
        print(i, " down")
    # Generate an image
        with torch.no_grad():
            image = pipe(prompt, height=1024, width=1024, num_inference_steps=steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt, num_images_per_prompt=1).images[0]
            
        # Save the generated image
        image.save(f"/home/apramey/FaceAugment/generated_images/generated_ya/generated_ya2_{i}.png")


generate_image()