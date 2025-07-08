from diffusers import StableDiffusionPipeline
from peft import PeftModel

# Load base model
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Load your trained LoRA
pipeline.unet = PeftModel.from_pretrained(
    pipeline.unet, 
    "./outputs/final_model"
)

# Generate images with your calligraphy style
image = pipeline("d").images[0]