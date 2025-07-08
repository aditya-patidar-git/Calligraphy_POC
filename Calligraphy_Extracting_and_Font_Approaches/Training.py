# Google Colab Setup for Stable Diffusion Training with Custom Calligraphy Dataset
# Run this in Google Colab (GPU runtime)

# ========================================================================================
# PART 1: Environment Setup and Dependencies
# ========================================================================================

# Install required packages
!pip install diffusers transformers accelerate xformers
!pip install datasets pillow numpy torch torchvision
!pip install wandb  # For experiment tracking (optional)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from pathlib import Path
import pickle
from typing import Optional, Union, List
from dataclasses import dataclass

from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDPMScheduler
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ========================================================================================
# PART 2: Custom Dataset Class for Calligraphy Images
# ========================================================================================

class CalligraphySDDataset(Dataset):
    """
    Custom dataset for Stable Diffusion training with calligraphy images
    Combines your extracted features with image data
    """
    
    def __init__(
        self,
        image_paths: List[str],
        features: np.ndarray,
        tokenizer,
        size: int = 512,
        center_crop: bool = True,
        feature_conditioning: bool = True
    ):
        self.image_paths = image_paths
        self.features = features
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.feature_conditioning = feature_conditioning
        
        # Create simple text prompts based on image characteristics
        self.prompts = self._generate_prompts()
    
    def _generate_prompts(self):
        """Generate descriptive prompts for each calligraphy image"""
        prompts = []
        for i, path in enumerate(self.image_paths):
            # Extract class/style info from path if available
            class_name = Path(path).parent.name
            
            # Create descriptive prompt
            if 'ellipse' in Path(path).name:
                style = "elliptical calligraphy"
            elif 'curve' in Path(path).name:
                style = "curved calligraphy"
            else:
                style = "calligraphy"
            
            prompt = f"beautiful {style} writing, {class_name} style, elegant handwriting"
            prompts.append(prompt)
        
        return prompts
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Resize and crop
        if self.center_crop:
            image = image.resize((self.size, self.size), Image.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1]
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Tokenize prompt
        prompt = self.prompts[idx]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Get feature conditioning
        feature_embedding = torch.from_numpy(self.features[idx]).float()
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids.squeeze(),
            "feature_conditioning": feature_embedding
        }

# ========================================================================================
# PART 3: Custom Stable Diffusion Model with Feature Conditioning
# ========================================================================================

class FeatureConditionedUNet(torch.nn.Module):
    """
    Modified UNet that accepts both text and feature conditioning
    """
    
    def __init__(self, original_unet, feature_dim: int, conditioning_dim: int = 768):
        super().__init__()
        self.unet = original_unet
        self.feature_dim = feature_dim
        self.conditioning_dim = conditioning_dim
        
        # Feature projection layer
        self.feature_projector = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, conditioning_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(conditioning_dim, conditioning_dim)
        )
        
        # Freeze original UNet initially (optional)
        # for param in self.unet.parameters():
        #     param.requires_grad = False
    
    def forward(self, sample, timestep, encoder_hidden_states, feature_conditioning=None):
        # Project features if provided
        if feature_conditioning is not None:
            projected_features = self.feature_projector(feature_conditioning)
            
            # Combine with text embeddings
            if encoder_hidden_states is not None:
                # Simple concatenation approach
                combined_conditioning = torch.cat([
                    encoder_hidden_states,
                    projected_features.unsqueeze(1).expand(-1, encoder_hidden_states.shape[1], -1)
                ], dim=-1)
            else:
                combined_conditioning = projected_features.unsqueeze(1)
        else:
            combined_conditioning = encoder_hidden_states
        
        return self.unet(sample, timestep, combined_conditioning)

# ========================================================================================
# PART 4: Training Configuration and Setup
# ========================================================================================

@dataclass
class TrainingConfig:
    # Model parameters
    model_name: str = "runwayml/stable-diffusion-v1-5"
    resolution: int = 512
    
    # Training parameters
    batch_size: int = 1  # Small batch for Colab
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    num_epochs: int = 10
    save_steps: int = 500
    
    # Paths
    output_dir: str = "./calligraphy-sd-model"
    logging_dir: str = "./logs"
    
    # Feature conditioning
    use_feature_conditioning: bool = True
    feature_weight: float = 0.1

def setup_models_and_data(config: TrainingConfig, sd_conditioning_file: str):
    """
    Setup all models, tokenizer, and data for training
    """
    # Load Stable Diffusion components
    tokenizer = CLIPTokenizer.from_pretrained(config.model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.model_name, subfolder="unet")
    
    # Load your conditioning data
    conditioning_data = np.load(sd_conditioning_file, allow_pickle=True).item()
    features = conditioning_data['features']
    image_paths = conditioning_data['image_paths']
    feature_dim = conditioning_data['feature_dim']
    
    print(f"Loaded {len(image_paths)} images with {feature_dim}D features")
    
    # Create feature-conditioned UNet
    if config.use_feature_conditioning:
        unet = FeatureConditionedUNet(unet, feature_dim)
    
    # Create dataset
    dataset = CalligraphySDDataset(
        image_paths=image_paths,
        features=features,
        tokenizer=tokenizer,
        size=config.resolution
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    return tokenizer, text_encoder, vae, unet, dataloader, dataset

# ========================================================================================
# PART 5: Training Loop
# ========================================================================================

def train_stable_diffusion(config: TrainingConfig, sd_conditioning_file: str):
    """
    Main training function
    """
    # Setup accelerator for distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=config.logging_dir
    )
    
    # Setup models and data
    tokenizer, text_encoder, vae, unet, dataloader, dataset = setup_models_and_data(
        config, sd_conditioning_file
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Setup noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_name, subfolder="scheduler")
    
    # Prepare everything with accelerator
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    
    # Move models to device
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    
    # Set models to appropriate modes
    text_encoder.eval()
    vae.eval()
    unet.train()
    
    # Training loop
    global_step = 0
    
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{config.num_epochs}"
        )
        
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Get batch data
                pixel_values = batch["pixel_values"]
                input_ids = batch["input_ids"]
                feature_conditioning = batch.get("feature_conditioning")
                
                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * 0.18215
                
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                ).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]
                
                # Predict noise
                if config.use_feature_conditioning and hasattr(unet.module, 'feature_projector'):
                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                        feature_conditioning=feature_conditioning
                    )
                else:
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states)
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise)
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress
            progress_bar.update(1)
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            
            global_step += 1
            
            # Save checkpoint
            if global_step % config.save_steps == 0:
                save_path = f"{config.output_dir}/checkpoint-{global_step}"
                accelerator.save_state(save_path)
        
        progress_bar.close()
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = StableDiffusionPipeline.from_pretrained(
            config.model_name,
            unet=unet,
            torch_dtype=torch.float16
        )
        pipeline.save_pretrained(config.output_dir)
        print(f"Model saved to {config.output_dir}")

# ========================================================================================
# PART 6: Usage Example
# ========================================================================================

# Configuration
config = TrainingConfig(
    batch_size=1,
    num_epochs=5,
    learning_rate=1e-5,
    use_feature_conditioning=True
)

# Start training (run this after uploading your sd_conditioning.npy file)
# train_stable_diffusion(config, "sd_conditioning.npy")

print("Setup complete! Upload your sd_conditioning.npy file and run the training.")