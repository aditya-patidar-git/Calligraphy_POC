# %%writefile main.py
#!/usr/bin/env python3
"""
Main training script for calligraphy LoRA model 
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import json
import gc

# For Jupyter/Colab - set arguments programmatically
sys.argv = [
    'main.py',
    '--data_dir', '/kaggle/input/character-simple-images',
    '--output_dir', './outputs',
    '--num_epochs', '30',
    '--batch_size', '1',
    '--lora_rank', '16',
    '--lora_alpha', '32',
    '--learning_rate', '1e-4',
    '--gradient_accumulation_steps', '4',
    '--save_steps', '500',
    '--validation_steps', '100',
    '--mixed_precision', 'fp16',
    '--use_8bit_adam',
    '--resolution', '512'
]

from dataset import BrushLetteringDataset
from utils import setup_logging, save_checkpoint, load_checkpoint, compute_snr
from config import TrainingConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for brush lettering generation")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing character images")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for checkpoints")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5", help="Base SD model")
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--validation_steps", type=int, default=100, help="Run validation every N steps")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam optimizer")
    
    return parser.parse_args()

class LoRATrainer:
    def __init__(self, config):
        print("Initializing LoRA Trainer...")
        self.config = config
        self.setup_models()
        self.setup_optimizer()
        self.setup_scheduler()
        
    def setup_models(self):
        """Initialize and setup models with LoRA"""
        print(f"Loading base model: {self.config.model_name}")
        
        # Load the pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.mixed_precision == "fp16" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Extract components
        self.vae = self.pipeline.vae
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler
        
        # Freeze base models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        # Setup LoRA for UNet
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=[
                "to_k", "to_q", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2"
            ],
            lora_dropout=0.1,
        )
        
        self.unet = get_peft_model(self.unet, lora_config)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(device)
        self.text_encoder.to(device)
        self.unet.to(device)
        
        print(f"Models loaded and moved to {device}")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.unet.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    def setup_optimizer(self):
        """Setup optimizer"""
        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
            except ImportError:
                raise ImportError("Please install bitsandbytes: pip install bitsandbytes")
        else:
            optimizer_cls = torch.optim.AdamW
        
        self.optimizer = optimizer_cls(
            self.unet.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )
    
    def setup_scheduler(self):
        """Setup noise scheduler"""
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.model_name, 
            subfolder="scheduler"
        )
    
    def encode_prompt(self, prompt):
        """Encode text prompt"""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.text_encoder.device)
            )[0]
        
        return text_embeddings
    
    def training_step(self, batch):
        """Single training step"""
        images = batch["images"].to(self.unet.device)
        prompts = batch["prompts"]
        
        # FIX: Ensure images match VAE dtype
        if self.config.mixed_precision == "fp16":
            images = images.to(torch.float16)
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=latents.device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Encode prompts
        encoder_hidden_states = self.encode_prompt(prompts)
        
        # Predict noise
        model_pred = self.unet(
            noisy_latents, 
            timesteps, 
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # Compute loss
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        return loss
    
    def validation_step(self, val_dataloader):
        """Validation step"""
        self.unet.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                loss = self.training_step(batch)
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches >= 10:  # Limit validation batches
                    break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.unet.train()
        return avg_loss
    
    def train(self, train_dataloader, val_dataloader=None):
        """Main training loop"""
        print("Starting training...")
        
        global_step = 0
        epoch_losses = []
        
        # Setup mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision == "fp16" else None
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision == "fp16"):
                    loss = self.training_step(batch)
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item()
                
                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                        'step': global_step
                    })
                    
                    # Validation
                    if val_dataloader and global_step % self.config.validation_steps == 0:
                        val_loss = self.validation_step(val_dataloader)
                        print(f"\nValidation loss: {val_loss:.4f}")
                    
                    # Save checkpoint
                    if global_step % self.config.save_steps == 0:
                        self.save_checkpoint(global_step, epoch, loss.item())
                
                # Clean up memory
                if step % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(global_step, epoch, avg_epoch_loss, is_epoch_end=True)
        
        print("Training completed!")
        return epoch_losses
    
    def save_checkpoint(self, step, epoch, loss, is_epoch_end=False):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        if is_epoch_end:
            lora_path = checkpoint_dir / f"lora_epoch_{epoch}"
        else:
            lora_path = checkpoint_dir / f"lora_step_{step}"
        
        self.unet.save_pretrained(lora_path)
        
        # Save training state
        state_dict = {
            'step': step,
            'epoch': epoch,
            'loss': loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }
        
        state_path = lora_path.parent / f"training_state_{step}.pt"
        torch.save(state_dict, state_path)
        
        print(f"Checkpoint saved: {lora_path}")

def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(args.output_dir)
    
    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        validation_steps=args.validation_steps,
        mixed_precision=args.mixed_precision,
        use_8bit_adam=args.use_8bit_adam
    )
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = BrushLetteringDataset(
        data_dir=config.data_dir,
        resolution=config.resolution,
        split='train'
    )
    
    val_dataset = BrushLetteringDataset(
        data_dir=config.data_dir,
        resolution=config.resolution,
        split='val'
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    ) if len(val_dataset) > 0 else None
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset) if val_dataset else 0}")
    
    # Create trainer and start training
    trainer = LoRATrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume_from:
        load_checkpoint(trainer, args.resume_from)
    
    # Train the model
    losses = trainer.train(train_dataloader, val_dataloader)
    
    # Save final model
    final_path = Path(config.output_dir) / "final_model"
    trainer.unet.save_pretrained(final_path)
    print(f"Final model saved to: {final_path}")

if __name__ == "__main__":
    main()