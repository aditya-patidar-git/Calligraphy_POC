# %%writefile /kaggle/working/trainer.py
#!/usr/bin/env python3
# Cell 5: Training Loop and Optimization


import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
import wandb
from datetime import datetime
import math
import matplotlib.pyplot as plt
from model_setup import TrainingConfig

class CalligraphyTrainer:
    """Main trainer class for calligraphy LoRA training"""
    
    def __init__(self, config: TrainingConfig, models: dict):
        self.config = config
        self.models = models
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="tensorboard",
            project_dir=config.logging_dir
        )
        
        # Extract models
        self.tokenizer = models["tokenizer"]
        self.text_encoder = models["text_encoder"]
        self.vae = models["vae"]
        self.unet = models["unet"]
        self.scheduler = models["scheduler"]
        
        # Setup weight dtype
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        
        # Move models to correct dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        
        # Training metrics
        self.training_losses = []
        self.validation_losses = []
        self.global_step = 0
        
        logger.info("Trainer initialized successfully")
    
    def setup_optimizer_and_scheduler(self, train_dataloader):
        """Setup optimizer and learning rate scheduler"""
        
        # Setup optimizer
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            self.unet.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon
        )
        
        # Calculate total training steps
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.config.gradient_accumulation_steps
        )
        max_train_steps = self.config.max_epochs * num_update_steps_per_epoch
        
        # Setup scheduler
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=max_train_steps
        )
        
        # Prepare with accelerator
        self.unet, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.unet, optimizer, train_dataloader, lr_scheduler
        )
        
        return optimizer, lr_scheduler, max_train_steps
    
    def encode_prompt(self, prompt):
        """Encode text prompt to embeddings"""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.accelerator.device)
            )[0]
        
        return text_embeddings
    
    def compute_loss(self, batch):
        """Compute training loss"""
        # Get the text embedding for conditioning
        encoder_hidden_states = self.encode_prompt(batch["text"])
        
        # Convert images to latent space
        latents = self.vae.encode(batch["pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        # Sample noise to add to the latents
        noise = torch.randn_like(latents)
        if self.config.noise_offset:
            # Add noise offset for better training stability
            noise += self.config.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )
        
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()
        
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        
        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        if self.config.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        
        return loss
    
    def compute_snr(self, timesteps):
        """Compute SNR for loss weighting"""
        alphas_cumprod = self.scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
        
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        
        # Compute SNR.
        snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
        return snr
    
    def validate(self, val_dataloader):
        """Run validation"""
        self.unet.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                loss = self.compute_loss(batch)
                val_losses.append(loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        self.validation_losses.append(avg_val_loss)
        
        self.unet.train()
        return avg_val_loss
    
    def generate_validation_images(self, validation_prompts):
        """Generate validation images to monitor training progress"""
        from diffusers import StableDiffusionPipeline
        
        # Create pipeline with current state
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.model_name,
            unet=self.accelerator.unwrap_model(self.unet),
            text_encoder=self.text_encoder,
            vae=self.vae,
            scheduler=self.scheduler,
            tokenizer=self.tokenizer,
            torch_dtype=self.weight_dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        
        images = []
        for prompt in validation_prompts:
            with torch.autocast("cuda"):
                image = pipeline(
                    prompt,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=self.config.resolution,
                    width=self.config.resolution
                ).images[0]
            images.append(image)
        
        # Clean up
        del pipeline
        torch.cuda.empty_cache()
        
        return images
    
    def save_checkpoint(self, epoch, step, save_path):
        """Save model checkpoint"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        self.accelerator.unwrap_model(self.unet).save_pretrained(save_path)
        
        # Save training state
        checkpoint = {
            'epoch': epoch,
            'global_step': step,
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, save_path / "training_state.pt")
        logger.info(f"Checkpoint saved to {save_path}")
    
    def plot_training_progress(self):
        """Plot training and validation losses"""
        if len(self.training_losses) > 0:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.training_losses)
            plt.title('Training Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.grid(True)
            
            if len(self.validation_losses) > 0:
                plt.subplot(1, 2, 2)
                plt.plot(self.validation_losses)
                plt.title('Validation Loss')
                plt.xlabel('Validation Step')
                plt.ylabel('Loss')
                plt.grid(True)
            
            plt.tight_layout()
            plt.show()
    
    def train(self, train_dataloader, val_dataloader=None):
        """Main training loop"""
        
        # Setup optimizer and scheduler
        optimizer, lr_scheduler, max_train_steps = self.setup_optimizer_and_scheduler(train_dataloader)
        
        # Create output directories
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation prompts for monitoring
        validation_prompts = [
            f"a {self.config.model_name.split('/')[-1]} style calligraphy letter 'a', black ink on white background",
            f"a {self.config.model_name.split('/')[-1]} style calligraphy word 'hello', black ink on white background",
            f"a {self.config.model_name.split('/')[-1]} style calligraphy letter 'g', black ink on white background",
            f"a {self.config.model_name.split('/')[-1]} style calligraphy word 'world', black ink on white background"
        ]
        
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {self.config.max_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.config.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.config.batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps}")
        logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        
        self.global_step = 0
        first_epoch = 0
        print("🔥 Training loop entered")
        print(f"📦 Epoch count: {self.config.max_epochs}")
        # Training loop
        for epoch in range(first_epoch, self.config.max_epochs):
            self.unet.train()
            train_loss = 0.0
            
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # Compute loss
                    loss = self.compute_loss(batch)
                    
                    # Gather the losses across all processes for logging
                    avg_loss = self.accelerator.gather(loss.repeat(self.config.batch_size)).mean()
                    train_loss += avg_loss.item() / self.config.gradient_accumulation_steps
                    
                    # Backpropagate
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), self.config.max_grad_norm)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    self.training_losses.append(train_loss)
                    
                    # Log training progress
                    if self.global_step % 50 == 0:
                        logger.info(f"Epoch {epoch}, Step {self.global_step}, Loss: {train_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                    
                    # Validation
                    if val_dataloader and self.global_step % self.config.validation_steps == 0:
                        val_loss = self.validate(val_dataloader)
                        logger.info(f"Validation Loss: {val_loss:.4f}")
                        
                        # Generate validation images
                        try:
                            val_images = self.generate_validation_images(validation_prompts[:2])  # Generate 2 images
                            # Display images
                            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                            for i, img in enumerate(val_images):
                                axes[i].imshow(img)
                                axes[i].set_title(f"Step {self.global_step}: {validation_prompts[i]}")
                                axes[i].axis('off')
                            plt.tight_layout()
                            plt.show()
                        except Exception as e:
                            logger.warning(f"Failed to generate validation images: {e}")
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        save_path = output_dir / f"checkpoint-{self.global_step}"
                        self.save_checkpoint(epoch, self.global_step, save_path)
                    
                    train_loss = 0.0
                
                if self.global_step >= max_train_steps:
                    break
            
            # End of epoch logging
            if val_dataloader:
                val_loss = self.validate(val_dataloader)
                logger.info(f"End of Epoch {epoch} - Validation Loss: {val_loss:.4f}")
        
        # Save final model
        final_save_path = output_dir / "final_model"
        self.save_checkpoint(self.config.max_epochs, self.global_step, final_save_path)
        
        # Plot training progress
        self.plot_training_progress()
        
        logger.info("Training completed!")
        return final_save_path

def create_trainer(config, models):
    """Factory function to create trainer"""
    return CalligraphyTrainer(config, models)

print("Training loop and optimization loaded successfully!")
print("Ready to start training your calligraphy LoRA model.")