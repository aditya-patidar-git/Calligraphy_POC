# %%writefile utils.py
#!/usr/bin/env python3
"""
Utility functions for brush lettering LoRA training
"""

import os
import torch
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import cv2
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

def setup_logging(output_dir: str, log_level: str = "INFO") -> None:
    """
    Setup logging configuration
    
    Args:
        output_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_file = output_path / "training.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    config: Dict[str, Any],
    checkpoint_dir: str,
    is_best: bool = False
) -> str:
    """
    Save training checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current step
        loss: Current loss
        config: Training configuration
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best checkpoint
    
    Returns:
        Path to saved checkpoint
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint_data = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }
    
    # Save checkpoint
    if is_best:
        checkpoint_file = checkpoint_path / "best_checkpoint.pt"
    else:
        checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch}_step_{step}.pt"
    
    torch.save(checkpoint_data, checkpoint_file)
    
    # Save latest checkpoint link
    latest_file = checkpoint_path / "latest_checkpoint.pt"
    torch.save(checkpoint_data, latest_file)
    
    logging.info(f"Checkpoint saved: {checkpoint_file}")
    return str(checkpoint_file)

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Load training checkpoint
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
    
    Returns:
        Dictionary with checkpoint information
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logging.info(f"Checkpoint loaded from: {checkpoint_path}")
    logging.info(f"Resuming from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
    
    return {
        'epoch': checkpoint['epoch'],
        'step': checkpoint['step'],
        'loss': checkpoint['loss'],
        'config': checkpoint.get('config', {})
    }

def compute_snr(timesteps, noise_scheduler):
    """
    Compute signal-to-noise ratio for timesteps
    Used for loss weighting in diffusion training
    
    Args:
        timesteps: Tensor of timesteps
        noise_scheduler: Diffusion noise scheduler
    
    Returns:
        SNR values for the timesteps
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    
    # Expand the tensors to match timesteps shape
    sqrt_alphas_cumprod = sqrt_alphas_cumprod[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    
    # Compute SNR
    snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
    return snr

def create_training_summary(
    losses: List[float],
    config: Dict[str, Any],
    output_dir: str,
    character_distribution: Optional[Dict[str, int]] = None
) -> None:
    """
    Create training summary with plots and statistics
    
    Args:
        losses: List of training losses
        config: Training configuration
        output_dir: Directory to save summary
        character_distribution: Distribution of characters in dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    axes[0, 0].plot(losses, linewidth=2, color='blue', alpha=0.7)
    axes[0, 0].set_title('Training Loss Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss smoothed (moving average)
    if len(losses) > 10:
        window_size = max(1, len(losses) // 20)
        smoothed_losses = []
        for i in range(len(losses)):
            start_idx = max(0, i - window_size)
            end_idx = min(len(losses), i + window_size + 1)
            smoothed_losses.append(np.mean(losses[start_idx:end_idx]))
        
        axes[0, 1].plot(smoothed_losses, linewidth=2, color='red', alpha=0.7)
        axes[0, 1].set_title(f'Smoothed Training Loss (window={window_size})')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Smoothed Loss')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Not enough data\nfor smoothing', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Smoothed Training Loss')
    
    # Plot 3: Character Distribution
    if character_distribution:
        chars = list(character_distribution.keys())
        counts = list(character_distribution.values())
        
        axes[1, 0].bar(chars, counts, color='green', alpha=0.7)
        axes[1, 0].set_title('Character Distribution in Dataset')
        axes[1, 0].set_xlabel('Characters')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, 'Character distribution\nnot available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Character Distribution')
    
    # Plot 4: Training Configuration
    config_text = "Training Configuration:\n\n"
    key_configs = [
        'learning_rate', 'batch_size', 'num_epochs', 
        'lora_rank', 'lora_alpha', 'resolution'
    ]
    
    for key in key_configs:
        if key in config:
            config_text += f"{key}: {config[key]}\n"
    
    axes[1, 1].text(0.1, 0.9, config_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_title('Configuration')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save plots
    summary_path = output_path / "training_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical summary
    summary_stats = {
        'final_loss': losses[-1] if losses else None,
        'min_loss': min(losses) if losses else None,
        'max_loss': max(losses) if losses else None,
        'avg_loss': np.mean(losses) if losses else None,
        'total_epochs': len(losses),
        'character_distribution': character_distribution,
        'config': config
    }
    
    with open(output_path / "training_summary.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logging.info(f"Training summary saved to {output_path}")

def visualize_samples(
    dataset,
    num_samples: int = 8,
    output_dir: str = "./sample_visualizations",
    title: str = "Dataset Samples"
) -> None:
    """
    Visualize samples from the dataset
    
    Args:
        dataset: Dataset to sample from
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
        title: Title for the visualization
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate grid size
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for i in range(num_samples):
        if i >= len(dataset):
            break
            
        sample = dataset[i]
        image = sample['images']
        character = sample['characters']
        prompt = sample['prompts']
        
        # Convert tensor to numpy for visualization
        if isinstance(image, torch.Tensor):
            # Denormalize from [-1, 1] to [0, 1]
            image_np = (image.permute(1, 2, 0).numpy() + 1) / 2
            image_np = np.clip(image_np, 0, 1)
        else:
            image_np = np.array(image)
        
        # Display image
        ax = axes[i] if num_samples > 1 else axes[0]
        ax.imshow(image_np, cmap='gray' if len(image_np.shape) == 2 else None)
        ax.set_title(f"'{character}'", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add prompt as text below image
        wrapped_prompt = '\n'.join([prompt[j:j+30] for j in range(0, len(prompt), 30)])
        ax.text(0.5, -0.1, wrapped_prompt, transform=ax.transAxes,
               ha='center', va='top', fontsize=8, wrap=True)
    
    # Hide remaining axes
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = output_path / f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Sample visualization saved to {viz_path}")

def preprocess_character_images(
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (512, 512),
    background_color: str = "white"
) -> Dict[str, List[str]]:
    """
    Preprocess character images for training
    
    Args:
        input_dir: Directory containing raw character images
        output_dir: Directory to save processed images
        target_size: Target size for processed images
        background_color: Background color for processed images
    
    Returns:
        Dictionary mapping characters to list of processed image paths
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_images = {}
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    for img_file in input_path.rglob("*"):
        if img_file.suffix.lower() in image_extensions:
            try:
                # Load image
                img = Image.open(img_file)
                
                # Convert to RGB
                if img.mode != 'RGB':
                    if img.mode == 'RGBA':
                        # Handle transparency
                        background = Image.new('RGB', img.size, background_color)
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    else:
                        img = img.convert('RGB')
                
                # Resize while maintaining aspect ratio
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                
                # Create new image with target size and paste centered
                new_img = Image.new('RGB', target_size, background_color)
                paste_x = (target_size[0] - img.width) // 2
                paste_y = (target_size[1] - img.height) // 2
                new_img.paste(img, (paste_x, paste_y))
                
                # Extract character from filename
                char = extract_character_from_filename(img_file.name)
                
                if char:
                    if char not in processed_images:
                        processed_images[char] = []
                    
                    # Save processed image
                    output_file = output_path / f"{char}_{len(processed_images[char]):03d}.png"
                    new_img.save(output_file, 'PNG')
                    processed_images[char].append(str(output_file))
                
            except Exception as e:
                logging.warning(f"Failed to process {img_file}: {e}")
    
    # Log statistics
    total_images = sum(len(images) for images in processed_images.values())
    logging.info(f"Processed {total_images} images for {len(processed_images)} characters")
    
    return processed_images

def extract_character_from_filename(filename: str) -> Optional[str]:
    """
    Extract character from filename
    
    Args:
        filename: Image filename
    
    Returns:
        Extracted character or None
    """
    # Remove extension
    name = Path(filename).stem.lower()
    
    # Method 1: Single character filename
    if len(name) == 1 and (name.isalpha() or name in '.,!?;:-\'"'):
        return name
    
    # Method 2: Character_number format
    if '_' in name:
        parts = name.split('_')
        if len(parts) >= 1:
            char = parts[0]
            if len(char) == 1 and (char.isalpha() or char in '.,!?;:-\'"'):
                return char
    
    # Method 3: Character followed by numbers
    import re
    match = re.match(r'^([a-z.,!?;:\-\'"]{1})\d*', name)
    if match:
        return match.group(1)
    
    return None

def calculate_dataset_statistics(dataset) -> Dict[str, Any]:
    """
    Calculate comprehensive dataset statistics
    
    Args:
        dataset: Dataset to analyze
    
    Returns:
        Dictionary with dataset statistics
    """
    if len(dataset) == 0:
        return {"error": "Empty dataset"}
    
    # Character distribution
    char_dist = dataset.get_character_distribution() if hasattr(dataset, 'get_character_distribution') else {}
    
    # Sample a few items to get image statistics
    sample_images = []
    sample_prompts = []
    
    sample_size = min(100, len(dataset))  # Sample up to 100 items
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in indices:
        try:
            sample = dataset[idx]
            sample_images.append(sample['images'])
            sample_prompts.append(sample['prompts'])
        except:
            continue
    
    # Image statistics
    if sample_images:
        if isinstance(sample_images[0], torch.Tensor):
            image_shapes = [img.shape for img in sample_images]
            image_means = [img.mean().item() for img in sample_images]
            image_stds = [img.std().item() for img in sample_images]
        else:
            image_shapes = [np.array(img).shape for img in sample_images]
            image_means = [np.array(img).mean() for img in sample_images]
            image_stds = [np.array(img).std() for img in sample_images]
    else:
        image_shapes, image_means, image_stds = [], [], []
    
    # Prompt statistics
    prompt_lengths = [len(prompt) for prompt in sample_prompts]
    
    stats = {
        "total_samples": len(dataset),
        "character_distribution": char_dist,
        "unique_characters": len(char_dist),
        "image_statistics": {
            "shapes": list(set(map(str, image_shapes))),
            "mean_pixel_value": {
                "mean": np.mean(image_means) if image_means else 0,
                "std": np.std(image_means) if image_means else 0
            },
            "pixel_std": {
                "mean": np.mean(image_stds) if image_stds else 0,
                "std": np.std(image_stds) if image_stds else 0
            }
        },
        "prompt_statistics": {
            "length": {
                "mean": np.mean(prompt_lengths) if prompt_lengths else 0,
                "std": np.std(prompt_lengths) if prompt_lengths else 0,
                "min": min(prompt_lengths) if prompt_lengths else 0,
                "max": max(prompt_lengths) if prompt_lengths else 0
            }
        }
    }
    
    return stats

def memory_cleanup():
    """Clean up GPU and system memory"""
    import gc
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()

def estimate_training_time(
    dataset_size: int,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation_steps: int = 1,
    seconds_per_step: float = 2.0
) -> Dict[str, str]:
    """
    Estimate training time
    
    Args:
        dataset_size: Number of samples in dataset
        batch_size: Training batch size
        num_epochs: Number of epochs
        gradient_accumulation_steps: Gradient accumulation steps
        seconds_per_step: Estimated seconds per training step
    
    Returns:
        Dictionary with time estimates
    """
    steps_per_epoch = dataset_size // (batch_size * gradient_accumulation_steps)
    total_steps = steps_per_epoch * num_epochs
    total_seconds = total_steps * seconds_per_step
    
    # Convert to human readable format
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    return {
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "estimated_time": f"{hours}h {minutes}m {seconds}s",
        "estimated_seconds": total_seconds
    }

def create_inference_pipeline(
    model_path: str,
    base_model: str = "runwayml/stable-diffusion-v1-5"
):
    """
    Create inference pipeline from trained LoRA model
    
    Args:
        model_path: Path to trained LoRA model
        base_model: Base Stable Diffusion model
    
    Returns:
        Configured pipeline for inference
    """
    from diffusers import StableDiffusionPipeline
    
    # Load base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Load LoRA weights
    pipeline.unet.load_attn_procs(model_path)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    
    # Enable memory efficient attention
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    
    if hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
    
    return pipeline

def generate_character_samples(
    pipeline,
    characters: List[str],
    output_dir: str,
    num_samples_per_char: int = 4,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50
) -> Dict[str, List[str]]:
    """
    Generate sample characters using trained model
    
    Args:
        pipeline: Trained diffusion pipeline
        characters: List of characters to generate
        output_dir: Directory to save generated samples
        num_samples_per_char: Number of samples per character
        guidance_scale: Guidance scale for generation
        num_inference_steps: Number of inference steps
    
    Returns:
        Dictionary mapping characters to generated image paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_samples = {}
    
    for char in characters:
        char_samples = []
        
        # Generate prompt
        prompt = f"brush lettering character '{char}', elegant calligraphy style, black ink on white paper"
        
        for i in range(num_samples_per_char):
            try:
                # Generate image
                with torch.autocast("cuda"):
                    image = pipeline(
                        prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        height=512,
                        width=512
                    ).images[0]
                
                # Save image
                output_file = output_path / f"generated_{char}_{i:03d}.png"
                image.save(output_file)
                char_samples.append(str(output_file))
                
            except Exception as e:
                logging.warning(f"Failed to generate sample for '{char}': {e}")
        
        generated_samples[char] = char_samples
        logging.info(f"Generated {len(char_samples)} samples for character '{char}'")
    
    return generated_samples