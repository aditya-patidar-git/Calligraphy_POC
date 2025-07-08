
#!/usr/bin/env python3
# config.py - Configuration settings for calligraphy LoRA training
# This file contains all configuration settings for both simple and complex calligraphy styles

import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
import os

class BaseConfig:
    """Base configuration class with common settings"""
    
    def __init__(self):
        # Hardware settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = "fp16" if torch.cuda.is_available() else "no"
        self.use_8bit_adam = True if torch.cuda.is_available() else False
        
        # Model settings
        self.model_name = "runwayml/stable-diffusion-v1-5"
        self.revision = "fp16" if torch.cuda.is_available() else "main"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Image settings
        self.resolution = 512  # Standard SD resolution
        self.center_crop = True
        self.random_flip = False  # Usually False for text/calligraphy
        
        # Dataset settings
        self.validation_split = 0.1
        self.max_train_samples = None  # None means use all samples
        self.preprocessing_num_workers = 4 if torch.cuda.is_available() else 2
        
        # Output settings
        self.output_base_dir = "/kaggle/working"
        self.save_precision = "fp16" if torch.cuda.is_available() else "fp32"
        self.save_model_as = "safetensors"  # or "ckpt"
        
        # Logging settings
        self.logging_level = "INFO"
        self.log_with = "tensorboard"  # "wandb", "tensorboard", or None
        self.report_to = None  # "wandb" if you want to use wandb
        
        # Safety settings
        self.enable_xformers_memory_efficient_attention = True
        self.gradient_checkpointing = True  # Saves memory at cost of speed
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class SimpleCalligraphyConfig(BaseConfig):
    """Configuration for simple calligraphy styles (elegant, cursive, etc.)"""
    
    def __init__(self, style_name: str = "elegant_calligraphy"):
        super().__init__()
        
        # Style settings
        self.style_name = style_name
        self.style_type = "simple"
        
        # LoRA settings - Conservative for simple styles
        self.lora_rank = 4
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.lora_target_modules = [
            "to_k", "to_q", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2"
        ]
        
        # Training settings - Optimized for simple styles
        self.batch_size = 1
        self.gradient_accumulation_steps = 4  # Effective batch size = 4
        self.learning_rate = 1e-4
        self.lr_scheduler = "cosine"
        self.lr_warmup_steps = 500
        self.max_epochs = 15
        self.max_train_steps = None  # Will be calculated
        
        # Optimization settings
        self.optimizer_type = "AdamW8bit" if self.use_8bit_adam else "AdamW"
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        
        # Noise and generation settings - Gentle for simple styles
        self.noise_offset = 0.1
        self.input_perturbation = 0.0
        self.snr_gamma = 5.0
        self.prediction_type = "epsilon"
        
        # Data augmentation - Minimal for text
        self.random_crop = False
        self.color_jitter = False
        
        # Validation settings
        self.validation_epochs = 5
        self.validation_steps = 250
        self.num_validation_images = 4
        self.validation_prompts = [
            f"a {style_name} calligraphy letter 'a', black ink on white background, elegant handwriting",
            f"a {style_name} calligraphy word 'hello', black ink on white background, flowing script",
            f"a {style_name} calligraphy letter 'g', black ink on white background, cursive style",
            f"a {style_name} calligraphy word 'world', black ink on white background, beautiful lettering"
        ]
        
        # Checkpointing
        self.save_steps = 500
        self.save_every_n_epochs = 10
        self.keep_only_last_checkpoint = False
        
        # Memory optimization for Kaggle
        self.enable_cpu_offload = False  # Can enable if running out of memory
        self.use_cpu_text_encoder = False
        self.pre_compute_text_embeddings = True

class ComplexCalligraphyConfig(BaseConfig):
    """Configuration for complex calligraphy styles (gothic, ornate, etc.)"""
    
    def __init__(self, style_name: str = "gothic_calligraphy"):
        super().__init__()
        
        # Style settings
        self.style_name = style_name
        self.style_type = "complex"
        
        # LoRA settings - More aggressive for complex styles
        self.lora_rank = 8  # Higher rank for complex patterns
        self.lora_alpha = 64  # Higher alpha for stronger adaptation
        self.lora_dropout = 0.15  # Slightly higher dropout
        self.lora_target_modules = [
            "to_k", "to_q", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2",
            "conv_in", "conv_out"  # Additional modules for complex styles
        ]
        
        # Training settings - More intensive for complex styles
        self.batch_size = 1
        self.gradient_accumulation_steps = 8  # Effective batch size = 8
        self.learning_rate = 5e-5  # Lower learning rate for stability
        self.lr_scheduler = "cosine_with_restarts"
        self.lr_warmup_steps = 1000  # Longer warmup
        self.max_epochs = 100  # More epochs for complex styles
        self.max_train_steps = None
        
        # Optimization settings
        self.optimizer_type = "AdamW8bit" if self.use_8bit_adam else "AdamW"
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        
        # Noise settings - More aggressive for complex styles
        self.noise_offset = 0.15  # Higher noise offset
        self.input_perturbation = 0.1  # Add input perturbation
        self.snr_gamma = 3.0  # Lower SNR gamma for more aggressive training
        self.prediction_type = "epsilon"
        
        # Data augmentation - Still minimal for text
        self.random_crop = False
        self.color_jitter = False
        
        # Validation settings
        self.validation_epochs = 10
        self.validation_steps = 500
        self.num_validation_images = 6
        self.validation_prompts = [
            f"a {style_name} gothic calligraphy letter 'd', black ink on white background, ornate medieval style",
            f"a {style_name} gothic calligraphy word 'dog', black ink on white background, decorative lettering",
            f"a {style_name} gothic calligraphy letter 'A', black ink on white background, elaborate flourishes",
            f"a {style_name} gothic calligraphy word 'fate', black ink on white background, medieval manuscript style",
            f"a {style_name} gothic calligraphy letter 'f', black ink on white background, ornamental design",
            f"a {style_name} gothic calligraphy word 'dreams', black ink on white background, gothic script"
        ]
        
        # Checkpointing - More frequent for complex training
        self.save_steps = 250
        self.save_every_n_epochs = 5
        self.keep_only_last_checkpoint = False
        
        # Memory optimization
        self.enable_cpu_offload = True  # More likely needed for complex training
        self.use_cpu_text_encoder = False
        self.pre_compute_text_embeddings = True
        self.gradient_checkpointing = True

class KaggleOptimizedConfig:
    """Additional optimizations specifically for Kaggle environment"""
    
    @staticmethod
    def apply_kaggle_optimizations(config: BaseConfig) -> BaseConfig:
        """Apply Kaggle-specific optimizations"""
        
        # Memory optimizations
        config.gradient_checkpointing = True
        config.enable_xformers_memory_efficient_attention = True
        config.pre_compute_text_embeddings = True
        
        # Batch size optimizations for limited memory
        if config.style_type == "complex":
            config.batch_size = 1
            config.gradient_accumulation_steps = 8
        else:
            config.batch_size = 1
            config.gradient_accumulation_steps = 4
        
        # Use 8-bit optimizers to save memory
        config.use_8bit_adam = True
        config.optimizer_type = "AdamW8bit"
        
        # Enable CPU offloading if needed
        if hasattr(config, 'enable_cpu_offload'):
            config.enable_cpu_offload = True
        
        # Adjust resolution if needed (can lower to 448 or 384 to save memory)
        # config.resolution = 448  # Uncomment if running out of memory
        
        return config

class DataConfig:
    """Configuration for data preparation and processing"""
    
    def __init__(self):
        # Data paths (will be set by main script)
        self.character_images_dir = None
        self.word_images_dir = None
        self.output_data_dir = None
        
        # Image processing settings
        self.target_size = (512, 512)
        self.background_color = (255, 255, 255)  # White background
        self.text_color = (0, 0, 0)  # Black text
        self.padding = 50  # Padding around text
        
        # Data augmentation settings
        self.enable_augmentation = True
        self.rotation_range = 5  # degrees
        self.brightness_range = 0.1
        self.contrast_range = 0.1
        self.noise_factor = 0.02
        
        # Text processing
        self.min_text_length = 1
        self.max_text_length = 50
        self.filter_non_alphabetic = False
        
        # Dataset splitting
        self.train_split = 0.8
        self.val_split = 0.1
        self.test_split = 0.1
        
        # Caching
        self.cache_latents = True  # Cache VAE latents to speed up training
        self.cache_text_embeddings = True

def get_config(style_type: str = "simple", 
               style_name: str = None,
               apply_kaggle_optimizations: bool = True) -> BaseConfig:
    """
    Factory function to get appropriate configuration
    
    Args:
        style_type: "simple" or "complex"
        style_name: Name of the calligraphy style
        apply_kaggle_optimizations: Whether to apply Kaggle-specific optimizations
    
    Returns:
        Configured config object
    """
    
    if style_type.lower() == "simple":
        if style_name is None:
            style_name = "elegant_calligraphy"
        config = SimpleCalligraphyConfig(style_name)
    elif style_type.lower() == "complex":
        if style_name is None:
            style_name = "gothic_calligraphy"
        config = ComplexCalligraphyConfig(style_name)
    else:
        raise ValueError(f"Unknown style_type: {style_type}. Must be 'simple' or 'complex'")
    
    # Apply Kaggle optimizations if requested
    if apply_kaggle_optimizations:
        config = KaggleOptimizedConfig.apply_kaggle_optimizations(config)
    
    return config

def create_experiment_config(character_dir: str,
                           word_dir: str,
                           style_name: str,
                           style_type: str = "simple",
                           output_dir: str = "/kaggle/working") -> Dict[str, Any]:
    """
    Create complete experiment configuration
    
    Args:
        character_dir: Path to character images directory
        word_dir: Path to word images directory
        style_name: Name of the calligraphy style
        style_type: "simple" or "complex"
        output_dir: Base output directory
    
    Returns:
        Complete experiment configuration dictionary
    """
    
    # Get main config
    main_config = get_config(style_type, style_name, apply_kaggle_optimizations=True)
    
    # Create data config
    data_config = DataConfig()
    data_config.character_images_dir = character_dir
    data_config.word_images_dir = word_dir
    data_config.output_data_dir = os.path.join(output_dir, f"{style_name}_processed_data")
    
    # Update main config with paths
    main_config.output_base_dir = output_dir
    main_config.character_dir = character_dir
    main_config.word_dir = word_dir
    
    return {
        "main_config": main_config,
        "data_config": data_config,
        "experiment_info": {
            "style_name": style_name,
            "style_type": style_type,
            "character_dir": character_dir,
            "word_dir": word_dir,
            "output_dir": output_dir
        }
    }

# Predefined style configurations
STYLE_PRESETS = {
    "elegant_script": {
        "type": "simple",
        "description": "Elegant flowing script calligraphy",
        "recommended_epochs": 50,
        "difficulty": "easy"
    },
    "modern_calligraphy": {
        "type": "simple", 
        "description": "Modern brush-style calligraphy",
        "recommended_epochs": 60,
        "difficulty": "easy"
    },
    "gothic_calligraphy": {
        "type": "complex",
        "description": "Medieval gothic style with ornate flourishes",
        "recommended_epochs": 100,
        "difficulty": "hard"
    },
    "blackletter": {
        "type": "complex",
        "description": "Traditional blackletter/textura style",
        "recommended_epochs": 120,
        "difficulty": "very_hard"
    },
    "copperplate": {
        "type": "simple",
        "description": "Classic copperplate business script",
        "recommended_epochs": 40,
        "difficulty": "medium"
    }
}

def get_style_preset(style_name: str) -> Dict[str, Any]:
    """Get predefined style configuration"""
    return STYLE_PRESETS.get(style_name, {
        "type": "simple",
        "description": "Custom calligraphy style",
        "recommended_epochs": 50,
        "difficulty": "medium"
    })

# Environment checks
def check_environment():
    """Check if environment is properly configured"""
    checks = {
        "torch_available": False,
        "cuda_available": False,
        "diffusers_available": False,
        "peft_available": False,
        "sufficient_memory": False
    }
    
    try:
        import torch
        checks["torch_available"] = True
        checks["cuda_available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            checks["sufficient_memory"] = gpu_memory >= 12  # At least 12GB recommended
        
        import diffusers
        checks["diffusers_available"] = True
        
        import peft
        checks["peft_available"] = True
        
    except ImportError as e:
        print(f"Import error: {e}")
    
    return checks

if __name__ == "__main__":
    # Example usage and testing
    print("=== Calligraphy LoRA Training Configuration ===")
    
    # Check environment
    env_checks = check_environment()
    print(f"Environment checks: {env_checks}")
    
    # Test simple configuration
    print("\n--- Simple Calligraphy Config ---")
    simple_config = get_config("simple", "elegant_script")
    print(f"Style: {simple_config.style_name}")
    print(f"LoRA Rank: {simple_config.lora_rank}")
    print(f"Learning Rate: {simple_config.learning_rate}")
    print(f"Max Epochs: {simple_config.max_epochs}")
    
    # Test complex configuration  
    print("\n--- Complex Calligraphy Config ---")
    complex_config = get_config("complex", "gothic_calligraphy")
    print(f"Style: {complex_config.style_name}")
    print(f"LoRA Rank: {complex_config.lora_rank}")
    print(f"Learning Rate: {complex_config.learning_rate}")
    print(f"Max Epochs: {complex_config.max_epochs}")
    
    # Test style presets
    print("\n--- Available Style Presets ---")
    for style, info in STYLE_PRESETS.items():
        print(f"{style}: {info['description']} ({info['type']}, {info['difficulty']})")
    
    print("\nConfiguration module loaded successfully!")