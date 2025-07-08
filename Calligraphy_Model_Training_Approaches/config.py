# %%writefile config.py
#!/usr/bin/env python3
"""
Configuration File for setting up data for other files
"""


# config.py - Configuration classes for brush lettering training
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

@dataclass
class TrainingConfig:
    """Configuration for LoRA training of brush lettering model"""
    
    # Data settings
    data_dir: str = "./data"
    resolution: int = 512
    
    # Model settings
    model_name: str = "runwayml/stable-diffusion-v1-5"
    
    # LoRA settings
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "to_k", "to_q", "to_v", "to_out.0",
        "proj_in", "proj_out", 
        "ff.net.0.proj", "ff.net.2"
    ])
    
    # Training settings
    num_epochs: int = 30
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Optimizer settings
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Mixed precision
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    
    # Validation and saving
    validation_steps: int = 100
    save_steps: int = 500
    max_checkpoints: int = 5
    
    # Output settings
    output_dir: str = "./outputs"
    logging_dir: Optional[str] = None
    
    # Noise scheduler settings
    noise_scheduler_type: str = "ddpm"
    prediction_type: str = "epsilon"  # "epsilon" or "v_prediction"
    
    # Data augmentation
    enable_augmentation: bool = True
    augmentation_probability: float = 0.3
    
    # Advanced training settings
    min_snr_gamma: Optional[float] = None  # For Min-SNR weighting
    use_ema: bool = False  # Exponential Moving Average
    ema_decay: float = 0.9999
    
    # Validation generation settings
   # Complete validation_prompts list based on dataset.py character templates
    # This should replace the existing validation_prompts in TrainingConfig

    validation_prompts: List[str] = field(default_factory=lambda: [
    # Lowercase letters (a-z)
    "brush lettering character 'a', elegant calligraphy style",
    "handwritten letter 'b' in brush calligraphy",
    "brush lettering character 'c', artistic style",
    "handwritten letter 'd' in brush calligraphy",
    "elegant brush stroke character 'e'",
    "brush lettering character 'f', calligraphy style",
    "flowing lowercase 'g' with descender loop",
    "elegant lowercase 'h' with tall ascender",
    "simple lowercase 'i' with dot",
    "elegant lowercase 'j' with curved descender",
    "dynamic lowercase 'k' with angled strokes",
    "elegant lowercase 'l' with tall ascender",
    "flowing lowercase 'm' with double arch",
    "elegant lowercase 'n' with smooth arch",
    "circular lowercase 'o' in brush calligraphy",
    "elegant lowercase 'p' with descender",
    "flowing lowercase 'q' with curved tail",
    "elegant lowercase 'r' with smooth shoulder",
    "curved lowercase 's' in brush calligraphy",
    "elegant lowercase 't' with crossbar",
    "flowing lowercase 'u' in brush calligraphy",
    "dynamic lowercase 'v' with angled strokes",
    "wide lowercase 'w' in brush calligraphy",
    "dynamic lowercase 'x' with crossed strokes",
    "flowing lowercase 'y' with descender",
    "dynamic lowercase 'z' with zigzag",
    
    # Uppercase letters (A-Z)
    "brush lettering capital 'A', elegant calligraphy style",
    "elegant uppercase 'B' with double bowls",
    "curved uppercase 'C' in brush calligraphy",
    "elegant uppercase 'D' with curved bowl",
    "structured uppercase 'E' in brush calligraphy",
    "elegant uppercase 'F' with horizontal bars",
    "curved uppercase 'G' in brush calligraphy",
    "structured uppercase 'H' with crossbar",
    "simple uppercase 'I' in brush calligraphy",
    "elegant uppercase 'J' with curved hook",
    "dynamic uppercase 'K' with angled strokes",
    "elegant uppercase 'L' with horizontal base",
    "majestic uppercase 'M' in brush calligraphy",
    "elegant uppercase 'N' with diagonal stroke",
    "circular uppercase 'O' in brush calligraphy",
    "elegant uppercase 'P' with closed bowl",
    "distinctive uppercase 'Q' with tail",
    "elegant uppercase 'R' with bowl and leg",
    "curved uppercase 'S' in brush calligraphy",
    "elegant uppercase 'T' with horizontal top",
    "curved uppercase 'U' in brush calligraphy",
    "dynamic uppercase 'V' with angled strokes",
    "wide uppercase 'W' in brush calligraphy",
    "dynamic uppercase 'X' with crossed strokes",
    "graceful uppercase 'Y' in brush calligraphy",
    "dynamic uppercase 'Z' with zigzag",
    
    # Numbers (0-9)
    "brush lettering number '0', elegant calligraphy style",
    "elegant numeral '1' with clean stem",
    "curved numeral '2' in brush calligraphy",
    "elegant numeral '3' with double curves",
    "angular numeral '4' in brush calligraphy",
    "elegant numeral '5' with curved bottom",
    "curved numeral '6' in brush calligraphy",
    "elegant numeral '7' with angled stroke",
    "curved numeral '8' in brush calligraphy",
    "elegant numeral '9' with curved top",
    
    # Punctuation marks
    "brush lettering period, calligraphy style",
    "curved comma in brush calligraphy",
    "dynamic exclamation point in brush script",
    "curved question mark in brush calligraphy",
    "elegant semicolon in brush script",
    "simple colon in brush calligraphy",
    "elegant dash in brush script",
    "curved apostrophe in brush calligraphy",
    "elegant quotes in brush script",
    
    # Mixed examples for better validation
    "brush lettering word 'art' in flowing script",
    "elegant brush calligraphy word 'ink'",
    "handwritten word 'pen' in brush style",
    "artistic brush lettering 'joy'",
    "flowing brush script word 'zen'",
    "elegant calligraphy word 'mix'",
    "brush lettering phrase 'A1' in artistic style",
    "handwritten characters 'Be' in brush calligraphy",
    "elegant brush stroke 'Go!' with exclamation",
    "artistic brush lettering 'OK?' with question mark"
  ])
    # Memory optimization
    gradient_checkpointing: bool = False
    enable_cpu_offload: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set logging directory if not specified
        if self.logging_dir is None:
            self.logging_dir = str(Path(self.output_dir) / "logs")
        
        # Validate mixed precision setting
        if self.mixed_precision not in ["no", "fp16", "bf16"]:
            raise ValueError(f"Invalid mixed_precision: {self.mixed_precision}")
        
        # Validate model name
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        
        # Ensure batch size is positive
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        # Ensure learning rate is positive
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TrainingConfig':
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        # Convert to dict, handling Path objects and other non-serializable types
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                config_dict[key] = value
            else:
                config_dict[key] = str(value)
        
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_effective_batch_size(self) -> int:
        """Get the effective batch size considering gradient accumulation"""
        return self.batch_size * self.gradient_accumulation_steps
    
    def get_total_steps(self, dataset_size: int) -> int:
        """Calculate total training steps"""
        steps_per_epoch = dataset_size // self.get_effective_batch_size()
        return steps_per_epoch * self.num_epochs
    
    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")

@dataclass 
class DatasetConfig:
    """Configuration for dataset processing"""
    
    # Data paths
    data_dir: str = "./data"
    images_subdir: str = "images"
    prompts_file: Optional[str] = None
    metadata_file: Optional[str] = None
    
    # Data processing
    resolution: int = 512
    train_split: float = 0.9
    random_seed: int = 42
    
    # Image processing
    image_extensions: List[str] = field(default_factory=lambda: [
        '.png', '.jpg', '.jpeg', '.bmp', '.tiff'
    ])
    convert_to_rgb: bool = True
    normalize_range: tuple = (-1, 1)  # Range for SD models
    
    # Augmentation settings
    enable_augmentation: bool = True
    rotation_degrees: float = 5.0
    perspective_distortion: float = 0.1
    brightness_jitter: float = 0.1
    contrast_jitter: float = 0.1
    random_erase_prob: float = 0.1
    
    # Character-specific settings
    supported_characters: List[str] = field(default_factory=lambda: [
        *[chr(i) for i in range(ord('a'), ord('z') + 1)],  # a-z
        *[chr(i) for i in range(ord('A'), ord('Z') + 1)],  # A-Z
        *[chr(i) for i in range(ord('0'), ord('9') + 1)],  # 0-9
        '.', ',', '!', '?', ';', ':', '-', "'", '"'
    ])
    
    def __post_init__(self):
        """Validate dataset configuration"""
        if not 0 < self.train_split < 1:
            raise ValueError("train_split must be between 0 and 1")
        
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        
        # Ensure data directory path exists
        if not Path(self.data_dir).exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

@dataclass
class InferenceConfig:
    """Configuration for inference/generation"""
    
    # Model paths
    base_model_path: str = "runwayml/stable-diffusion-v1-5"
    lora_model_path: str = "./outputs/final_model"
    
    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy"
    
    # Output settings
    output_resolution: int = 512
    num_images_per_prompt: int = 1
    
    # Sampling settings
    scheduler_type: str = "ddim"  # "ddim", "ddpm", "dpm", "euler"
    eta: float = 0.0
    
    # Safety settings
    safety_checker: bool = False
    nsfw_filter: bool = False
    
    def __post_init__(self):
        """Validate inference configuration"""
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")

# Preset configurations for different use cases
class ConfigPresets:
    """Predefined configuration presets for different scenarios"""
    
    @staticmethod
    def quick_test() -> TrainingConfig:
        """Quick test configuration for debugging"""
        return TrainingConfig(
            num_epochs=5,
            batch_size=1,
            save_steps=50,
            validation_steps=25,
            lora_rank=8,
            learning_rate=5e-4
        )
    
    @staticmethod
    def production_training() -> TrainingConfig:
        """Production training configuration"""
        return TrainingConfig(
            num_epochs=200,
            batch_size=2,
            gradient_accumulation_steps=8,
            save_steps=1000,
            validation_steps=500,
            lora_rank=32,
            lora_alpha=64,
            learning_rate=1e-4,
            use_8bit_adam=True,
            gradient_checkpointing=True
        )
    
    @staticmethod
    def memory_efficient() -> TrainingConfig:
        """Memory-efficient configuration for limited GPU memory"""
        return TrainingConfig(
            batch_size=1,
            gradient_accumulation_steps=16,
            resolution=256,
            lora_rank=8,
            gradient_checkpointing=True,
            enable_cpu_offload=True,
            mixed_precision="fp16"
        )
    
    @staticmethod
    def high_quality() -> TrainingConfig:
        """High-quality training configuration"""
        return TrainingConfig(
            resolution=768,
            num_epochs=300,
            lora_rank=64,
            lora_alpha=128,
            learning_rate=5e-5,
            batch_size=1,
            gradient_accumulation_steps=32,
            use_ema=True
        )

# Utility functions
def load_config_from_args(args) -> TrainingConfig:
    """Create TrainingConfig from argparse arguments"""
    config_dict = {}
    
    # Map argument names to config attributes
    arg_mapping = {
        'data_dir': 'data_dir',
        'output_dir': 'output_dir',
        'model_name': 'model_name',
        'resolution': 'resolution',
        'batch_size': 'batch_size',
        'num_epochs': 'num_epochs',
        'learning_rate': 'learning_rate',
        'lora_rank': 'lora_rank',
        'lora_alpha': 'lora_alpha',
        'gradient_accumulation_steps': 'gradient_accumulation_steps',
        'save_steps': 'save_steps',
        'validation_steps': 'validation_steps',
        'mixed_precision': 'mixed_precision',
        'use_8bit_adam': 'use_8bit_adam'
    }
    
    for arg_name, config_key in arg_mapping.items():
        if hasattr(args, arg_name):
            value = getattr(args, arg_name)
            if value is not None:
                config_dict[config_key] = value
    
    return TrainingConfig(**config_dict)

def save_config_template(output_path: str = "./config_template.json"):
    """Save a template configuration file"""
    config = TrainingConfig()
    config.to_json(output_path)
    print(f"Configuration template saved to: {output_path}")

if __name__ == "__main__":
    # Example usage and testing
    print("Testing configuration classes...")
    
    # Test default configuration
    config = TrainingConfig()
    print(f"Default config created: {config.model_name}")
    
    # Test preset configurations
    quick_config = ConfigPresets.quick_test()
    print(f"Quick test config: {quick_config.num_epochs} epochs")
    
    prod_config = ConfigPresets.production_training()
    print(f"Production config: {prod_config.lora_rank} rank")
    
    # Test configuration saving/loading
    config.to_json("test_config.json")
    loaded_config = TrainingConfig.from_json("test_config.json")
    print(f"Config loaded successfully: {loaded_config.learning_rate}")
    
    # Test dataset configuration
    dataset_config = DatasetConfig()
    print(f"Dataset config: {len(dataset_config.supported_characters)} supported characters")
    
    print("All configuration tests passed!")