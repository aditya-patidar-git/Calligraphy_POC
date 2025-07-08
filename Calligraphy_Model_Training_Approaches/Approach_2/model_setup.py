# %%writefile /kaggle/working/model_setup.py
#!/usr/bin/env python3
# Cell 4: LoRA Model Setup and Configuration
from typing import List
import torch
import logging
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from peft import get_peft_model, LoraConfig, TaskType
logger = logging.getLogger(__name__)


class LoRAModelSetup:
    """Setup and configure LoRA model for calligraphy training"""
    
    def __init__(self, 
                 model_name: str = "runwayml/stable-diffusion-v1-5",
                 revision: str = "fp16",
                 torch_dtype: torch.dtype = torch.float16):
        
        self.model_name = model_name
        self.revision = revision
        self.torch_dtype = torch_dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_models(self):
        """Load and setup all required models"""
        logger.info(f"Loading models from {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_name,
            subfolder="tokenizer",
            revision=self.revision,
            torch_dtype=self.torch_dtype
        )
        
        # Load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_name,
            subfolder="text_encoder",
            revision=self.revision,
            torch_dtype=self.torch_dtype
        )
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name,
            subfolder="vae",
            revision=self.revision,
            torch_dtype=self.torch_dtype
        )
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_name,
            subfolder="unet",
            revision=self.revision,
            torch_dtype=self.torch_dtype
        )
        
        # Load scheduler
        self.scheduler = DDPMScheduler.from_pretrained(
            self.model_name,
            subfolder="scheduler"
        )
        
        # Move models to device
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        self.unet.to(self.device)
        
        # Set models to eval mode (except UNet which we'll train)
        self.text_encoder.eval()
        self.vae.eval()
        
        # Freeze parameters
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        logger.info("Models loaded successfully")
        
        return {
            "tokenizer": self.tokenizer,
            "text_encoder": self.text_encoder,
            "vae": self.vae,
            "unet": self.unet,
            "scheduler": self.scheduler
        }
    
    def setup_lora_config(self, 
                         rank: int = 4,
                         alpha: int = 32,
                         dropout: float = 0.1,
                         target_modules: List[str] = None):
        """Setup LoRA configuration"""
        
        if target_modules is None:
            # Target attention modules in UNet
            target_modules = [
                "to_k", "to_q", "to_v", "to_out.0",
                "ff.net.0.proj", "ff.net.2"
            ]
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.DIFFUSION
        )
        
        logger.info(f"LoRA config created with rank={rank}, alpha={alpha}")
        return lora_config
    
    def apply_lora_to_unet(self, lora_config: LoraConfig):
        """Apply LoRA to UNet model"""
        
        # Apply LoRA
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        logger.info("LoRA applied to UNet successfully")
        return self.unet

class TrainingConfig:
    """Training configuration and hyperparameters"""
    
    def __init__(self):
        # Model settings
        self.model_name = "runwayml/stable-diffusion-v1-5"
        self.revision = "fp16"
        self.resolution = 512
        
        # LoRA settings
        self.lora_rank = 4
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        
        # Training settings
        self.batch_size = 1  # Small batch size for limited GPU memory
        self.gradient_accumulation_steps = 4  # Effective batch size = 4
        self.learning_rate = 1e-4
        self.max_epochs = 50
        self.save_steps = 500
        self.validation_steps = 250
        
        # Optimization settings
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        
        # Noise settings
        self.noise_offset = 0.1
        self.snr_gamma = 5.0
        
        # Output settings
        self.output_dir = "/kaggle/working/calligraphy_lora"
        self.logging_dir = "/kaggle/working/logs"
        self.mixed_precision = "fp16"
        
        # Validation settings
        self.validation_split = 0.1
        self.num_validation_images = 4
        
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

def calculate_memory_usage():
    """Calculate and display memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Total: {total:.2f} GB")
        print(f"  Free: {total - reserved:.2f} GB")
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "total": total,
            "free": total - reserved
        }
    else:
        print("CUDA not available")
        return None

# Initialize configuration
config = TrainingConfig()
model_setup = LoRAModelSetup(
    model_name=config.model_name,
    revision=config.revision,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

print("LoRA model setup classes loaded successfully!")
print(f"Configuration: {config.to_dict()}")
print("\nMemory status:")
calculate_memory_usage()