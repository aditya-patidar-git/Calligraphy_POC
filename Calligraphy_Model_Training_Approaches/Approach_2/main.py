# %%writefile /kaggle/working/main.py
#!/usr/bin/env python3

# main.py - Complete LoRA Training Script for Calligraphy Generation
# This script coordinates the entire training process for both simple and complex calligraphy styles

import os
import sys
import argparse
from pathlib import Path
import torch
import gc
from datetime import datetime
import json
import shutil

# Import our custom modules (assuming they're in the same directory or properly installed)
# If running in Kaggle, make sure all the other .py files are uploaded
try:
    from setup import *  # This imports all the basic setup, logging, etc.
    from data_preparation import ImageProcessor, DatasetBuilder
    from dataset import CalligraphyDataset, create_data_loaders, preview_dataset
    from model_setup import LoRAModelSetup, TrainingConfig, calculate_memory_usage
    from trainer import CalligraphyTrainer, create_trainer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all the component files (setup.py, data_preparation.py, etc.) are in the same directory")
    sys.exit(1)

class CalligraphyTrainingPipeline:
    """Main pipeline for calligraphy LoRA training"""
    
    def __init__(self, 
                 character_dir: str,
                 word_dir: str,
                 style_name: str = "calligraphy",
                 output_base_dir: str = "/kaggle/working",
                 complex_style: bool = False):
        
        self.character_dir = Path(character_dir)
        self.word_dir = Path(word_dir)
        self.style_name = style_name
        self.output_base_dir = Path(output_base_dir)
        self.complex_style = complex_style
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_base_dir / f"{style_name}_lora_{timestamp}"
        self.data_dir = self.experiment_dir / "processed_data"
        self.model_dir = self.experiment_dir / "model"
        self.logs_dir = self.experiment_dir / "logs"
        
        # Create directories
        for dir_path in [self.experiment_dir, self.data_dir, self.model_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized training pipeline for {style_name}")
        logger.info(f"Output directory: {self.experiment_dir}")
        
    def validate_input_data(self):
        """Validate input directories and data"""
        if not self.character_dir.exists():
            raise FileNotFoundError(f"Character directory not found: {self.character_dir}")
        
        if not self.word_dir.exists():
            raise FileNotFoundError(f"Word directory not found: {self.word_dir}")
        
        # Count available images
        char_images = list(self.character_dir.glob("*.png")) + list(self.character_dir.glob("*.jpg"))
        word_images = list(self.word_dir.glob("*.png")) + list(self.word_dir.glob("*.jpg"))
        
        logger.info(f"Found {len(char_images)} character images")
        logger.info(f"Found {len(word_images)} word images")
        
        if len(char_images) == 0 and len(word_images) == 0:
            raise ValueError("No training images found in the specified directories")
        
        return len(char_images), len(word_images)
    
    def setup_config(self, char_count: int, word_count: int):
        """Setup training configuration based on data and hardware"""
        config = TrainingConfig()
        
        # Adjust configuration based on complexity and data size
        total_samples = char_count + word_count
        
        if self.complex_style:
            # More aggressive training for complex styles
            config.lora_rank = 8  # Higher rank for complex patterns
            config.lora_alpha = 64
            config.learning_rate = 5e-5  # Lower learning rate for stability
            config.max_epochs = 100  # More epochs for complex styles
            config.noise_offset = 0.15  # Higher noise offset
            config.style_name = f"complex_{self.style_name}"
        else:
            # Standard configuration for simple styles
            config.lora_rank = 4
            config.lora_alpha = 32
            config.learning_rate = 1e-4
            config.max_epochs = 50
            config.noise_offset = 0.1
            config.style_name = self.style_name
        
        # Adjust batch size and epochs based on data size
        if total_samples < 50:
            config.batch_size = 1
            config.gradient_accumulation_steps = 8
            config.max_epochs = max(config.max_epochs, 80)  # More epochs for small datasets
        elif total_samples > 200:
            config.batch_size = 2 if torch.cuda.is_available() else 1
            config.gradient_accumulation_steps = 2
        
        # Update paths
        config.output_dir = str(self.model_dir)
        config.logging_dir = str(self.logs_dir)
        
        logger.info(f"Configuration setup complete:")
        logger.info(f"  LoRA rank: {config.lora_rank}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  Max epochs: {config.max_epochs}")
        logger.info(f"  Batch size: {config.batch_size}")
        
        return config
    
    def prepare_data(self):
        """Prepare training data"""
        logger.info("Starting data preparation...")
        
        # Initialize dataset builder
        dataset_builder = DatasetBuilder(style_name=self.style_name)
        
        # Create training data
        training_info = dataset_builder.create_training_data(
            character_dir=str(self.character_dir),
            word_dir=str(self.word_dir),
            output_dir=str(self.data_dir)
        )
        
        logger.info(f"Data preparation complete: {training_info['total_samples']} samples created")
        
        return training_info
    
    def setup_models(self, config):
        """Setup LoRA models"""
        logger.info("Setting up models...")
        
        # Initialize model setup
        model_setup = LoRAModelSetup(
            model_name=config.model_name,
            revision=config.revision,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Load base models
        models = model_setup.load_models()
        
        # Setup LoRA configuration
        lora_config = model_setup.setup_lora_config(
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout
        )
        
        # Apply LoRA to UNet
        models["unet"] = model_setup.apply_lora_to_unet(lora_config)
        
        logger.info("Models setup complete")
        
        return models
    
    def create_data_loaders(self, training_info, config, models):
        """Create training and validation data loaders"""
        logger.info("Creating data loaders...")
        
        train_loader, val_loader = create_data_loaders(
            images_dir=training_info["images_dir"],
            metadata_path=training_info["metadata_path"],
            tokenizer=models["tokenizer"],
            batch_size=config.batch_size,
            validation_split=config.validation_split,
            resolution=config.resolution
        )
        
        logger.info(f"Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        return train_loader, val_loader
    
    def train_model(self, config, models, train_loader, val_loader):
        """Train the LoRA model"""
        
        print("✅ BEGIN TRAINING DIAGNOSTICS")
        print(f"Epochs to train: {config.max_epochs}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Batch size: {config.batch_size}")
        print("✅ END DIAGNOSTICS")
        
        logger.info("Starting model training...")
        
        # Create trainer
        trainer = create_trainer(config, models)
        
        # Start training
        final_model_path = trainer.train(train_loader, val_loader)
        
        logger.info(f"Training complete! Final model saved to: {final_model_path}")
        
        return final_model_path
    
    def create_inference_pipeline(self, final_model_path, config):
        """Create inference pipeline for testing"""
        logger.info("Creating inference pipeline...")
        
        try:
            from diffusers import StableDiffusionPipeline
            
            # Load the trained LoRA model
            pipeline = StableDiffusionPipeline.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Load LoRA weights
            pipeline.unet.load_adapter(final_model_path)
            
            if torch.cuda.is_available():
                pipeline = pipeline.to("cuda")
            
            # Save pipeline for later use
            pipeline_save_path = self.experiment_dir / "inference_pipeline"
            pipeline.save_pretrained(pipeline_save_path)
            
            logger.info(f"Inference pipeline saved to: {pipeline_save_path}")
            
            return pipeline, pipeline_save_path
            
        except Exception as e:
            logger.warning(f"Failed to create inference pipeline: {e}")
            return None, None
    
    def test_generation(self, pipeline, config):
        """Test the trained model with sample prompts"""
        if pipeline is None:
            logger.warning("No pipeline available for testing")
            return
        
        logger.info("Testing model with sample prompts...")
        
        # Test prompts based on style complexity
        if self.complex_style:
            test_prompts = [
                f"a {config.style_name} gothic calligraphy letter 'd', black ink on white background, ornate medieval style",
                f"a {config.style_name} gothic calligraphy word 'dog', black ink on white background, decorative lettering",
                f"a {config.style_name} gothic calligraphy letter 'A', black ink on white background, elaborate flourishes",
                f"a {config.style_name} gothic calligraphy word 'fate', black ink on white background, medieval manuscript style"
            ]
        else:
            test_prompts = [
                f"a {config.style_name} calligraphy letter 'a', black ink on white background, elegant handwriting",
                f"a {config.style_name} calligraphy word 'hello', black ink on white background, flowing script",
                f"a {config.style_name} calligraphy letter 'g', black ink on white background, cursive style",
                f"a {config.style_name} calligraphy word 'world', black ink on white background, beautiful lettering"
            ]
        
        test_results_dir = self.experiment_dir / "test_results"
        test_results_dir.mkdir(exist_ok=True)
        
        try:
            for i, prompt in enumerate(test_prompts):
                logger.info(f"Generating: {prompt}")
                
                with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    image = pipeline(
                        prompt,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        height=config.resolution,
                        width=config.resolution,
                        generator=torch.Generator().manual_seed(42)  # For reproducible results
                    ).images[0]
                
                # Save result
                image.save(test_results_dir / f"test_{i+1:02d}.png")
                
                # Also save prompt for reference
                with open(test_results_dir / f"test_{i+1:02d}_prompt.txt", "w") as f:
                    f.write(prompt)
            
            logger.info(f"Test results saved to: {test_results_dir}")
            
        except Exception as e:
            logger.error(f"Error during testing: {e}")
    
    def save_experiment_info(self, config, training_info, final_model_path):
        """Save experiment information and metadata"""
        experiment_info = {
            "experiment_name": f"{self.style_name}_lora_training",
            "timestamp": datetime.now().isoformat(),
            "style_name": self.style_name,
            "complex_style": self.complex_style,
            "input_directories": {
                "character_dir": str(self.character_dir),
                "word_dir": str(self.word_dir)
            },
            "output_directories": {
                "experiment_dir": str(self.experiment_dir),
                "model_dir": str(self.model_dir),
                "data_dir": str(self.data_dir)
            },
            "training_data": training_info,
            "final_model_path": str(final_model_path),
            "config": config.to_dict(),
            "hardware_info": {
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            }
        }
        
        # Save experiment info
        info_path = self.experiment_dir / "experiment_info.json"
        with open(info_path, "w") as f:
            json.dump(experiment_info, f, indent=2)
        
        logger.info(f"Experiment info saved to: {info_path}")
        
        return experiment_info
    
    def run_complete_training(self):
        """Run the complete training pipeline"""
        logger.info("=" * 80)
        logger.info(f"STARTING CALLIGRAPHY LORA TRAINING PIPELINE")
        logger.info(f"Style: {self.style_name}")
        logger.info(f"Complex Style: {self.complex_style}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Validate input data
            char_count, word_count = self.validate_input_data()
            
            # Step 2: Setup configuration
            config = self.setup_config(char_count, word_count)
            
            # Step 3: Check memory
            logger.info("Initial memory status:")
            calculate_memory_usage()
            
            # Step 4: Prepare data
            training_info = self.prepare_data()
            
            # Step 5: Setup models
            models = self.setup_models(config)
            
            # Step 6: Create data loaders
            train_loader, val_loader = self.create_data_loaders(training_info, config, models)
            
            # Step 7: Preview dataset (optional)
            logger.info("Previewing dataset...")
            try:
                dataset = CalligraphyDataset(
                    images_dir=training_info["images_dir"],
                    metadata_path=training_info["metadata_path"],
                    tokenizer=models["tokenizer"],
                    resolution=config.resolution
                )
                preview_dataset(dataset, num_samples=3)
            except Exception as e:
                logger.warning(f"Failed to preview dataset: {e}")
            
            # Step 8: Train model
            final_model_path = self.train_model(config, models, train_loader, val_loader)
            
            # Step 9: Create inference pipeline
            pipeline, pipeline_path = self.create_inference_pipeline(final_model_path, config)
            
            # Step 10: Test generation
            self.test_generation(pipeline, config)
            
            # Step 11: Save experiment info
            experiment_info = self.save_experiment_info(config, training_info, final_model_path)
            
            # Clean up memory
            del models, train_loader, val_loader
            if pipeline:
                del pipeline
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            logger.info("=" * 80)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Results saved to: {self.experiment_dir}")
            logger.info(f"Final model: {final_model_path}")
            logger.info("=" * 80)
            
            return {
                "success": True,
                "experiment_dir": str(self.experiment_dir),
                "final_model_path": str(final_model_path),
                "pipeline_path": str(pipeline_path) if pipeline_path else None,
                "experiment_info": experiment_info
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            
            return {
                "success": False,
                "error": str(e),
                "experiment_dir": str(self.experiment_dir)
            }

def main():
    """Main function with command line interface"""
    
    # Create and run training pipeline
    pipeline = CalligraphyTrainingPipeline(
        character_dir='/kaggle/input/character-simple-images',
        word_dir='/kaggle/input/word-simple-images',
        style_name="calligraphy",
        output_base_dir='/kaggle/working/output-data',
        complex_style=False
    )
    
    result = pipeline.run_complete_training()
    
    if result["success"]:
        print("\n" + "="*50)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print("="*50)
        print(f"📁 Results: {result['experiment_dir']}")
        print(f"🤖 Model: {result['final_model_path']}")
        if result['pipeline_path']:
            print(f"🔧 Pipeline: {result['pipeline_path']}")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ TRAINING FAILED")
        print("="*50)
        print(f"Error: {result['error']}")
        print(f"Check logs in: {result['experiment_dir']}")
        print("="*50)
        sys.exit(1)

# Convenience function for Jupyter notebook usage
def train_calligraphy_model(character_dir: str, 
                           word_dir: str, 
                           style_name: str = "calligraphy",
                           complex_style: bool = False,
                           output_dir: str = "/kaggle/working"):
    """
    Convenience function for training in Jupyter notebooks
    
    Args:
        character_dir: Path to directory with character images
        word_dir: Path to directory with word images  
        style_name: Name of the calligraphy style
        complex_style: Whether to use complex style settings
        output_dir: Base output directory
    
    Returns:
        Training results dictionary
    """
    pipeline = CalligraphyTrainingPipeline(
        character_dir=character_dir,
        word_dir=word_dir,
        style_name=style_name,
        output_base_dir=output_dir,
        complex_style=complex_style
    )
    
    return pipeline.run_complete_training()

if __name__ == "__main__":
    main()


"""  
result = train_calligraphy_model(
    character_dir="/kaggle/input/character-simple-images",
    word_dir="/kaggle/input/word-simple-images", 
    style_name="elegant_script",
    complex_style=False
)

# For simple calligraphy (like your first image):

# Example usage for Kaggle/Jupyter:

# For complex/gothic calligraphy (like your second image):
result = train_calligraphy_model(
    character_dir="/kaggle/input/your-data/character_images",
    word_dir="/kaggle/input/your-data/word_images",
    style_name="gothic_calligraphy", 
    complex_style=True
)
"""