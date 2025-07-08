#!/usr/bin/env python3
"""
Main training script for calligraphy LoRA model
"""

import os
import json
import logging
import argparse
from pathlib import Path
import torch

from trainer import CalligraphyTrainer
from utils import (
    setup_directories, 
    validate_setup, 
    create_training_config,
    visualize_character_samples,
    prepare_dataset,
    check_image_quality
)

def setup_logging(log_dir: str = "logs"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete")
    return logger

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train LoRA model for calligraphy style learning')
    
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing character images')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to training configuration file')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for models and logs')
    parser.add_argument('--prepare_data', action='store_true',
                        help='Preprocess and prepare dataset before training')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations of the dataset')
    parser.add_argument('--validate_only', action='store_true',
                        help='Only validate setup without training')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from checkpoint')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup directories
    setup_directories(args.output_dir)
    
    # Setup logging
    logger = setup_logging(os.path.join(args.output_dir, 'logs'))
    
    logger.info("=== Calligraphy LoRA Training Started ===")
    logger.info(f"Image directory: {args.image_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Validate setup
    if not validate_setup(args.image_dir):
        logger.error("Setup validation failed. Please fix the issues and try again.")
        return
    
    if args.validate_only:
        logger.info("Validation complete. Exiting.")
        return
    
    # Check image quality and get statistics
    logger.info("Analyzing image quality...")
    stats, issues = check_image_quality(args.image_dir, 
                                      os.path.join(args.output_dir, 'quality_report.txt'))
    
    if issues:
        logger.warning(f"Found {len(issues)} image quality issues. Check quality_report.txt for details.")
    
    # Prepare dataset if requested
    dataset_dir = args.image_dir
    if args.prepare_data:
        logger.info("Preparing dataset...")
        dataset_dir = prepare_dataset(
            args.image_dir, 
            os.path.join(args.output_dir, 'prepared_dataset')
        )
        logger.info(f"Dataset prepared in: {dataset_dir}")
    
    # Create visualizations if requested
    if args.visualize:
        logger.info("Creating dataset visualizations...")
        visualize_character_samples(dataset_dir, max_samples=16)
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from: {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        logger.info("Creating default configuration...")
        config = create_training_config(
            dataset_dir, 
            os.path.join(args.output_dir, 'training_config.json')
        )
    
    # Update config with output directory
    config['output']['save_dir'] = os.path.join(args.output_dir, 'checkpoints')
    config['output']['log_dir'] = os.path.join(args.output_dir, 'logs')
    
    # Log configuration
    logger.info("Training Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = CalligraphyTrainer(config)
        
        # Setup data
        logger.info("Setting up data loaders...")
        trainer.setup_data(dataset_dir)
        
        # Setup model
        logger.info("Setting up model...")
        trainer.setup_model()
        
        # Resume from checkpoint if specified
        if args.resume_from and os.path.exists(args.resume_from):
            logger.info(f"Resuming training from: {args.resume_from}")
            trainer.load_model(args.resume_from)
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("=== Training Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()