#!/usr/bin/env python3
"""
Training module for calligraphy style learning with LoRA
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

from preprocessor import ImagePreprocessor
from dataset import CalligraphyDataset
from models import (
    CalligraphyStyleEncoder, 
    CalligraphyGenerator, 
    StyleConsistencyLoss,
    count_lora_parameters,
    freeze_non_lora_parameters
)

logger = logging.getLogger(__name__)

class CalligraphyTrainer:
    """Main training class for LoRA-based calligraphy learning"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(
            target_size=(config['image_size'], config['image_size'])
        )
        
        # Will be initialized in setup_data
        self.train_loader = None
        self.val_loader = None
        self.dataset = None
        
        # Will be initialized in setup_model
        self.encoder = None
        self.generator = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def setup_data(self, image_dir: str, labels_file: Optional[str] = None):
        """Setup data loaders"""
        logger.info("Setting up data...")
        
        # Get image paths
        image_paths = []
        labels = []
        
        image_dir = Path(image_dir)
        
        if labels_file and os.path.exists(labels_file):
            # Load labels from file
            with open(labels_file, 'r') as f:
                label_data = json.load(f)
            
            for img_path, label in label_data.items():
                full_path = image_dir / img_path
                if full_path.exists():
                    image_paths.append(str(full_path))
                    labels.append(label)
        else:
            # Extract labels from filename (assuming format: character_*.png)
            for img_file in image_dir.glob("*.png"):
                try:
                    # Extract character from filename (e.g., 'a_001.png' -> 'a')
                    char_name = img_file.stem.split('_')[0]
                    if len(char_name) == 1:  # Single character
                        image_paths.append(str(img_file))
                        labels.append(char_name.lower())
                except:
                    logger.warning(f"Could not extract label from {img_file}")
            
            # Also check for jpg files
            for img_file in image_dir.glob("*.jpg"):
                try:
                    char_name = img_file.stem.split('_')[0]
                    if len(char_name) == 1:
                        image_paths.append(str(img_file))
                        labels.append(char_name.lower())
                except:
                    logger.warning(f"Could not extract label from {img_file}")
        
        if not image_paths:
            raise ValueError(f"No valid images found in {image_dir}")
        
        logger.info(f"Found {len(image_paths)} images with labels")
        
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, 
            test_size=self.config.get('val_split', 0.2),
            random_state=42,
            stratify=labels if len(set(labels)) > 1 else None
        )
        
        # Create transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
        ])
        
        # Create datasets
        train_dataset = CalligraphyDataset(
            train_paths, train_labels, self.preprocessor, transform=transform
        )
        val_dataset = CalligraphyDataset(
            val_paths, val_labels, self.preprocessor, transform=None
        )
        
        # Store dataset for model initialization
        self.dataset = train_dataset
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        logger.info(f"Classes: {train_dataset.get_num_classes()}")
        
    def setup_model(self):
        """Initialize model, optimizer, and loss function"""
        logger.info("Setting up model...")
        
        if self.dataset is None:
            raise ValueError("Must call setup_data() before setup_model()")
        
        # Initialize encoder
        self.encoder = CalligraphyStyleEncoder(
            input_channels=1,
            style_dim=self.config['style_dim'],
            num_classes=self.dataset.get_num_classes(),
            lora_rank=self.config['lora_rank'],
            lora_alpha=self.config['lora_alpha']
        ).to(self.device)
        
        # Initialize generator (optional, for future use)
        self.generator = CalligraphyGenerator(
            style_dim=self.config['style_dim'],
            char_embed_dim=self.config.get('char_embed_dim', 128),
            output_size=self.config['image_size'],
            num_chars=self.dataset.get_num_classes(),
            lora_rank=self.config['lora_rank'],
            lora_alpha=self.config['lora_alpha']
        ).to(self.device)
        
        # Freeze non-LoRA parameters
        freeze_non_lora_parameters(self.encoder)
        freeze_non_lora_parameters(self.generator)
        
        # Count parameters
        encoder_lora_params, encoder_total_params = count_lora_parameters(self.encoder)
        generator_lora_params, generator_total_params = count_lora_parameters(self.generator)
        
        logger.info(f"Encoder - LoRA params: {encoder_lora_params:,}, Total params: {encoder_total_params:,}")
        logger.info(f"Generator - LoRA params: {generator_lora_params:,}, Total params: {generator_total_params:,}")
        
        # Combine parameters for optimization
        lora_parameters = []
        for model in [self.encoder, self.generator]:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    lora_parameters.append(param)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            lora_parameters,
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        # Initialize loss function
        self.criterion = StyleConsistencyLoss(
            lambda_style=self.config.get('lambda_style', 1.0),
            lambda_content=self.config.get('lambda_content', 1.0),
            lambda_perceptual=self.config.get('lambda_perceptual', 0.1)
        )
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.encoder.train()
        self.generator.train()
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_style_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass through encoder
            encoder_outputs = self.encoder(images)
            
            # Calculate loss
            targets = {'labels': labels}
            loss_dict = self.criterion(encoder_outputs, targets)
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.encoder.parameters() if p.requires_grad], 
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_class_loss += loss_dict['class_loss'].item()
            total_style_loss += loss_dict['style_loss'].item()
            
            # Calculate accuracy
            predictions = torch.argmax(encoder_outputs['class_logits'], dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct_predictions/total_samples:.2f}%',
                'LR': f'{current_lr:.6f}'
            })
        
        # Calculate averages
        avg_loss = total_loss / len(self.train_loader)
        avg_class_loss = total_class_loss / len(self.train_loader)
        avg_style_loss = total_style_loss / len(self.train_loader)
        accuracy = 100 * correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'class_loss': avg_class_loss,
            'style_loss': avg_style_loss,
            'accuracy': accuracy
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate the model"""
        self.encoder.eval()
        
        total_loss = 0.0
        total_class_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                encoder_outputs = self.encoder(images)
                
                # Calculate loss
                targets = {'labels': labels}
                loss_dict = self.criterion(encoder_outputs, targets)
                
                total_loss += loss_dict['total_loss'].item()
                total_class_loss += loss_dict['class_loss'].item()
                
                # Calculate accuracy
                predictions = torch.argmax(encoder_outputs['class_logits'], dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_class_loss = total_class_loss / len(self.val_loader)
        accuracy = 100 * correct_predictions / total_samples
        
        return {
            'loss': avg_loss,
            'class_loss': avg_class_loss,
            'accuracy': accuracy
        }
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = self.config.get('patience', 10)
        
        for epoch in range(self.config['epochs']):
            # Training
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['loss'])
            
            # Validation
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Update scheduler
            self.scheduler.step()
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']}:")
            logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                self.save_model('best_model.pth')
                logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping after {patience} epochs without improvement")
                break
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
        
        logger.info("Training completed!")
        self.plot_training_curves()
    
    def save_model(self, filename: str):
        """Save model state"""
        os.makedirs(self.config['save_dir'], exist_ok=True)
        save_path = os.path.join(self.config['save_dir'], filename)
        
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'char_to_idx': self.dataset.char_to_idx,
            'idx_to_char': self.dataset.idx_to_char,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, filename: str):
        """Load model state"""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        
        logger.info(f"Model loaded from {filename}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config['save_dir'], 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Training curves saved to {plot_path}")

def main():
    """Main training script"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = {
        'image_size': 128,
        'batch_size': 16,
        'epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'val_split': 0.2,
        'num_workers': 2,
        'style_dim': 512,
        'char_embed_dim': 128,
        'lora_rank': 16,
        'lora_alpha': 16.0,
        'lambda_style': 1.0,
        'lambda_content': 1.0,
        'lambda_perceptual': 0.1,
        'save_dir': 'checkpoints',
        'save_every': 10,
        'patience': 15
    }
    
    # Initialize trainer
    trainer = CalligraphyTrainer(config)
    
    # Setup data (adjust path to your image directory)
    image_directory = "character_images"  # Replace with your image directory
    trainer.setup_data(image_directory)
    
    # Setup model
    trainer.setup_model()
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()