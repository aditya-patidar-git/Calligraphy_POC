#!/usr/bin/env python3
"""
Calligraphy Style Learning System using LoRA and Diffusion Models
This script trains a model to learn calligraphic strokes and generate text in the learned style.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Handles preprocessing of calligraphy character images"""
    
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size
        
    def binarize_image(self, image: np.ndarray, method: str = 'otsu') -> np.ndarray:
        """Binarize image using different methods"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        else:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
        return binary
    
    def extract_stroke_features(self, image: np.ndarray) -> Dict:
        """Extract stroke characteristics from the image"""
        binary = self.binarize_image(image)
        
        # Find contours to analyze stroke patterns
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = {
            'stroke_width': self._calculate_stroke_width(binary),
            'curvature': self._calculate_curvature(contours),
            'aspect_ratio': binary.shape[1] / binary.shape[0],
            'density': np.sum(binary == 0) / (binary.shape[0] * binary.shape[1])
        }
        
        return features
    
    def _calculate_stroke_width(self, binary: np.ndarray) -> float:
        """Calculate average stroke width using distance transform"""
        # Invert binary (stroke pixels should be white for distance transform)
        inverted = cv2.bitwise_not(binary)
        dist_transform = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
        return np.mean(dist_transform[dist_transform > 0]) * 2 if np.any(dist_transform > 0) else 1.0
    
    def _calculate_curvature(self, contours: List) -> float:
        """Calculate average curvature of strokes"""
        if not contours:
            return 0.0
            
        total_curvature = 0.0
        total_points = 0
        
        for contour in contours:
            if len(contour) > 10:  # Need sufficient points for curvature calculation
                # Approximate contour to reduce noise
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) > 3:
                    curvature = self._compute_contour_curvature(approx)
                    total_curvature += curvature
                    total_points += 1
        
        return total_curvature / total_points if total_points > 0 else 0.0
    
    def _compute_contour_curvature(self, contour: np.ndarray) -> float:
        """Compute curvature for a single contour"""
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            return 0.0
            
        # Calculate curvature using discrete approximation
        curvatures = []
        for i in range(1, len(points) - 1):
            p1, p2, p3 = points[i-1], points[i], points[i+1]
            
            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Cross product for curvature
            cross = np.cross(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 > 0 and norm_v2 > 0:
                curvature = abs(cross) / (norm_v1 * norm_v2)
                curvatures.append(curvature)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Main preprocessing function"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Extract features before heavy preprocessing
        features = self.extract_stroke_features(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binarize
        binary = self.binarize_image(gray, method='otsu')
        
        # Noise removal
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Resize while maintaining aspect ratio
        processed = self._resize_with_padding(cleaned, self.target_size)
        
        # Normalize to [0, 1]
        processed = processed.astype(np.float32) / 255.0
        
        return processed, features
    
    def _resize_with_padding(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image while maintaining aspect ratio with padding"""
        h, w = image.shape
        target_h, target_w = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.ones(target_size, dtype=np.uint8) * 255  # White background
        
        # Calculate padding offsets
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded

class CalligraphyDataset(Dataset):
    """Dataset for calligraphy character images"""
    
    def __init__(self, image_paths: List[str], labels: List[str], 
                 preprocessor: ImagePreprocessor, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.transform = transform
        
        # Create character to index mapping
        unique_chars = list(set(labels))
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Preprocess image
        processed_image, features = self.preprocessor.preprocess_image(image_path)
        
        # Convert to tensor
        if self.transform:
            processed_image = self.transform(processed_image)
        else:
            processed_image = torch.from_numpy(processed_image).unsqueeze(0)  # Add channel dimension
        
        # Convert label to index
        label_idx = self.char_to_idx[label]
        
        return {
            'image': processed_image,
            'label': torch.tensor(label_idx, dtype=torch.long),
            'char': label,
            'features': features,
            'path': image_path
        }

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.1)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Original layer (frozen)
        self.original_layer = nn.Linear(in_features, out_features)
        self.original_layer.requires_grad_(False)
        
    def forward(self, x):
        # Original output
        original_out = self.original_layer(x)
        
        # LoRA adaptation
        lora_out = torch.matmul(torch.matmul(x, self.lora_A), self.lora_B)
        lora_out = lora_out * (self.alpha / self.rank)
        
        return original_out + lora_out

class CalligraphyStyleEncoder(nn.Module):
    """Encoder network to learn calligraphy style features"""
    
    def __init__(self, input_channels: int = 1, style_dim: int = 512, num_classes: int = 26):
        super().__init__()
        
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Style embedding layers with LoRA
        self.style_layers = nn.Sequential(
            LoRALayer(512 * 4 * 4, 1024, rank=32),
            nn.ReLU(),
            nn.Dropout(0.3),
            LoRALayer(1024, style_dim, rank=16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Character classification head
        self.classifier = nn.Linear(style_dim, num_classes)
        
        # Style consistency head
        self.style_head = nn.Linear(style_dim, style_dim)
        
    def forward(self, x):
        # Extract convolutional features
        conv_features = self.conv_layers(x)
        conv_features = conv_features.view(conv_features.size(0), -1)
        
        # Get style embedding
        style_embedding = self.style_layers(conv_features)
        
        # Classification output
        class_logits = self.classifier(style_embedding)
        
        # Style consistency output
        style_output = self.style_head(style_embedding)
        
        return {
            'style_embedding': style_embedding,
            'class_logits': class_logits,
            'style_output': style_output,
            'conv_features': conv_features
        }

class CalligraphyGenerator(nn.Module):
    """Generator network to create calligraphy from style codes"""
    
    def __init__(self, style_dim: int = 512, char_embed_dim: int = 128, 
                 output_size: int = 128):
        super().__init__()
        
        self.style_dim = style_dim
        self.char_embed_dim = char_embed_dim
        self.output_size = output_size
        
        # Character embedding
        self.char_embedding = nn.Embedding(26, char_embed_dim)  # Assuming 26 characters
        
        # Combined input dimension
        combined_dim = style_dim + char_embed_dim
        
        # Generator layers with LoRA
        self.generator = nn.Sequential(
            LoRALayer(combined_dim, 1024, rank=32),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            
            LoRALayer(1024, 2048, rank=32),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            
            LoRALayer(2048, 4096, rank=32),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            
            # Reshape and upsample
        )
        
        # Convolutional decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16x16 -> 32x32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32x32 -> 64x64
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, 1, 4, 2, 1),     # 64x64 -> 128x128
            nn.Sigmoid()
        )
        
    def forward(self, style_code, char_indices):
        # Get character embeddings
        char_embeds = self.char_embedding(char_indices)
        
        # Combine style and character information
        combined = torch.cat([style_code, char_embeds], dim=1)
        
        # Generate features
        generated_features = self.generator(combined)
        
        # Reshape for convolutional decoder
        batch_size = generated_features.size(0)
        reshaped = generated_features.view(batch_size, 256, 4, 4)
        
        # Generate image
        generated_image = self.decoder(reshaped)
        
        return generated_image

class StyleConsistencyLoss(nn.Module):
    """Custom loss for maintaining style consistency"""
    
    def __init__(self, lambda_style: float = 1.0, lambda_content: float = 1.0):
        super().__init__()
        self.lambda_style = lambda_style
        self.lambda_content = lambda_content
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets, style_ref=None):
        # Classification loss
        class_loss = self.ce_loss(outputs['class_logits'], targets['labels'])
        
        # Style consistency loss
        if style_ref is not None:
            style_loss = self.mse_loss(outputs['style_output'], style_ref)
        else:
            style_loss = torch.tensor(0.0, device=outputs['style_output'].device)
        
        # Total loss
        total_loss = self.lambda_content * class_loss + self.lambda_style * style_loss
        
        return {
            'total_loss': total_loss,
            'class_loss': class_loss,
            'style_loss': style_loss
        }

class CalligraphyTrainer:
    """Main training class"""
    
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
        self.criterion = None
        
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
            # Try to infer labels from filenames
            for img_path in image_dir.glob('*.png'):
                # Assume filename format: "char_X.png" where X is the character
                filename = img_path.stem
                if '_' in filename:
                    char = filename.split('_')[-1]
                    if len(char) == 1 and char.isalpha():
                        image_paths.append(str(img_path))
                        labels.append(char.lower())
            
            # Also try .jpg files
            for img_path in image_dir.glob('*.jpg'):
                filename = img_path.stem
                if '_' in filename:
                    char = filename.split('_')[-1]
                    if len(char) == 1 and char.isalpha():
                        image_paths.append(str(img_path))
                        labels.append(char.lower())
        
        if not image_paths:
            raise ValueError("No valid images found. Please check your image directory and naming convention.")
        
        logger.info(f"Found {len(image_paths)} images with labels")
        
        # Create dataset
        self.dataset = CalligraphyDataset(
            image_paths=image_paths,
            labels=labels,
            preprocessor=self.preprocessor
        )
        
        # Split data
        train_idx, val_idx = train_test_split(
            range(len(image_paths)), 
            test_size=0.2, 
            random_state=42, 
            stratify=labels
        )
        
        train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
        val_dataset = torch.utils.data.Subset(self.dataset, val_idx)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
    def setup_model(self):
        """Initialize model components"""
        logger.info("Setting up model...")
        
        num_classes = len(self.dataset.char_to_idx)
        
        # Initialize encoder
        self.encoder = CalligraphyStyleEncoder(
            input_channels=1,
            style_dim=self.config['style_dim'],
            num_classes=num_classes
        ).to(self.device)
        
        # Initialize generator
        self.generator = CalligraphyGenerator(
            style_dim=self.config['style_dim'],
            char_embed_dim=self.config['char_embed_dim'],
            output_size=self.config['image_size']
        ).to(self.device)
        
        # Setup optimizer (only LoRA parameters)
        lora_params = []
        for module in self.encoder.modules():
            if isinstance(module, LoRALayer):
                lora_params.extend([module.lora_A, module.lora_B])
        
        for module in self.generator.modules():
            if isinstance(module, LoRALayer):
                lora_params.extend([module.lora_A, module.lora_B])
        
        # Add other trainable parameters
        other_params = [p for p in self.encoder.parameters() if p not in lora_params]
        other_params.extend([p for p in self.generator.parameters() if p not in lora_params])
        
        self.optimizer = torch.optim.AdamW([
            {'params': lora_params, 'lr': self.config['lora_lr']},
            {'params': other_params, 'lr': self.config['base_lr']}
        ], weight_decay=self.config['weight_decay'])
        
        # Setup criterion
        self.criterion = StyleConsistencyLoss(
            lambda_style=self.config['style_weight'],
            lambda_content=self.config['content_weight']
        )
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)} trainable parameters")
        
    def train_epoch(self):
        """Train for one epoch"""
        self.encoder.train()
        self.generator.train()
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_style_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass through encoder
            encoder_outputs = self.encoder(images)
            
            # Calculate loss
            targets = {'labels': labels}
            loss_dict = self.criterion(encoder_outputs, targets)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss_dict['total_loss'].item()
            total_class_loss += loss_dict['class_loss'].item()
            total_style_loss += loss_dict['style_loss'].item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(self.train_loader)}: "
                          f"Loss={loss_dict['total_loss'].item():.4f}")
        
        return {
            'total_loss': total_loss / len(self.train_loader),
            'class_loss': total_class_loss / len(self.train_loader),
            'style_loss': total_style_loss / len(self.train_loader)
        }
    
    def validate(self):
        """Validate the model"""
        self.encoder.eval()
        self.generator.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                encoder_outputs = self.encoder(images)
                
                # Calculate loss
                targets = {'labels': labels}
                loss_dict = self.criterion(encoder_outputs, targets)
                total_loss += loss_dict['total_loss'].item()
                
                # Calculate accuracy
                _, predicted = torch.max(encoder_outputs['class_logits'], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.val_loader)
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, num_epochs: int):
        """Main training loop"""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Accuracy: {val_metrics['accuracy']:.2f}%")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model('best_model.pth')
                logger.info("Saved best model")
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'char_to_idx': self.dataset.char_to_idx,
            'idx_to_char': self.dataset.idx_to_char
        }
        torch.save(checkpoint, filename)
        logger.info(f"Model saved to {filename}")
    
    def load_model(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {filename}")
        return checkpoint

class CalligraphyTextGenerator:
    """Generate calligraphy text using trained model"""
    
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = checkpoint['idx_to_char']
        
        # Initialize models
        self.encoder = CalligraphyStyleEncoder(
            input_channels=1,
            style_dim=self.config['style_dim'],
            num_classes=len(self.char_to_idx)
        ).to(self.device)
        
        self.generator = CalligraphyGenerator(
            style_dim=self.config['style_dim'],
            char_embed_dim=self.config['char_embed_dim'],
            output_size=self.config['image_size']
        ).to(self.device)
        
        # Load weights
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        
        self.encoder.eval()
        self.generator.eval()
        
    def extract_style_from_reference(self, reference_image_path: str) -> torch.Tensor:
        """Extract style code from a reference calligraphy image"""
        preprocessor = ImagePreprocessor(
            target_size=(self.config['image_size'], self.config['image_size'])
        )
        
        # Preprocess reference image
        processed_image, _ = preprocessor.preprocess_image(reference_image_path)
        processed_image = torch.from_numpy(processed_image).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            encoder_output = self.encoder(processed_image)
            style_code = encoder_output['style_embedding']
        
        return style_code
    
    def generate_character(self, char: str, style_code: torch.Tensor) -> np.ndarray:
        """Generate a single character with given style"""
        if char.lower() not in self.char_to_idx:
            raise ValueError(f"Character '{char}' not in training vocabulary")
        
        char_idx = torch.tensor([self.char_to_idx[char.lower()]], device=self.device)
        
        with torch.no_grad():
            generated_image = self.generator(style_code, char_idx)
            generated_image = generated_image.squeeze().cpu().numpy()
        
        # Convert back to 0-255 range
        generated_image = (generated_image * 255).astype(np.uint8)
        
        return generated_image
    
    def generate_text(self, text: str, reference_image_path: str, 
                     output_path: str, spacing: int = 10) -> str:
        """Generate full text in