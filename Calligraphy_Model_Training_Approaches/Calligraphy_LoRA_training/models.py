#!/usr/bin/env python3
"""
Model architecture module with LoRA implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.1)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Dropout for LoRA
        self.lora_dropout = nn.Dropout(p=dropout)
        
        # Original layer (frozen)
        self.original_layer = nn.Linear(in_features, out_features)
        self.original_layer.requires_grad_(False)
        
        # Initialize LoRA weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize LoRA parameters"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Original output
        original_out = self.original_layer(x)
        
        # LoRA adaptation
        lora_out = self.lora_dropout(x)
        lora_out = torch.matmul(lora_out, self.lora_A)
        lora_out = torch.matmul(lora_out, self.lora_B)
        lora_out = lora_out * (self.alpha / self.rank)
        
        return original_out + lora_out

class LoRAConv2d(nn.Module):
    """LoRA adaptation for Conv2d layers"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 rank: int = 16, alpha: float = 16.0, **kwargs):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Original conv layer (frozen)
        self.original_conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.original_conv.requires_grad_(False)
        
        # LoRA decomposition for conv layers
        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size, **kwargs)
        self.lora_B = nn.Conv2d(rank, out_channels, 1, bias=False)
        
        # Initialize LoRA weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize LoRA parameters"""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        # Original output
        original_out = self.original_conv(x)
        
        # LoRA adaptation
        lora_out = self.lora_A(x)
        lora_out = self.lora_B(lora_out)
        lora_out = lora_out * (self.alpha / self.rank)
        
        return original_out + lora_out

class ResidualBlock(nn.Module):
    """Residual block with LoRA adaptation"""
    
    def __init__(self, channels: int, rank: int = 16):
        super().__init__()
        self.conv1 = LoRAConv2d(channels, channels, 3, rank=rank, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = LoRAConv2d(channels, channels, 3, rank=rank, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class CalligraphyStyleEncoder(nn.Module):
    """Encoder network to learn calligraphy style features with LoRA"""
    
    def __init__(self, input_channels: int = 1, style_dim: int = 512, num_classes: int = 26, 
                 lora_rank: int = 16, lora_alpha: float = 16.0):
        super().__init__()
        
        self.style_dim = style_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # Convolutional feature extractor with LoRA
        self.conv1 = nn.Sequential(
            LoRAConv2d(input_channels, 64, 3, rank=lora_rank, alpha=lora_alpha, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            LoRAConv2d(64, 64, 3, rank=lora_rank, alpha=lora_alpha, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            LoRAConv2d(64, 128, 3, rank=lora_rank, alpha=lora_alpha, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            LoRAConv2d(128, 128, 3, rank=lora_rank, alpha=lora_alpha, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            LoRAConv2d(128, 256, 3, rank=lora_rank, alpha=lora_alpha, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            LoRAConv2d(256, 256, 3, rank=lora_rank, alpha=lora_alpha, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv4 = nn.Sequential(
            LoRAConv2d(256, 512, 3, rank=lora_rank, alpha=lora_alpha, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Residual blocks for better feature extraction
        self.residual_blocks = nn.Sequential(
            ResidualBlock(512, rank=lora_rank),
            ResidualBlock(512, rank=lora_rank)
        )
        
        # Style embedding layers with LoRA
        self.style_layers = nn.Sequential(
            LoRALayer(512 * 4 * 4, 1024, rank=lora_rank*2, alpha=lora_alpha, dropout=0.3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            LoRALayer(1024, style_dim, rank=lora_rank, alpha=lora_alpha, dropout=0.2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Character classification head
        self.classifier = LoRALayer(style_dim, num_classes, rank=lora_rank//2, alpha=lora_alpha)
        
        # Style consistency head
        self.style_head = LoRALayer(style_dim, style_dim, rank=lora_rank//2, alpha=lora_alpha)
        
    def forward(self, x):
        # Extract convolutional features
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        
        # Apply residual blocks
        residual_out = self.residual_blocks(conv4_out)
        
        # Flatten for fully connected layers
        flattened = residual_out.view(residual_out.size(0), -1)
        
        # Get style embedding
        style_embedding = self.style_layers(flattened)
        
        # Classification output
        class_logits = self.classifier(style_embedding)
        
        # Style consistency output
        style_output = self.style_head(style_embedding)
        
        return {
            'style_embedding': style_embedding,
            'class_logits': class_logits,
            'style_output': style_output,
            'conv_features': flattened,
            'feature_maps': {
                'conv1': conv1_out,
                'conv2': conv2_out,
                'conv3': conv3_out,
                'conv4': conv4_out,
                'residual': residual_out
            }
        }

class CalligraphyGenerator(nn.Module):
    """Generator network to create calligraphy from style codes with LoRA"""
    
    def __init__(self, style_dim: int = 512, char_embed_dim: int = 128, 
                 output_size: int = 128, num_chars: int = 26, 
                 lora_rank: int = 16, lora_alpha: float = 16.0):
        super().__init__()
        
        self.style_dim = style_dim
        self.char_embed_dim = char_embed_dim
        self.output_size = output_size
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # Character embedding
        self.char_embedding = nn.Embedding(num_chars, char_embed_dim)
        
        # Combined input dimension
        combined_dim = style_dim + char_embed_dim
        
        # Generator layers with LoRA
        self.generator_fc = nn.Sequential(
            LoRALayer(combined_dim, 1024, rank=lora_rank*2, alpha=lora_alpha),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            
            LoRALayer(1024, 2048, rank=lora_rank*2, alpha=lora_alpha),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            
            LoRALayer(2048, 4096, rank=lora_rank*2, alpha=lora_alpha),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
        )
        
        # Convolutional decoder with LoRA
        self.decoder = nn.Sequential(
            LoRAConv2d(256, 128, 4, rank=lora_rank, alpha=lora_alpha, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            LoRAConv2d(128, 64, 4, rank=lora_rank, alpha=lora_alpha, stride=2, padding=1),   # 16x16 -> 32x32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            LoRAConv2d(64, 32, 4, rank=lora_rank, alpha=lora_alpha, stride=2, padding=1),    # 32x32 -> 64x64
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            LoRAConv2d(32, 1, 4, rank=lora_rank//2, alpha=lora_alpha, stride=2, padding=1),  # 64x64 -> 128x128
            nn.Sigmoid()
        )
        
    def forward(self, style_code, char_indices):
        # Get character embeddings
        char_embeds = self.char_embedding(char_indices)
        
        # Combine style and character information
        combined = torch.cat([style_code, char_embeds], dim=1)
        
        # Generate features
        generated_features = self.generator_fc(combined)
        
        # Reshape for convolutional decoder
        batch_size = generated_features.size(0)
        reshaped = generated_features.view(batch_size, 256, 4, 4)
        
        # Generate image
        generated_image = self.decoder(reshaped)
        
        return generated_image

class StyleConsistencyLoss(nn.Module):
    """Custom loss for maintaining style consistency"""
    
    def __init__(self, lambda_style: float = 1.0, lambda_content: float = 1.0, 
                 lambda_perceptual: float = 0.1):
        super().__init__()
        self.lambda_style = lambda_style
        self.lambda_content = lambda_content
        self.lambda_perceptual = lambda_perceptual
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        
    def perceptual_loss(self, pred_features, target_features):
        """Calculate perceptual loss using feature maps"""
        loss = 0.0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += self.mse_loss(pred_feat, target_feat)
        return loss / len(pred_features)
        
    def forward(self, outputs, targets, style_ref=None, perceptual_features=None):
        # Classification loss
        class_loss = self.ce_loss(outputs['class_logits'], targets['labels'])
        
        # Style consistency loss
        if style_ref is not None:
            style_loss = self.mse_loss(outputs['style_output'], style_ref)
        else:
            style_loss = torch.tensor(0.0, device=outputs['style_output'].device)
        
        # Perceptual loss
        if perceptual_features is not None:
            perceptual_loss = self.perceptual_loss(
                list(outputs['feature_maps'].values()),
                perceptual_features
            )
        else:
            perceptual_loss = torch.tensor(0.0, device=outputs['style_output'].device)
        
        # Total loss
        total_loss = (self.lambda_content * class_loss + 
                     self.lambda_style * style_loss + 
                     self.lambda_perceptual * perceptual_loss)
        
        return {
            'total_loss': total_loss,
            'class_loss': class_loss,
            'style_loss': style_loss,
            'perceptual_loss': perceptual_loss
        }

def count_lora_parameters(model):
    """Count the number of LoRA parameters in the model"""
    lora_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'lora_A' in name or 'lora_B' in name:
            lora_params += param.numel()
    
    return lora_params, total_params

def freeze_non_lora_parameters(model):
    """Freeze all parameters except LoRA parameters"""
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False
    
    logger.info("Frozen all non-LoRA parameters")