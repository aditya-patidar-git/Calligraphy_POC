#!/usr/bin/env python3
"""
Dataset module for calligraphy character images
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class CalligraphyDataset(Dataset):
    """Dataset for calligraphy character images"""
    
    def __init__(self, image_paths: List[str], labels: List[str], 
                 preprocessor, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.transform = transform
        
        # Create character to index mapping
        unique_chars = list(set(labels))
        unique_chars.sort()  # Sort for consistency
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        logger.info(f"Created dataset with {len(unique_chars)} unique characters: {unique_chars}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
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
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            # Return a dummy sample to avoid breaking the dataloader
            dummy_image = torch.zeros(1, self.preprocessor.target_size[0], self.preprocessor.target_size[1])
            return {
                'image': dummy_image,
                'label': torch.tensor(0, dtype=torch.long),
                'char': 'a',
                'features': {'stroke_width': 1.0, 'curvature': 0.0, 'aspect_ratio': 1.0, 'density': 0.1},
                'path': image_path
            }
    
    def get_num_classes(self):
        """Get number of unique characters"""
        return len(self.char_to_idx)
    
    def get_class_distribution(self):
        """Get distribution of characters in dataset"""
        distribution = {}
        for label in self.labels:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution