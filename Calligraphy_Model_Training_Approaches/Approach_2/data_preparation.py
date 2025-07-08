# %%writefile /kaggle/working/data_preparation.py
#!/usr/bin/env python3
# Cell 2: Data Preparation and Processing Functions

import cv2
import numpy as np
import re
import json
from pathlib import Path
import logging
from tqdm import tqdm
from PIL import Image 
from typing import Dict

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image preprocessing for calligraphy training"""
    
    def __init__(self, target_size: int = 512):
        self.target_size = target_size
        
    def preprocess_image(self, image_path: str) -> Image.Image:
        """Preprocess calligraphy image for training"""
        img = Image.open(image_path).convert('RGB')
        
        # Convert to grayscale first for better text detection
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding for better text extraction
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to PIL Image
        processed = Image.fromarray(binary).convert('RGB')
        
        # Resize while maintaining aspect ratio
        processed = self._resize_with_padding(processed)
        
        return processed
    
    def _resize_with_padding(self, img: Image.Image) -> Image.Image:
        """Resize image to target size with padding"""
        # Calculate aspect ratio
        w, h = img.size
        aspect = w / h
        
        if aspect > 1:  # Width > Height
            new_w = self.target_size
            new_h = int(self.target_size / aspect)
        else:  # Height >= Width
            new_h = self.target_size
            new_w = int(self.target_size * aspect)
        
        # Resize image
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create white background
        background = Image.new('RGB', (self.target_size, self.target_size), 'white')
        
        # Paste resized image in center
        x_offset = (self.target_size - new_w) // 2
        y_offset = (self.target_size - new_h) // 2
        background.paste(img, (x_offset, y_offset))
        
        return background

class DatasetBuilder:
    """Builds training dataset from character and word images"""
    
    def __init__(self, style_name: str = "calligraphy"):
        self.style_name = style_name
        self.processor = ImageProcessor()
        
    def extract_character_from_filename(self, filename: str) -> str:
        """Extract character from filename"""
        # Remove extension and clean filename
        char = Path(filename).stem
        # Handle special characters and clean up
        char = re.sub(r'[^a-zA-Z0-9]', '', char)
        return char.lower() if char else 'unknown'
    
    def extract_word_from_filename(self, filename: str) -> str:
        """Extract word from filename"""
        word = Path(filename).stem
        # Clean and normalize
        word = re.sub(r'[^a-zA-Z0-9\s]', '', word)
        return word.lower().strip() if word else 'unknown'
    
    def create_training_data(self, 
                           character_dir: str, 
                           word_dir: str, 
                           output_dir: str) -> Dict:
        """Create training dataset with proper structure"""
        
        output_path = Path(output_dir)
        images_dir = output_path / "images"
        metadata_path = output_path / "metadata.jsonl"
        
        # Create directories
        images_dir.mkdir(parents=True, exist_ok=True)
        
        training_data = []
        image_counter = 0
        
        # Process character images
        char_dir = Path(character_dir)
        if char_dir.exists():
            logger.info(f"Processing character images from {char_dir}")
            for img_file in tqdm(char_dir.glob("*.png"), desc="Processing characters"):
                try:
                    char = self.extract_character_from_filename(img_file.name)
                    if char and char != 'unknown':
                        # Process and save image
                        processed_img = self.processor.preprocess_image(str(img_file))
                        
                        # Save processed image
                        output_filename = f"char_{image_counter:06d}.png"
                        output_filepath = images_dir / output_filename
                        processed_img.save(output_filepath)
                        
                        # Create training prompt
                        prompt = f"a {self.style_name} style calligraphy letter '{char}', black ink on white background, high quality handwriting"
                        
                        training_data.append({
                            "file_name": output_filename,
                            "text": prompt,
                            "type": "character",
                            "content": char
                        })
                        
                        image_counter += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing character {img_file}: {e}")
        
        # Process word images
        word_dir = Path(word_dir)
        if word_dir.exists():
            logger.info(f"Processing word images from {word_dir}")
            for img_file in tqdm(word_dir.glob("*.png"), desc="Processing words"):
                try:
                    word = self.extract_word_from_filename(img_file.name)
                    if word and word != 'unknown':
                        # Process and save image
                        processed_img = self.processor.preprocess_image(str(img_file))
                        
                        # Save processed image
                        output_filename = f"word_{image_counter:06d}.png"
                        output_filepath = images_dir / output_filename
                        processed_img.save(output_filepath)
                        
                        # Create training prompt
                        prompt = f"a {self.style_name} style calligraphy word '{word}', black ink on white background, elegant handwriting"
                        
                        training_data.append({
                            "file_name": output_filename,
                            "text": prompt,
                            "type": "word",
                            "content": word
                        })
                        
                        image_counter += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing word {img_file}: {e}")
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Created {len(training_data)} training samples")
        logger.info(f"Data saved to {output_path}")
        
        return {
            "total_samples": len(training_data),
            "images_dir": str(images_dir),
            "metadata_path": str(metadata_path),
            "training_data": training_data
        }

# Initialize the dataset builder
dataset_builder = DatasetBuilder(style_name="gothic_calligraphy")

print("Data preparation functions loaded successfully!")
print("Ready to process your character and word images.")