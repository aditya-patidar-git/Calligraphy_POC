
# %%writefile dataset.py
#!/usr/bin/env python3
"""
Dataset module for calligraphy character images
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import json
import random
from pathlib import Path
import numpy as np

class BrushLetteringDataset(Dataset):
    def __init__(self, data_dir, resolution=512, split='train', augment=True):
        """
        Dataset for brush lettering character images
        
        Expected directory structure:
        data_dir/
        ├── images/
        │   ├── shape_ellipse_72.png
        │   ├── shape_ellipse_118.png
        │   └── ...
        ├── metadata.json (optional)
        └── prompts.json (optional)
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.split = split
        self.augment = augment
        
        # Load image paths and metadata
        self.image_paths = []
        self.prompts = []
        self.characters = []
        
        self._load_data()
        self._setup_transforms()
        
        print(f"Loaded {len(self.image_paths)} images for {split} split")
    
    def _load_data(self):
        """Load image paths and generate prompts"""
        images_dir = self.data_dir 
        
        if not images_dir.exists():
            # Try current directory as images directory
            if self.data_dir.exists():
                images_dir = self.data_dir
            else:
                raise ValueError(f"Images directory not found: {images_dir}")
        
        # Get all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(list(images_dir.glob(f"*{ext}")))
            all_images.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        if not all_images:
            raise ValueError(f"No images found in {images_dir}")
        
        # Sort for consistent ordering
        all_images.sort()
        
        print(f"Found {len(all_images)} total images")
        print(f"Sample filenames: {[img.name for img in all_images[:5]]}")
        
        # Extract shape/character information from filenames
        for img_path in all_images:
            # Handle different filename patterns
            stem = img_path.stem.lower()
            
            # Pattern 1: shape_ellipse_72 -> extract shape type
            if 'shape_' in stem:
                parts = stem.split('_')
                if len(parts) >= 2:
                    shape_type = parts[1]  # 'ellipse', 'circle', etc.
                    self.image_paths.append(img_path)
                    self.characters.append(shape_type)
                    
                    # Generate prompt for this shape
                    prompt = self._generate_shape_prompt(shape_type)
                    self.prompts.append(prompt)
            
            # Pattern 2: character_number (original format)
            elif '_' in stem:
                parts = stem.split('_')
                if len(parts) >= 1:
                    character = parts[0].lower()
                    
                    # Validate character (single letter or common punctuation)
                    if len(character) == 1 and (character.isalpha() or character in '.,!?'):
                        self.image_paths.append(img_path)
                        self.characters.append(character)
                        
                        # Generate prompt for this character
                        prompt = self._generate_character_prompt(character)
                        self.prompts.append(prompt)
            
            # Pattern 3: Single character filename
            elif len(stem) == 1 and (stem.isalpha() or stem in '.,!?'):
                self.image_paths.append(img_path)
                self.characters.append(stem)
                
                prompt = self._generate_character_prompt(stem)
                self.prompts.append(prompt)
        
        print(f"Successfully processed {len(self.image_paths)} images")
        if self.characters:
            print(f"Sample characters/shapes: {list(set(self.characters))[:10]}")
        
        # Split data
        if self.split in ['train', 'val']:
            total_size = len(self.image_paths)
            if total_size == 0:
                print("Warning: No valid images found!")
                return
                
            val_size = max(1, int(total_size * 0.1))  # 10% for validation
            
            if self.split == 'train':
                self.image_paths = self.image_paths[val_size:]
                self.prompts = self.prompts[val_size:]
                self.characters = self.characters[val_size:]
            else:  # validation
                self.image_paths = self.image_paths[:val_size]
                self.prompts = self.prompts[:val_size]
                self.characters = self.characters[:val_size]
        
        # Load custom prompts if available
        prompts_file = self.data_dir / "prompts.json"
        if prompts_file.exists():
            self._load_custom_prompts(prompts_file)
    
    def _generate_shape_prompt(self, shape_type):
        """Generate training prompt for a shape"""
        shape_templates = {
            'ellipse': [
                "geometric ellipse shape, clean line art",
                "oval elliptical form, minimal design",
                "elegant ellipse outline, simple geometry",
                "smooth elliptical shape, black on white",
                "geometric ellipse, mathematical form"
            ],
            'circle': [
                "perfect circle shape, clean geometry",
                "circular form, minimal line art",
                "geometric circle outline, simple design",
                "round circular shape, black on white",
                "mathematical circle, precise form"
            ],
            'rectangle': [
                "rectangular shape, clean geometry",
                "geometric rectangle outline, minimal design",
                "square rectangular form, simple line art",
                "rectangular shape, black on white",
                "geometric rectangle, precise form"
            ],
            'triangle': [
                "triangular shape, clean geometry",
                "geometric triangle outline, minimal design",
                "three-sided triangular form, simple line art",
                "triangular shape, black on white",
                "geometric triangle, precise angles"
            ],
            'square': [
                "square shape, clean geometry",
                "geometric square outline, minimal design",
                "perfect square form, simple line art",
                "square shape, black on white",
                "mathematical square, equal sides"
            ]
        }
        
        # Default template for unknown shapes
        default_templates = [
            f"geometric {shape_type} shape, clean line art",
            f"{shape_type} form, minimal design",
            f"simple {shape_type} outline, geometric shape",
            f"{shape_type} shape, black on white background",
            f"mathematical {shape_type}, precise form"
        ]
        
        templates = shape_templates.get(shape_type, default_templates)
        base_prompt = random.choice(templates)
        
        # Add style modifiers
        style_modifiers = [
            ", high contrast",
            ", clean and minimal",
            ", professional drawing",
            ", precise geometry",
            ", simple line art",
            "", # Sometimes no modifier
            ""
        ]
        
        style = random.choice(style_modifiers)
        return base_prompt + style
    
    def _generate_character_prompt(self, character):
        """Generate training prompt for a character (original method)"""
        base_templates = [
            "brush lettering character '{char}', calligraphy style",
            "handwritten letter '{char}' in brush calligraphy",
            "elegant brush stroke character '{char}'",
            "calligraphic letter '{char}' with brush pen",
            "artistic brush lettering '{char}'",
            "flowing brush calligraphy character '{char}'",
            "hand lettered '{char}' in brush style",
            "brush script letter '{char}'"
        ]
        
        style_modifiers = [
            ", black ink on white paper",
            ", elegant and flowing",
            ", artistic calligraphy",
            ", traditional brush style",
            ", modern brush lettering",
            ", expressive strokes",
            ", clean and minimal",
            ", bold brush strokes"
        ]
        
        quality_modifiers = [
            ", high quality",
            ", professional",
            ", detailed",
            ", sharp and clear",
            ", well composed",
            ", artistic",
            "",
            ""
        ]
        
        base = random.choice(base_templates).format(char=character)
        style = random.choice(style_modifiers)
        quality = random.choice(quality_modifiers)
        
        return base + style + quality
    
    def _load_custom_prompts(self, prompts_file):
        """Load custom prompts from JSON file"""
        try:
            with open(prompts_file, 'r') as f:
                custom_prompts = json.load(f)
            
            # Update prompts if custom ones are provided
            for i, char in enumerate(self.characters):
                if char in custom_prompts:
                    if isinstance(custom_prompts[char], list):
                        self.prompts[i] = random.choice(custom_prompts[char])
                    else:
                        self.prompts[i] = custom_prompts[char]
        except Exception as e:
            print(f"Warning: Could not load custom prompts: {e}")
    
    def _setup_transforms(self):
        """Setup image transformations"""
        transforms_list = []
        
        # Base transforms
        transforms_list.extend([
            transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        
        # Augmentation transforms for training
        if self.augment and self.split == 'train':
            augment_transforms = [
                transforms.RandomRotation(degrees=5, fill=1.0),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3, fill=1.0),
                transforms.RandomAffine(
                    degrees=3,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                    shear=2,
                    fill=1.0
                ),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=1.0),
            ]
            
            # Add augmentations randomly
            for aug in augment_transforms:
                if random.random() < 0.3:
                    transforms_list.insert(-1, aug)
        
        # Normalization (to [-1, 1] range expected by SD)
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
        
        self.transform = transforms.Compose(transforms_list)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        # Convert to RGB if needed (SD expects RGB)
        if image.mode != 'RGB':
            if image.mode == 'L':
                image = ImageOps.colorize(image, black="black", white="white")
            else:
                image = image.convert('RGB')
        
        # Apply transforms
        image = self.transform(image)
        
        # Get prompt
        prompt = self.prompts[idx]
        character = self.characters[idx]
        
        return {
            'images': image,
            'prompts': prompt,
            'characters': character,
            'image_paths': str(img_path)
        }
    
    def get_character_distribution(self):
        """Get distribution of characters/shapes in dataset"""
        from collections import Counter
        return Counter(self.characters)
    
    def save_sample_batch(self, save_dir, num_samples=5):
        """Save sample images and prompts for inspection"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        samples = []
        for i in range(min(num_samples, len(self))):
            sample = self[i]
            
            # Convert tensor back to PIL image
            image_tensor = sample['images']
            # Denormalize
            image_tensor = (image_tensor + 1) / 2
            image_tensor = torch.clamp(image_tensor, 0, 1)
            
            # Convert to PIL
            image_pil = transforms.ToPILImage()(image_tensor)
            
            # Save image
            image_path = save_dir / f"sample_{i}_{sample['characters']}.png"
            image_pil.save(image_path)
            
            samples.append({
                'character': sample['characters'],
                'prompt': sample['prompts'],
                'image_path': str(image_path),
                'original_path': sample['image_paths']
            })
        
        # Save metadata
        with open(save_dir / "samples_metadata.json", 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"Saved {len(samples)} sample images to {save_dir}")


# Utility function to analyze your dataset
def analyze_dataset_files(data_dir):
    """Analyze the files in your dataset directory"""
    data_path = Path(data_dir)
    
    # Look for images in various locations
    possible_dirs = [
        data_path,
        data_path / "images",
        data_path / "data",
        data_path / "train"
    ]
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    for check_dir in possible_dirs:
        if check_dir.exists():
            images = []
            for ext in image_extensions:
                images.extend(list(check_dir.glob(f"*{ext}")))
                images.extend(list(check_dir.glob(f"*{ext.upper()}")))
            
            if images:
                print(f"\nFound {len(images)} images in: {check_dir}")
                print("Sample filenames:")
                for img in images[:10]:
                    print(f"  {img.name}")
                
                # Analyze filename patterns
                patterns = {}
                for img in images:
                    stem = img.stem.lower()
                    if 'shape_' in stem:
                        pattern = "shape_TYPE_NUMBER"
                        shape_type = stem.split('_')[1] if '_' in stem else "unknown"
                        patterns[pattern] = patterns.get(pattern, []) + [shape_type]
                    elif '_' in stem and len(stem.split('_')[0]) == 1:
                        pattern = "CHAR_NUMBER"
                        char = stem.split('_')[0]
                        patterns[pattern] = patterns.get(pattern, []) + [char]
                    else:
                        pattern = "OTHER"
                        patterns[pattern] = patterns.get(pattern, []) + [stem]
                
                print("\nFilename patterns detected:")
                for pattern, examples in patterns.items():
                    unique_examples = list(set(examples))[:5]
                    print(f"  {pattern}: {unique_examples}")
                
                return check_dir, images
    
    print("No images found in any expected directory!")
    return None, []


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset files")
    parser.add_argument("--save_samples", action="store_true", help="Save sample images")
    args = parser.parse_args()
    
    if args.analyze:
        print("Analyzing dataset files...")
        analyze_dataset_files(args.data_dir)
    
    # Test dataset loading
    try:
        dataset = BrushLetteringDataset(args.data_dir, resolution=512, split='train')
        print(f"\nDataset loaded successfully!")
        print(f"Dataset size: {len(dataset)}")
        print(f"Character/Shape distribution: {dataset.get_character_distribution()}")
        
        # Test loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample:")
            print(f"  Character/Shape: {sample['characters']}")
            print(f"  Prompt: {sample['prompts']}")
            print(f"  Image shape: {sample['images'].shape}")
            print(f"  Original path: {sample['image_paths']}")
        
        # Save samples if requested
        if args.save_samples:
            dataset.save_sample_batch("./sample_outputs")
            
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nTry running with --analyze flag first to check your file structure")