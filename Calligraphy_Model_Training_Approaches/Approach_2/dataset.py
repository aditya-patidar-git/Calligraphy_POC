# %%writefile /kaggle/working/dataset.py
#!/usr/bin/env python3
# Cell 3: Custom Dataset Class for LoRA Training
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import json
from PIL import Image
from transformers import CLIPTokenizer
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class CalligraphyDataset(Dataset):
    """Custom dataset for calligraphy LoRA training"""
    
    def __init__(self, 
                 images_dir: str,
                 metadata_path: str,
                 tokenizer: CLIPTokenizer,
                 resolution: int = 512,
                 flip_p: float = 0.5):
        
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.flip_p = flip_p
        
        # Load metadata
        self.data = []
        with open(metadata_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Validation transforms (no augmentation)
        self.val_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        logger.info(f"Loaded {len(self.data)} training samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = self.images_dir / item["file_name"]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        image = self.image_transforms(image)
        
        # Tokenize text
        text = item["text"]
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids[0],
            "attention_mask": text_inputs.attention_mask[0],
            "text": text,
            "type": item["type"],
            "content": item["content"]
        }

class DataCollator:
    """Custom data collator for batch processing"""
    
    def __init__(self, tokenizer: CLIPTokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, examples):
        batch = {}
        batch["pixel_values"] = torch.stack([example["pixel_values"] for example in examples])
        batch["input_ids"] = torch.stack([example["input_ids"] for example in examples])
        batch["attention_mask"] = torch.stack([example["attention_mask"] for example in examples])
        
        return batch

def create_data_loaders(images_dir: str, 
                       metadata_path: str, 
                       tokenizer: CLIPTokenizer,
                       batch_size: int = 1,
                       validation_split: float = 0.1,
                       resolution: int = 512) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    # Load all data
    full_dataset = CalligraphyDataset(
        images_dir=images_dir,
        metadata_path=metadata_path,
        tokenizer=tokenizer,
        resolution=resolution
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data collator
    collator = DataCollator(tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # Set to 0 for Kaggle compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info(f"Created data loaders: {len(train_dataset)} train, {len(val_dataset)} validation samples")
    
    return train_loader, val_loader

def preview_dataset(dataset: CalligraphyDataset, num_samples: int = 3):
    """Preview dataset samples"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        # Convert tensor back to image
        image = sample["pixel_values"]
        image = (image + 1.0) / 2.0  # Denormalize
        image = torch.clamp(image, 0, 1)
        image = transforms.ToPILImage()(image)
        
        axes[i].imshow(image)
        axes[i].set_title(f"Type: {sample['type']}\nContent: {sample['content']}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print sample prompts
    print("\nSample prompts:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"{i+1}. {sample['text']}")

print("Custom dataset classes loaded successfully!")
print("Ready to create data loaders from your processed images.")