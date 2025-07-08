#!/usr/bin/env python3
"""
Utility functions for preparing calligraphy images and training
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage
import seaborn as sns

logger = logging.getLogger(__name__)

def create_labels_file(image_dir: str, output_file: str = "labels.json"):
    """
    Create a labels file for character images
    Assumes image naming convention: character_number.extension (e.g., 'a_001.png', 'b_002.jpg')
    """
    image_dir = Path(image_dir)
    labels = {}
    
    # Common image extensions
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    for ext in extensions:
        for img_file in image_dir.glob(f"*{ext}"):
            try:
                # Extract character from filename
                char_name = img_file.stem.split('_')[0]
                if len(char_name) == 1:  # Single character
                    labels[img_file.name] = char_name.lower()
                else:
                    logger.warning(f"Could not extract character from {img_file.name}")
            except Exception as e:
                logger.warning(f"Error processing {img_file.name}: {e}")
    
    # Save labels file
    with open(output_file, 'w') as f:
        json.dump(labels, f, indent=2)
    
    logger.info(f"Created labels file with {len(labels)} entries: {output_file}")
    return labels

def visualize_character_samples(image_dir: str, labels_file: str = None, max_samples: int = 16):
    """
    Visualize character samples to verify data quality
    """
    image_dir = Path(image_dir)
    
    if labels_file and os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
    else:
        labels = create_labels_file(str(image_dir))
    
    # Group images by character
    char_images = {}
    for img_name, char in labels.items():
        if char not in char_images:
            char_images[char] = []
        char_images[char].append(img_name)
    
    # Create visualization
    chars = sorted(char_images.keys())[:max_samples]
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, char in enumerate(chars):
        if i >= len(axes):
            break
            
        # Load first image for this character
        img_name = char_images[char][0]
        img_path = image_dir / img_name
        
        if img_path.exists():
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f"Character: '{char}' ({len(char_images[char])} samples)")
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f"Failed to load\n{char}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f"File not found\n{char}", 
                       ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(chars), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Character Samples from {image_dir}", fontsize=16, y=0.98)
    plt.show()
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Total characters: {len(char_images)}")
    print(f"Total images: {sum(len(imgs) for imgs in char_images.values())}")
    print(f"Characters: {sorted(char_images.keys())}")
    print(f"Samples per character: {[len(imgs) for imgs in char_images.values()]}")

def check_image_quality(image_dir: str, output_report: str = "quality_report.txt"):
    """
    Check image quality and report issues
    """
    image_dir = Path(image_dir)
    issues = []
    stats = {
        'total_images': 0,
        'valid_images': 0,
        'corrupted_images': 0,
        'small_images': 0,
        'large_images': 0,
        'sizes': []
    }
    
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    for ext in extensions:
        for img_file in image_dir.glob(f"*{ext}"):
            stats['total_images'] += 1
            
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    issues.append(f"Corrupted image: {img_file.name}")
                    stats['corrupted_images'] += 1
                    continue
                
                h, w = img.shape[:2]
                stats['sizes'].append((w, h))
                stats['valid_images'] += 1
                
                # Check for very small images
                if w < 32 or h < 32:
                    issues.append(f"Very small image: {img_file.name} ({w}x{h})")
                    stats['small_images'] += 1
                
                # Check for very large images
                if w > 1000 or h > 1000:
                    issues.append(f"Very large image: {img_file.name} ({w}x{h})")
                    stats['large_images'] += 1
                    
            except Exception as e:
                issues.append(f"Error reading {img_file.name}: {e}")
    
    # Calculate size statistics
    if stats['sizes']:
        widths, heights = zip(*stats['sizes'])
        stats['avg_width'] = np.mean(widths)
        stats['avg_height'] = np.mean(heights)
        stats['min_width'] = min(widths)
        stats['max_width'] = max(widths)
        stats['min_height'] = min(heights)
        stats['max_height'] = max(heights)
    
    # Generate report
    report = f"""
Image Quality Report for {image_dir}
=====================================

Summary:
- Total images: {stats['total_images']}
- Valid images: {stats['valid_images']}
- Corrupted images: {stats['corrupted_images']}
- Small images (< 32px): {stats['small_images']}
- Large images (> 1000px): {stats['large_images']}

Size Statistics:
- Average size: {stats.get('avg_width', 0):.1f} x {stats.get('avg_height', 0):.1f}
- Size range: {stats.get('min_width', 0)}-{stats.get('max_width', 0)} x {stats.get('min_height', 0)}-{stats.get('max_height', 0)}

Issues Found:
"""
    
    for issue in issues:
        report += f"- {issue}\n"
    
    if not issues:
        report += "No issues found!\n"
    
    # Save report
    with open(output_report, 'w') as f:
        f.write(report)
    
    print(report)
    logger.info(f"Quality report saved to {output_report}")
    
    return stats, issues

def prepare_dataset(image_dir: str, target_dir: str = "prepared_dataset", 
                   target_size: Tuple[int, int] = (128, 128)):
    """
    Prepare dataset by organizing and preprocessing images
    """
    image_dir = Path(image_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(exist_ok=True)
    
    # Create labels file
    labels = create_labels_file(str(image_dir), str(target_dir / "labels.json"))
    
    # Process each image
    processed_count = 0
    failed_count = 0
    
    for img_name, char in labels.items():
        src_path = image_dir / img_name
        dst_path = target_dir / img_name
        
        try:
            # Load image
            img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Could not load {src_path}")
                failed_count += 1
                continue
            
            # Basic preprocessing
            # Resize while maintaining aspect ratio
            h, w = img.shape
            scale = min(target_size[0] / w, target_size[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create padded image
            padded = np.ones(target_size, dtype=np.uint8) * 255  # White background
            
            # Calculate padding offsets
            y_offset = (target_size[1] - new_h) // 2
            x_offset = (target_size[0] - new_w) // 2
            
            # Place resized image in center
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            
            # Save processed image
            cv2.imwrite(str(dst_path), padded)
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {src_path}: {e}")
            failed_count += 1
    
    logger.info(f"Dataset preparation complete: {processed_count} processed, {failed_count} failed")
    return str(target_dir)

def analyze_stroke_patterns(image_dir: str, labels_file: str = None):
    """
    Analyze stroke patterns across different characters
    """
    from preprocessor import ImagePreprocessor
    
    preprocessor = ImagePreprocessor()
    image_dir = Path(image_dir)
    
    if labels_file and os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
    else:
        labels = create_labels_file(str(image_dir))
    
    # Analyze features for each character
    char_features = {}
    
    for img_name, char in labels.items():
        img_path = image_dir / img_name
        
        if not img_path.exists():
            continue
            
        try:
            _, features = preprocessor.preprocess_image(str(img_path))
            
            if char not in char_features:
                char_features[char] = []
            char_features[char].append(features)
            
        except Exception as e:
            logger.warning(f"Could not analyze {img_path}: {e}")
    
    # Calculate average features per character
    char_avg_features = {}
    for char, feature_list in char_features.items():
        if not feature_list:
            continue
            
        avg_features = {}
        for key in feature_list[0].keys():
            if isinstance(feature_list[0][key], (int, float)):
                avg_features[key] = np.mean([f[key] for f in feature_list])
            else:
                avg_features[key] = feature_list[0][key]  # Take first if not numeric
        
        char_avg_features[char] = avg_features
    
    return char_avg_features

def extract_stroke_features(image: np.ndarray) -> Dict:
    """
    Extract detailed stroke features from a character image
    """
    # Ensure binary image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold if not already binary
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    
    features = {}
    
    # Basic measurements
    features['stroke_area'] = np.sum(binary > 0)
    features['bounding_box_area'] = binary.shape[0] * binary.shape[1]
    features['stroke_density'] = features['stroke_area'] / features['bounding_box_area']
    
    # Contour analysis
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        
        # Contour properties
        features['perimeter'] = cv2.arcLength(main_contour, True)
        features['area'] = cv2.contourArea(main_contour)
        features['compactness'] = (features['perimeter'] ** 2) / (4 * np.pi * features['area']) if features['area'] > 0 else 0
        
        # Convex hull
        hull = cv2.convexHull(main_contour)
        features['convexity'] = cv2.contourArea(main_contour) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(main_contour)
        features['aspect_ratio'] = w / h if h > 0 else 0
        features['extent'] = features['area'] / (w * h) if (w * h) > 0 else 0
    
    # Skeleton analysis for stroke width
    skeleton = skeletonize_image(binary)
    features['skeleton_length'] = np.sum(skeleton > 0)
    features['avg_stroke_width'] = features['stroke_area'] / features['skeleton_length'] if features['skeleton_length'] > 0 else 0
    
    # Directional features
    sobel_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude and direction
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y, sobel_x)
    
    # Dominant stroke directions
    valid_gradients = gradient_direction[gradient_magnitude > np.mean(gradient_magnitude)]
    if len(valid_gradients) > 0:
        # Convert to degrees and normalize
        directions_deg = (valid_gradients * 180 / np.pi) % 180
        hist, bins = np.histogram(directions_deg, bins=18, range=(0, 180))
        features['dominant_direction'] = bins[np.argmax(hist)]
        features['direction_variance'] = np.var(directions_deg)
    else:
        features['dominant_direction'] = 0
        features['direction_variance'] = 0
    
    return features

def skeletonize_image(binary_image: np.ndarray) -> np.ndarray:
    """
    Create skeleton of binary image using morphological operations
    """
    # Simple skeletonization using morphological operations
    skeleton = np.zeros_like(binary_image)
    eroded = binary_image.copy()
    
    while True:
        # Erode the image
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        eroded_new = cv2.erode(eroded, kernel)
        
        # Dilate the eroded image
        dilated = cv2.dilate(eroded_new, kernel)
        
        # Subtract dilated from original eroded
        subset = eroded - dilated
        
        # Union with skeleton
        skeleton = cv2.bitwise_or(skeleton, subset)
        
        # Update eroded image
        eroded = eroded_new.copy()
        
        # Stop when eroded image is empty
        if cv2.countNonZero(eroded) == 0:
            break
    
    return skeleton

def visualize_stroke_analysis(image_dir: str, output_dir: str = "stroke_analysis"):
    """
    Create visualizations of stroke analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze all images
    char_features = analyze_stroke_patterns(image_dir)
    
    if not char_features:
        logger.warning("No character features found")
        return
    
    # Create feature comparison plots
    features_to_plot = ['stroke_density', 'aspect_ratio', 'compactness', 'avg_stroke_width']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(features_to_plot):
        if i >= len(axes):
            break
            
        chars = list(char_features.keys())
        values = [char_features[char].get(feature, 0) for char in chars]
        
        axes[i].bar(chars, values)
        axes[i].set_title(f'{feature.replace("_", " ").title()} by Character')
        axes[i].set_xlabel('Character')
        axes[i].set_ylabel(feature.replace("_", " ").title())
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stroke_features_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create correlation heatmap
    feature_matrix = []
    feature_names = []
    
    for char, features in char_features.items():
        row = []
        if not feature_names:
            feature_names = [k for k, v in features.items() if isinstance(v, (int, float))]
        
        for feature_name in feature_names:
            row.append(features.get(feature_name, 0))
        feature_matrix.append(row)
    
    if feature_matrix:
        feature_df = pd.DataFrame(feature_matrix, columns=feature_names, index=list(char_features.keys()))
        
        plt.figure(figsize=(12, 8))
        correlation_matrix = feature_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
        plt.show()

def create_training_config(image_dir: str, output_file: str = "training_config.json"):
    """
    Create a training configuration file based on dataset analysis
    """
    # Analyze dataset
    stats, issues = check_image_quality(image_dir)
    
    # Create adaptive configuration
    config = {
        "dataset": {
            "image_dir": image_dir,
            "image_size": 128,  # Standard size
            "num_classes": len(set(create_labels_file(image_dir).values())),
            "total_images": stats['valid_images']
        },
        "model": {
            "style_dim": 512,
            "char_embed_dim": 128,
            "lora_rank": 16,
            "lora_alpha": 16.0
        },
        "training": {
            "batch_size": min(16, max(4, stats['valid_images'] // 10)),  # Adaptive batch size
            "epochs": 100,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "val_split": 0.2,
            "patience": 15,
            "save_every": 10
        },
        "loss": {
            "lambda_style": 1.0,
            "lambda_content": 1.0,
            "lambda_perceptual": 0.1
        },
        "augmentation": {
            "rotation_degrees": 5,
            "translate_percent": 0.05,
            "enable_random_affine": True
        },
        "output": {
            "save_dir": "checkpoints",
            "log_dir": "logs"
        }
    }
    
    # Adjust configuration based on dataset size
    if stats['valid_images'] < 50:
        config['training']['epochs'] = 150
        config['training']['learning_rate'] = 5e-4
        logger.info("Small dataset detected - adjusted training parameters")
    
    # Save configuration
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Training configuration saved to {output_file}")
    return config

def setup_directories(base_dir: str = "."):
    """
    Setup directory structure for training
    """
    dirs_to_create = [
        "checkpoints",
        "logs", 
        "outputs",
        "prepared_dataset",
        "visualizations"
    ]
    
    base_path = Path(base_dir)
    created_dirs = []
    
    for dir_name in dirs_to_create:
        dir_path = base_path / dir_name
        dir_path.mkdir(exist_ok=True)
        created_dirs.append(str(dir_path))
    
    logger.info(f"Created directories: {created_dirs}")
    return created_dirs

def validate_setup(image_dir: str, min_images: int = 10) -> bool:
    """
    Validate that the setup is ready for training
    """
    issues = []
    
    # Check if image directory exists
    if not os.path.exists(image_dir):
        issues.append(f"Image directory not found: {image_dir}")
        return False
    
    # Check image count
    labels = create_labels_file(image_dir)
    if len(labels) < min_images:
        issues.append(f"Not enough images: {len(labels)} < {min_images}")
    
    # Check image quality
    stats, quality_issues = check_image_quality(image_dir)
    if stats['corrupted_images'] > 0:
        issues.append(f"Found {stats['corrupted_images']} corrupted images")
    
    # Check for character diversity
    unique_chars = len(set(labels.values()))
    if unique_chars < 2:
        issues.append(f"Need at least 2 different characters, found {unique_chars}")
    
    if issues:
        logger.error("Setup validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("Setup validation passed!")
    return True

# Additional imports needed for some functions
try:
    import pandas as pd
except ImportError:
    logger.warning("pandas not available - some visualization features may not work")
    pd = None