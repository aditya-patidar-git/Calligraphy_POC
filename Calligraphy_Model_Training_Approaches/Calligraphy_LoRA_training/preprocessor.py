#!/usr/bin/env python3
"""
Image preprocessing module for calligraphy character images
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List
import logging

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
    
    def enhance_strokes(self, image: np.ndarray) -> np.ndarray:
        """Enhance stroke visibility and smoothness"""
        # Apply Gaussian blur to smooth strokes
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Apply morphological operations to connect broken strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        return opened
    
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
        
        # Enhance strokes
        enhanced = self.enhance_strokes(binary)
        
        # Noise removal
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
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