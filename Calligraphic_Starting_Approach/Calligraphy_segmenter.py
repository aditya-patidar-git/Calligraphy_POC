import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy import ndimage
import json
import math
from pathlib import Path

class EnhancedCalligraphyCharacterExtractor:
    def __init__(self):
        self.min_char_area = 100  # Increased minimum area for letters
        self.max_char_area = 8000  # Reduced max area to avoid objects
        self.min_aspect_ratio = 0.2  # Letters typically have reasonable aspect ratios
        self.max_aspect_ratio = 4.0
        self.min_width = 15  # Minimum width for a character
        self.min_height = 20  # Minimum height for a character
        
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj
        
    def preprocess_image(self, image_path):
        """Enhanced preprocessing with better text isolation"""
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        original = img.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Try multiple thresholding approaches
        binary_images = []
        
        # Method 1: Adaptive threshold with different parameters
        binary1 = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 8
        )
        binary_images.append(binary1)
        
        binary2 = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, 15, 10
        )
        binary_images.append(binary2)
        
        # Method 2: Otsu's threshold
        _, binary3 = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_images.append(binary3)
        
        # Method 3: Custom threshold based on mean and std
        mean_val = np.mean(filtered)
        std_val = np.std(filtered)
        threshold_val = int(mean_val - 0.5 * std_val)
        _, binary4 = cv2.threshold(filtered, threshold_val, 255, cv2.THRESH_BINARY_INV)
        binary_images.append(binary4)
        
        # Choose the best binary image
        best_binary = self._select_best_binary_for_text(binary_images, filtered)
        
        # Clean up the binary image
        cleaned = self._clean_binary_image(best_binary)
        
        return original, gray, cleaned
    
    def _select_best_binary_for_text(self, binary_images, gray_img):
        """Select binary image that best captures text while avoiding objects"""
        scores = []
        
        for binary in binary_images:
            # Count valid text-like components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            text_score = 0
            object_penalty = 0
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                
                aspect_ratio = h / w if w > 0 else 0
                
                # Check if it looks like text
                if (self.min_char_area <= area <= self.max_char_area and
                    self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio and
                    w >= self.min_width and h >= self.min_height):
                    
                    # Extract the component for further analysis
                    component = binary[y:y+h, x:x+w]
                    
                    # Check if it has text-like characteristics
                    if self._is_text_like(component, area):
                        text_score += 1
                    else:
                        object_penalty += 0.5
                
                # Penalize very large or very small components (likely objects or noise)
                elif area > self.max_char_area or area < 50:
                    object_penalty += 0.2
            
            # Calculate final score
            final_score = text_score - object_penalty
            scores.append(final_score)
        
        # Return the binary with the highest score
        best_idx = np.argmax(scores)
        return binary_images[best_idx]
    
    def _is_text_like(self, component, area):
        """Determine if a component looks like text rather than an object"""
        if component.size == 0:
            return False
        
        h, w = component.shape
        
        # Text characteristics to check:
        
        # 1. Density should be reasonable (not too sparse or too dense)
        density = np.sum(component > 0) / component.size
        if density < 0.05 or density > 0.95:
            return False
        
        # 2. Check for vertical strokes (common in letters)
        vertical_profile = np.sum(component, axis=0)
        has_vertical_variation = np.std(vertical_profile) > np.mean(vertical_profile) * 0.3
        
        # 3. Check for horizontal strokes
        horizontal_profile = np.sum(component, axis=1)
        has_horizontal_variation = np.std(horizontal_profile) > np.mean(horizontal_profile) * 0.3
        
        # 4. Check edge complexity (text has more complex edges than simple objects)
        edges = cv2.Canny(component, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 5. Check for holes (many letters have holes or enclosed areas)
        contours, hierarchy = cv2.findContours(component, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        has_holes = hierarchy is not None and len(hierarchy[0]) > 1
        
        # 6. Circularity check (reject very circular objects like stamps)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.85:  # Very circular, likely not text
                    return False
        
        # Score based on text-like features
        text_score = 0
        if has_vertical_variation:
            text_score += 1
        if has_horizontal_variation:
            text_score += 1
        if 0.01 < edge_density < 0.3:
            text_score += 1
        if has_holes:
            text_score += 0.5
        if 0.1 < density < 0.8:
            text_score += 1
        
        return text_score >= 2.5
    
    def _clean_binary_image(self, binary):
        """Clean the binary image to remove noise and enhance text"""
        # Remove small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        
        # Close small gaps in characters
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        
        return cleaned
    
    def _is_punctuation_or_symbol(self, component):
        """Check if component is likely punctuation or small symbol"""
        if component.size == 0:
            return True
            
        h, w = component.shape
        area = np.sum(component > 0)
        
        # Very small components are likely punctuation
        if area < 80 or w < 10 or h < 15:
            return True
        
        # Check for dot-like shapes (periods, commas)
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio > 3 and area < 200:  # Tall thin shapes like commas
            return True
        
        if aspect_ratio < 1.5 and area < 300:  # Small squarish shapes like periods
            return True
        
        # Check density - punctuation is often very dense or very sparse
        density = area / component.size
        if density > 0.9 or density < 0.03:
            return True
        
        return False
    
    def segment_characters_advanced(self, binary_img, original_img):
        """Advanced character segmentation focusing on letters"""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_img, connectivity=8
        )
        
        # Extract potential character regions
        potential_chars = []
        
        # Get image dimensions for position-based filtering
        img_height, img_width = binary_img.shape
        
        # Define text regions (ignore extreme edges where objects might be)
        text_margin_x = img_width * 0.1
        text_margin_y = img_height * 0.1
        
        for i in range(1, num_labels):  # Skip background
            x, y, w, h, area = stats[i]
            
            # Filter based on size and position
            aspect_ratio = h / w if w > 0 else 0
            
            # Skip if it's in the very edge areas (likely objects)
            in_edge_area = (x < text_margin_x or y < text_margin_y or 
                          x + w > img_width - text_margin_x or 
                          y + h > img_height - text_margin_y)
            
            # Apply stricter filtering for better character detection
            if (self.min_char_area <= area <= self.max_char_area and
                self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio and
                w >= self.min_width and h >= self.min_height):
                
                # Extract component for analysis
                padding = 5
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(binary_img.shape[1], x + w + padding)
                y_end = min(binary_img.shape[0], y + h + padding)
                
                char_binary = binary_img[y_start:y_end, x_start:x_end]
                char_original = original_img[y_start:y_end, x_start:x_end]
                
                # Skip if it's punctuation or small symbol
                if self._is_punctuation_or_symbol(char_binary):
                    continue
                
                # Check if it looks like text
                if not self._is_text_like(char_binary, area):
                    continue
                
                # Calculate confidence
                confidence = self._calculate_text_confidence(char_binary, in_edge_area)
                
                if confidence > 0.3:  # Higher threshold for better quality
                    centroid = (float(centroids[i][0]), float(centroids[i][1]))
                    
                    potential_chars.append({
                        'binary': char_binary,
                        'original': char_original,
                        'bbox': (int(x_start), int(y_start), int(x_end), int(y_end)),
                        'centroid': centroid,
                        'area': int(area),
                        'aspect_ratio': float(aspect_ratio),
                        'confidence': confidence,
                        'in_edge_area': in_edge_area
                    })
        
        # Sort by confidence and reading order (left to right, top to bottom)
        potential_chars.sort(key=lambda c: (-c['confidence'], c['bbox'][1], c['bbox'][0]))
        
        # Remove overlapping characters
        final_chars = self._remove_overlaps(potential_chars, overlap_threshold=0.3)
        
        # If we still don't have enough, try a more lenient approach
        if len(final_chars) < 3:
            print(f"Found only {len(final_chars)} high-confidence characters, trying lenient approach...")
            additional_chars = self._lenient_character_search(binary_img, original_img, final_chars)
            final_chars.extend(additional_chars)
            final_chars = self._remove_overlaps(final_chars, overlap_threshold=0.3)
        
        return final_chars[:3]  # Return top 3 characters
    
    def _calculate_text_confidence(self, char_binary, in_edge_area):
        """Calculate confidence that this is a text character"""
        if char_binary.size == 0:
            return 0.0
        
        h, w = char_binary.shape
        area = np.sum(char_binary > 0)
        
        # Base confidence on multiple factors
        confidence = 0.0
        
        # 1. Size appropriateness
        size_score = 1.0
        if area < self.min_char_area or area > self.max_char_area:
            size_score = 0.1
        confidence += size_score * 0.2
        
        # 2. Aspect ratio appropriateness
        aspect_ratio = h / w if w > 0 else 0
        if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
            confidence += 0.2
        
        # 3. Density appropriateness for text
        density = area / char_binary.size
        if 0.1 <= density <= 0.7:
            confidence += 0.2
        elif 0.05 <= density <= 0.9:
            confidence += 0.1
        
        # 4. Edge complexity (text has complex edges)
        edges = cv2.Canny(char_binary, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if 0.02 <= edge_density <= 0.25:
            confidence += 0.2
        
        # 5. Stroke-like features
        if self._has_stroke_features(char_binary):
            confidence += 0.2
        
        # Penalty for edge areas
        if in_edge_area:
            confidence *= 0.7
        
        return confidence
    
    def _has_stroke_features(self, char_binary):
        """Check if the character has stroke-like features typical of handwriting"""
        if char_binary.size == 0:
            return False
        
        # Apply thinning to find skeleton
        skeleton = morphology.skeletonize(char_binary > 0)
        
        # Check if skeleton has reasonable length compared to area
        skeleton_length = np.sum(skeleton)
        area = np.sum(char_binary > 0)
        
        if area == 0:
            return False
        
        stroke_ratio = skeleton_length / area
        
        # Good text characters typically have stroke ratios in this range
        return 0.05 <= stroke_ratio <= 0.4
    
    def _lenient_character_search(self, binary_img, original_img, existing_chars):
        """More lenient search for additional characters"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_img, connectivity=8
        )
        
        additional_chars = []
        existing_bboxes = [char['bbox'] for char in existing_chars]
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # More lenient criteria
            if (50 <= area <= 12000 and  # Wider area range
                0.1 <= (h/w if w > 0 else 0) <= 6.0 and  # Wider aspect ratio
                w >= 10 and h >= 15):  # Smaller minimum size
                
                padding = 5
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(binary_img.shape[1], x + w + padding)
                y_end = min(binary_img.shape[0], y + h + padding)
                
                char_binary = binary_img[y_start:y_end, x_start:x_end]
                char_original = original_img[y_start:y_end, x_start:x_end]
                
                # Skip punctuation
                if self._is_punctuation_or_symbol(char_binary):
                    continue
                
                # Check for overlap with existing characters
                bbox = (int(x_start), int(y_start), int(x_end), int(y_end))
                if any(self._calculate_overlap(bbox, existing_bbox) > 0.3 for existing_bbox in existing_bboxes):
                    continue
                
                # Basic text check
                density = np.sum(char_binary > 0) / char_binary.size if char_binary.size > 0 else 0
                if 0.05 <= density <= 0.85:
                    centroid = (float(centroids[i][0]), float(centroids[i][1]))
                    
                    additional_chars.append({
                        'binary': char_binary,
                        'original': char_original,
                        'bbox': bbox,
                        'centroid': centroid,
                        'area': int(area),
                        'aspect_ratio': float(h/w if w > 0 else 0),
                        'confidence': float(density * 0.5),  # Lower confidence
                        'in_edge_area': False
                    })
        
        # Sort by area and confidence
        additional_chars.sort(key=lambda c: (-c['area'], -c['confidence']))
        
        return additional_chars[:3-len(existing_chars)]  # Fill up to 3 total
    
    def _calculate_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x1 < x2 and y1 < y2:
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            union = area1 + area2 - intersection
            return intersection / union if union > 0 else 0
        return 0.0
    
    def _remove_overlaps(self, characters, overlap_threshold=0.3):
        """Remove overlapping character detections"""
        if not characters:
            return characters
        
        # Sort by confidence (descending)
        sorted_chars = sorted(characters, key=lambda c: c['confidence'], reverse=True)
        
        final_chars = []
        for char in sorted_chars:
            bbox1 = char['bbox']
            
            # Check if this character overlaps significantly with any accepted character
            overlap = False
            for accepted_char in final_chars:
                bbox2 = accepted_char['bbox']
                
                if self._calculate_overlap(bbox1, bbox2) > overlap_threshold:
                    overlap = True
                    break
            
            if not overlap:
                final_chars.append(char)
        
        return final_chars
    
    def extract_three_characters(self, image_path, output_dir=None):
        """Main function to extract three characters from calligraphy image"""
        image_path = Path(image_path)
        
        if output_dir is None:
            output_dir = image_path.parent / f"{image_path.stem}_characters"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Preprocess image
            original, gray, binary = self.preprocess_image(image_path)
            
            # Segment characters
            characters = self.segment_characters_advanced(binary, original)
            
            if len(characters) == 0:
                print(f"No valid characters found in {image_path}")
                return []
            
            print(f"Successfully extracted {len(characters)} characters from {image_path.name}")
            
            # Save extracted characters
            saved_chars = []
            for i, char_data in enumerate(characters):
                # Save binary version (for training)
                binary_filename = f"char_{i+1}_binary.png"
                binary_path = output_dir / binary_filename
                cv2.imwrite(str(binary_path), char_data['binary'])
                
                # Save original version (for reference)
                original_filename = f"char_{i+1}_original.png"
                original_path = output_dir / original_filename
                cv2.imwrite(str(original_path), char_data['original'])
                
                # Create a clean version with white background
                clean_char = self._create_clean_character(char_data['binary'])
                clean_filename = f"char_{i+1}_clean.png"
                clean_path = output_dir / clean_filename
                cv2.imwrite(str(clean_path), clean_char)
                
                # Prepare character data for JSON serialization
                char_info = {
                    'index': i + 1,
                    'binary_path': str(binary_path),
                    'original_path': str(original_path),
                    'clean_path': str(clean_path),
                    'bbox': char_data['bbox'],
                    'confidence': char_data['confidence'],
                    'area': char_data['area'],
                    'aspect_ratio': char_data['aspect_ratio'],
                    'centroid': char_data['centroid']
                }
                
                saved_chars.append(char_info)
            
            # Save metadata
            metadata = {
                'source_image': str(image_path),
                'characters_found': len(characters),
                'characters': saved_chars,
                'extraction_settings': {
                    'min_char_area': self.min_char_area,
                    'max_char_area': self.max_char_area,
                    'min_width': self.min_width,
                    'min_height': self.min_height
                }
            }
            
            # Convert any remaining numpy types before JSON serialization
            metadata = self._convert_numpy_types(metadata)
            
            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved to: {output_dir}")
            
            return saved_chars
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_clean_character(self, binary_char):
        """Create a clean version of the character with white background"""
        # Invert so character is black on white
        clean = 255 - binary_char
        
        # Resize to standard size while maintaining aspect ratio
        target_size = 64
        h, w = clean.shape
        
        if h > w:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_w = target_size
            new_h = int(h * target_size / w)
        
        # Ensure minimum size
        if new_w < 1:
            new_w = 1
        if new_h < 1:
            new_h = 1
        
        resized = cv2.resize(clean, (new_w, new_h))
        
        # Pad to square
        delta_w = target_size - new_w
        delta_h = target_size - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                   cv2.BORDER_CONSTANT, value=255)
        
        return padded
    
    def visualize_extraction(self, image_path, show_steps=True):
        """Visualize the character extraction process"""
        original, gray, binary = self.preprocess_image(image_path)
        characters = self.segment_characters_advanced(binary, original)
        
        if show_steps:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image
            axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Grayscale
            axes[0, 1].imshow(gray, cmap='gray')
            axes[0, 1].set_title('Grayscale')
            axes[0, 1].axis('off')
            
            # Binary with detected characters marked
            binary_with_boxes = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            for char in characters:
                bbox = char['bbox']
                cv2.rectangle(binary_with_boxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            axes[0, 2].imshow(binary_with_boxes)
            axes[0, 2].set_title('Binary with Detections')
            axes[0, 2].axis('off')
            
            # Show extracted characters
            for i in range(3):
                if i < len(characters):
                    clean_char = self._create_clean_character(characters[i]['binary'])
                    axes[1, i].imshow(clean_char, cmap='gray')
                    axes[1, i].set_title(f'Character {i+1}\nConf: {characters[i]["confidence"]:.3f}')
                else:
                    axes[1, i].set_title('No character found')
                    axes[1, i].imshow(np.ones((64, 64)) * 255, cmap='gray')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.show()
    
    def batch_process(self, image_dir, output_base_dir):
        """Process multiple images in batch"""
        image_dir = Path(image_dir)
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(exist_ok=True)
        
        # Supported image formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        results = []
        processed_count = 0
        
        for img_path in image_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                print(f"\nProcessing: {img_path.name}")
                output_dir = output_base_dir / img_path.stem
                chars = self.extract_three_characters(img_path, output_dir)
                
                results.append({
                    'image': str(img_path),
                    'characters_extracted': len(chars),
                    'characters': chars
                })
                
                processed_count += 1
        
        # Convert any numpy types before saving
        results = self._convert_numpy_types(results)
        
        # Save batch results
        batch_summary = {
            'total_images_processed': processed_count,
            'results': results
        }
        
        with open(output_base_dir / 'batch_results.json', 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        print(f"\nBatch processing complete. Processed {processed_count} images.")
        return results

# Usage example
if __name__ == "__main__":

    # Initialize the enhanced extractor
    extractor = EnhancedCalligraphyCharacterExtractor()

    # Define input and output paths
    input_dir = "./input"
    output_dir = "./outputimages"

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process images in batch
    extractor.batch_process(input_dir, output_dir)
