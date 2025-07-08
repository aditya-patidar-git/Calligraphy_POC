import cv2
import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

def preprocess_image(img, method='adaptive'):
    """
    Preprocess image with different methods for different text styles
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if method == 'adaptive':
        # Good for varying lighting and handwritten text
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
    elif method == 'otsu':
        # Good for clear text with consistent lighting
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Manual threshold
        _, thresh = cv2.threshold(gray, method, 255, cv2.THRESH_BINARY_INV)
    
    return gray, thresh

def extract_words_calligraphy(image_path, output_dir, padding=15):
    """
    Extract words from calligraphic/handwritten text
    Uses adaptive thresholding and gentle morphological operations
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Could not read the image from {image_path}")
    
    # Use adaptive thresholding for handwritten text
    gray, thresh = preprocess_image(img, 'adaptive')
    
    # Gentle morphological operations to connect broken characters
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small)
    
    # Horizontal kernel to connect characters into words (smaller for calligraphy)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    dilated = cv2.dilate(thresh, kernel_horizontal, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return process_contours(img, contours, output_dir, padding, min_area=300, min_width=15)

def extract_words_printed(image_path, output_dir, padding=10):
    """
    Extract words from printed/clear text
    Uses OTSU thresholding and stronger morphological operations
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Could not read the image from {image_path}")
    
    # Use OTSU thresholding for clear text
    gray, thresh = preprocess_image(img, 'otsu')
    
    # Stronger morphological operations for printed text
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    
    # Horizontal kernel to connect characters
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    dilated = cv2.dilate(thresh, kernel_horizontal, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return process_contours(img, contours, output_dir, padding, min_area=500, min_width=20)

def extract_words_mixed_dense(image_path, output_dir, padding=12):
    """
    Extract words from dense mixed text (like menus)
    Uses projection-based approach with careful spacing analysis
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Could not read the image from {image_path}")
    
    gray, thresh = preprocess_image(img, 'adaptive')
    
    # Clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return extract_words_by_projection(img, thresh, output_dir, padding)

def extract_words_by_projection(img, thresh, output_dir, padding):
    """
    Extract words using horizontal and vertical projection analysis
    """
    # Find text lines using horizontal projection
    horizontal_projection = np.sum(thresh, axis=1)
    
    # Find line boundaries with minimum line height
    line_boundaries = []
    in_line = False
    start_y = 0
    min_line_height = 10
    
    for y, pixel_count in enumerate(horizontal_projection):
        if pixel_count > 0 and not in_line:
            start_y = y
            in_line = True
        elif pixel_count == 0 and in_line:
            if y - start_y >= min_line_height:  # Only keep lines with minimum height
                line_boundaries.append((start_y, y))
            in_line = False
    
    if in_line and len(horizontal_projection) - start_y >= min_line_height:
        line_boundaries.append((start_y, len(horizontal_projection)))
    
    kept_count = 0
    
    # Process each line
    for line_idx, (start_y, end_y) in enumerate(line_boundaries):
        line_thresh = thresh[start_y:end_y, :]
        vertical_projection = np.sum(line_thresh, axis=0)
        
        # Adaptive gap threshold based on line characteristics
        line_height = end_y - start_y
        gap_threshold = max(3, line_height // 8)  # Adaptive gap threshold
        
        # Find word boundaries
        word_boundaries = []
        in_word = False
        start_x = 0
        
        for x, pixel_count in enumerate(vertical_projection):
            if pixel_count > 0 and not in_word:
                start_x = x
                in_word = True
            elif pixel_count == 0 and in_word:
                # Look ahead to see if this is a real gap
                gap_width = 0
                temp_x = x
                while temp_x < len(vertical_projection) and vertical_projection[temp_x] == 0:
                    gap_width += 1
                    temp_x += 1
                
                if gap_width >= gap_threshold or temp_x == len(vertical_projection):
                    word_boundaries.append((start_x, x))
                    in_word = False
        
        if in_word:
            word_boundaries.append((start_x, len(vertical_projection)))
        
        # Extract words from this line
        for word_idx, (start_x, end_x) in enumerate(word_boundaries):
            w = end_x - start_x
            h = end_y - start_y
            
            # More lenient filtering for dense text
            if w * h < 100 or w < 8 or h < 8:
                continue
            
            # Add padding
            x_start = max(0, start_x - padding)
            y_start = max(0, start_y - padding)
            x_end = min(img.shape[1], end_x + padding)
            y_end = min(img.shape[0], end_y + padding)
            
            # Extract word
            word_img = img[y_start:y_end, x_start:x_end]
            
            # Save with descriptive filename
            save_path = os.path.join(output_dir, f"line_{line_idx:02d}_word_{word_idx:03d}.png")
            cv2.imwrite(save_path, word_img)
            kept_count += 1
            
            print(f"Line {line_idx+1}, Word {word_idx+1}: {w}x{h} pixels")
    
    return kept_count

def process_contours(img, contours, output_dir, padding, min_area=300, min_width=15):
    """
    Process contours and extract word regions
    """
    # Sort contours by position (top to bottom, left to right)
    def sort_contours(cnts):
        bounding_boxes = [cv2.boundingRect(c) for c in cnts]
        sorted_contours = sorted(zip(cnts, bounding_boxes), 
                               key=lambda x: (x[1][1] // 50, x[1][0]))
        return [c[0] for c in sorted_contours]
    
    sorted_contours = sort_contours(contours)
    kept_count = 0
    
    for i, cnt in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter based on size
        area = w * h
        if area < min_area or w < min_width or h < 8:
            continue
        
        # Add padding
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)
        
        # Extract word region
        word_img = img[y_start:y_end, x_start:x_end]
        
        # Save word
        save_path = os.path.join(output_dir, f"word_{i:03d}.png")
        cv2.imwrite(save_path, word_img)
        kept_count += 1
        
        print(f"Word {i+1}: {w}x{h} pixels")
    
    return kept_count

def auto_detect_text_type(image_path):
    """
    Automatically detect text type based on image characteristics
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate image statistics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # Try different thresholding methods
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
    
    # Count white pixels (text pixels)
    otsu_text_pixels = np.sum(thresh_otsu > 0)
    adaptive_text_pixels = np.sum(thresh_adaptive > 0)
    
    # Calculate text density
    total_pixels = gray.shape[0] * gray.shape[1]
    text_density = otsu_text_pixels / total_pixels
    
    # Decision logic
    if text_density > 0.3:  # Very dense text
        return 'dense'
    elif std_intensity > 50 and mean_intensity > 200:  # High contrast, light background
        return 'printed'
    else:  # Default to calligraphy for handwritten/artistic text
        return 'calligraphy'

def extract_words_auto(image_path, output_dir, padding=12):
    """
    Automatically choose the best extraction method
    """
    text_type = auto_detect_text_type(image_path)
    print(f"Detected text type: {text_type}")
    
    if text_type == 'dense':
        return extract_words_mixed_dense(image_path, output_dir, padding)
    elif text_type == 'printed':
        return extract_words_printed(image_path, output_dir, padding)
    else:
        return extract_words_calligraphy(image_path, output_dir, padding)

def main(input_dir, output_dir, method='auto', padding=12):
    """
    Main function to process all images
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save extracted words
        method: Extraction method ('auto', 'calligraphy', 'printed', 'dense')
        padding: Padding around extracted words
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
        image_files.extend(list(Path(input_dir).glob(ext)))
        image_files.extend(list(Path(input_dir).glob(ext.upper())))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    total_words = 0
    
    for idx, image_file in enumerate(image_files):
        print(f"\n{'='*50}")
        print(f"Processing {image_file.name}...")
        print(f"{'='*50}")
        
        output_subdir = Path(output_dir) / f"image_{idx:02d}_{image_file.stem}"
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        try:
            if method == 'auto':
                kept = extract_words_auto(str(image_file), str(output_subdir), padding)
            elif method == 'calligraphy':
                kept = extract_words_calligraphy(str(image_file), str(output_subdir), padding)
            elif method == 'printed':
                kept = extract_words_printed(str(image_file), str(output_subdir), padding)
            elif method == 'dense':
                kept = extract_words_mixed_dense(str(image_file), str(output_subdir), padding)
            else:
                print(f"Unknown method: {method}")
                continue
            
            print(f"\n✓ {image_file.name}: Extracted {kept} words")
            total_words += kept
            
        except Exception as e:
            print(f"✗ Error processing {image_file.name}: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total words extracted: {total_words}")
    print(f"Output directory: {output_dir}")
    print(f"Ready for LoRA training!")

if __name__ == "__main__":
    print("Word Extractor for LoRA Training")
    print("=" * 40)
    main("images/test_images", "output_words", "auto", 15)