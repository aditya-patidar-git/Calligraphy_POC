import cv2
import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

def extract_characters_from_image(image_path, output_dir, threshold_value=150, min_area=50, padding=5):
    """
    Extract individual characters from a text image using contours and bounding rectangles
    
    Args:
        image_path: Path to input text image
        output_dir: Directory to save extracted characters
        threshold_value: Threshold for binary conversion
        min_area: Minimum contour area to consider as a character
        padding: Padding around each character bounding rectangle
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Could not read the image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    # For dark text on light background, use THRESH_BINARY_INV
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get image dimensions
    img_height, img_width = img.shape[:2]
    
    # Store bounding rectangles with their positions for sorting
    char_boxes = []
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Filter out noise and very small contours
        if area < min_area:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter out very thin or very short rectangles (likely noise)
        if w < 3 or h < 3:
            continue
        
        char_boxes.append((x, y, w, h, i))
    
    # Sort characters by position (top to bottom, left to right)
    char_boxes = sort_characters_by_position(char_boxes)
    
    kept_count = 0
    
    for idx, (x, y, w, h, original_idx) in enumerate(char_boxes):
        # Apply padding to bounding rectangle
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        w_padded = min(img_width - x_padded, w + 2 * padding)
        h_padded = min(img_height - y_padded, h + 2 * padding)
        
        # Extract the character region from original image
        char_img = img[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]
        
        # Save the character image
        save_path = os.path.join(output_dir, f"char_{idx:03d}.png")
        cv2.imwrite(save_path, char_img)
        
        # Save character info
        info_path = os.path.join(output_dir, f"char_{idx:03d}_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Character index: {idx}\n")
            f.write(f"Original bounding rect: x={x}, y={y}, w={w}, h={h}\n")
            f.write(f"Padded bounding rect: x={x_padded}, y={y_padded}, w={w_padded}, h={h_padded}\n")
            f.write(f"Character area: {cv2.contourArea(contours[original_idx])}\n")
        
        kept_count += 1
    
    return kept_count

def sort_characters_by_position(char_boxes, line_threshold=20):
    """
    Sort characters by reading order (top to bottom, left to right)
    Groups characters into lines based on y-coordinate similarity
    """
    if not char_boxes:
        return []
    
    # Group characters by lines (similar y-coordinates)
    lines = []
    sorted_by_y = sorted(char_boxes, key=lambda box: box[1])  # Sort by y-coordinate
    
    current_line = [sorted_by_y[0]]
    current_y = sorted_by_y[0][1]
    
    for box in sorted_by_y[1:]:
        x, y, w, h, idx = box
        
        # If y-coordinate is close to current line, add to current line
        if abs(y - current_y) <= line_threshold:
            current_line.append(box)
        else:
            # Start a new line
            lines.append(current_line)
            current_line = [box]
            current_y = y
    
    # Don't forget the last line
    if current_line:
        lines.append(current_line)
    
    # Sort each line by x-coordinate (left to right)
    sorted_chars = []
    for line in lines:
        line_sorted = sorted(line, key=lambda box: box[0])  # Sort by x-coordinate
        sorted_chars.extend(line_sorted)
    
    return sorted_chars

def create_debug_image(image_path, output_dir, threshold_value=150, min_area=50):
    """
    Create a debug image showing detected character bounding boxes
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy for drawing
    debug_img = img.copy()
    
    char_boxes = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 3 or h < 3:
            continue
        
        char_boxes.append((x, y, w, h, i))
    
    # Sort characters
    char_boxes = sort_characters_by_position(char_boxes)
    
    # Draw bounding boxes with indices
    for idx, (x, y, w, h, original_idx) in enumerate(char_boxes):
        # Draw rectangle
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw index number
        cv2.putText(debug_img, str(idx), (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Save debug image
    debug_path = os.path.join(output_dir, "debug_character_boxes.png")
    cv2.imwrite(debug_path, debug_img)
    
    # Save threshold image for reference
    thresh_path = os.path.join(output_dir, "threshold_image.png")
    cv2.imwrite(thresh_path, thresh)

def extract_characters_with_morphology(image_path, output_dir, threshold_value=150, min_area=50, padding=5):
    """
    Enhanced character extraction with morphological operations to clean up the image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Could not read the image from {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    
    # Remove noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Fill gaps in characters
    kernel_close = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_height, img_width = img.shape[:2]
    char_boxes = []
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 3 or h < 3:
            continue
        
        # Additional filtering based on aspect ratio (optional)
        aspect_ratio = float(w) / h
        if aspect_ratio > 5 or aspect_ratio < 0.1:  # Filter out very wide or very tall shapes
            continue
        
        char_boxes.append((x, y, w, h, i))
    
    char_boxes = sort_characters_by_position(char_boxes)
    
    kept_count = 0
    for idx, (x, y, w, h, original_idx) in enumerate(char_boxes):
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        w_padded = min(img_width - x_padded, w + 2 * padding)
        h_padded = min(img_height - y_padded, h + 2 * padding)
        
        char_img = img[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]
        
        save_path = os.path.join(output_dir, f"char_{idx:03d}.png")
        cv2.imwrite(save_path, char_img)
        kept_count += 1
    
    return kept_count

def main():
    parser = argparse.ArgumentParser(description="Extract individual characters from text images")
    parser.add_argument("image_path", help="Path to the input text image")
    parser.add_argument("--output", "-o", default="extracted_characters", help="Output directory")
    parser.add_argument("--threshold", "-t", type=int, default=150, help="Binary threshold value")
    parser.add_argument("--min-area", "-a", type=int, default=50, help="Minimum character area")
    parser.add_argument("--padding", "-p", type=int, default=5, help="Padding around characters")
    parser.add_argument("--morphology", "-m", action="store_true", help="Use morphological operations")
    parser.add_argument("--debug", "-d", action="store_true", help="Create debug images")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing image: {args.image_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Create debug images if requested
        if args.debug:
            print("Creating debug images...")
            create_debug_image(args.image_path, str(output_dir), args.threshold, args.min_area)
        
        # Extract characters
        if args.morphology:
            print("Using morphological operations...")
            kept = extract_characters_with_morphology(
                args.image_path, str(output_dir), 
                args.threshold, args.min_area, args.padding
            )
        else:
            kept = extract_characters_from_image(
                args.image_path, str(output_dir), 
                args.threshold, args.min_area, args.padding
            )
        
        print(f"Successfully extracted {kept} characters")
        print(f"Characters saved in: {output_dir}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

# Example usage function for direct calling
def extract_from_image(image_path, output_dir="extracted_characters"):
    """
    Simple function to extract characters from an image
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create debug image
    create_debug_image(image_path, output_dir)
    
    # Extract characters
    kept = extract_characters_from_image(image_path, output_dir)
    
    print(f"Extracted {kept} characters from {image_path}")
    return kept

if __name__ == "__main__":
    # You can either run with command line arguments or uncomment the line below
    # and provide your image path directly
    
    # For direct testing, uncomment and modify this line:
    extract_from_image(r"C:\Users\AdityaPatidar\Documents\Calligraphy_Approaches\images\img.png", "output_characters")
    
    main()