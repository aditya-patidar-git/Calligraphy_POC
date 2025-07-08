import cv2
import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

def extract_shapes_for_image(image_path, output_dir, threshold_value, min_area, scale_factor):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Could not read the image from {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept_count = 0

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area or len(cnt) < 5:
            continue
        (center_x, center_y), (width, height), angle = cv2.fitEllipse(cnt)
        scaled_ellipse = ((center_x, center_y), (width * scale_factor, height * scale_factor), angle)
        mask = np.zeros_like(gray)
        cv2.ellipse(mask, scaled_ellipse, 255, -1) 
        result = cv2.bitwise_and(img, img, mask=mask)
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = result[y:y + h, x:x + w]
        save_path = os.path.join(output_dir, f"shape_ellipse_{i}.png")
        cv2.imwrite(save_path, cropped)
        kept_count += 1
    return kept_count

def main(input_dir, output_dir, threshold_value, min_area, scale_factor):
    image_files = list(Path(input_dir).glob("*.png")) + list(Path(input_dir).glob("*.jpg"))
    for idx, image_file in enumerate(image_files):
        output_subdir = Path(output_dir) / f"image_{idx}"
        output_subdir.mkdir(parents=True, exist_ok=True)
        kept = extract_shapes_for_image(str(image_file), str(output_subdir), threshold_value, min_area, scale_factor)
        print(f"{image_file.name}: Kept {kept} shapes")


if __name__ == "__main__":
    main("images", "output_images", 150, 100, 1.5)
