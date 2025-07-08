import cv2
import pytesseract
import os
from glob import glob

# Set path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\AdityaPatidar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Input and output folders
input_dir = r'C:\Users\AdityaPatidar\Documents\Calligraphic_Agent\input'
output_dir = 'char_images'
os.makedirs(output_dir, exist_ok=True)

# Supported image types
image_files = glob(os.path.join(input_dir, '*.[jp][pn]g'))

print(f"Found {len(image_files)} images to process.")

for image_path in image_files:
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\nProcessing: {image_name}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Skipped invalid image: {image_path}")
        continue

    # Preprocess
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = h / float(w) if w != 0 else 0

        if (
            w < 10 or h < 10 or area < 300 or area > 15000 or
            x < 20 or x + w > image.shape[1] - 10 or
            aspect_ratio > 5 or aspect_ratio < 0.2
        ):
            continue

        roi = image[y:y+h, x:x+w]
        save_path = os.path.join(output_dir, f"{image_name}_char_{count+1}.png")
        cv2.imwrite(save_path, roi)
        count += 1

        if count == 3:
            break

    print(f"✅ Saved {count} character(s) from {image_name} to {output_dir}")
