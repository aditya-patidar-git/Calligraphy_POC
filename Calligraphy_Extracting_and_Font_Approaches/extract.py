import cv2
import easyocr
import os

# Directory where the images are stored
image_dir = "output_images"

# Initialize the EasyOCR reader
reader = easyocr.Reader(["en"], gpu=False)

results = {}

# Loop through all files in the directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(image_dir, filename)

        # Read the image
        img = cv2.imread(filepath)
        if img is None:
            results[filename] = None
            continue

        # Perform OCR
        ocr_result = reader.readtext(img)

        # Collect detected text
        detected_text = " ".join([res[1] for res in ocr_result]).strip()
        results[filename] = detected_text

# Print Results
for filename, text in results.items():
    print(f"{filename}: {text}")

# Results available as the `results` dict
