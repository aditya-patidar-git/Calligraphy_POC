import cv2
import pytesseract
from PIL import Image
import numpy as np

# ✅ Optional: If tesseract is NOT in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Path\to\tesseract.exe'

def preprocess_for_calligraphy(image_path):
    """Load and preprocess calligraphy-style image for better OCR results."""
    # Read the image
    img = cv2.imread(image_path)

    # 1️⃣ Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2️⃣ Apply bilateral filter for denoising while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # 3️⃣ Threshold to create a clean binary image
    _, thresh = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4️⃣ Optional: Resize for better OCR (sometimes improves accuracy for stylized text)
    scale_factor = 1.5
    new_size = (int(thresh.shape[1] * scale_factor), int(thresh.shape[0] * scale_factor))
    resized = cv2.resize(thresh, new_size, interpolation=cv2.INTER_CUBIC)

    return resized

def extract_calligraphy_text(image_path, lang='eng'):
    """Extract text from calligraphy-style image using Tesseract."""
    preprocessed = preprocess_for_calligraphy(image_path)

    # ✅ Perform OCR
    text = pytesseract.image_to_string(preprocessed, lang=lang)

    return text.strip()

if __name__ == '__main__':
    image_path = 'img.png'  # <-- Replace with your image
    text = extract_calligraphy_text(image_path)

    print("\nExtracted Text:\n------------------\n", text)
