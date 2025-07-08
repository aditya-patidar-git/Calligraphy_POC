import cv2
import os
import torch
import json
from PIL import Image
import easyocr
from transformers import CLIPProcessor, CLIPModel

# --- CONFIG ---
image_path = r'C:\Users\AdityaPatidar\Documents\Calligraphic_Agent\input\img.png'
output_dir = "char_images_easyocr"
os.makedirs(output_dir, exist_ok=True)

# Load EasyOCR and CLIP
print("Loading models...")
reader = easyocr.Reader(['en'])
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

all_char_data = []
image_name = os.path.splitext(os.path.basename(image_path))[0]
print(f"\nProcessing: {image_name}")

# Preprocess image to improve OCR detection
orig_image = cv2.imread(image_path)
gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
preprocessed_path = "temp_preprocessed.png"
cv2.imwrite(preprocessed_path, thresh)

results = reader.readtext(preprocessed_path, detail=1)
count = 0

for (bbox, text, confidence) in results:
    if confidence < 0.2 or len(text.strip()) == 0:
        continue

    (tl, tr, br, bl) = bbox
    x_min = int(min(tl[0], bl[0]))
    y_min = int(min(tl[1], tr[1]))
    x_max = int(max(tr[0], br[0]))
    y_max = int(max(bl[1], br[1]))

    roi = orig_image[y_min:y_max, x_min:x_max]
    char_img_path = os.path.join(output_dir, f"{image_name}_char_{count+1}.png")
    cv2.imwrite(char_img_path, roi)

    # Encode style using CLIP
    pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        style_vector = clip_model.get_image_features(**inputs)
    style_vector = style_vector / style_vector.norm(p=2, dim=-1, keepdim=True)

    char_data = {
        "image": char_img_path,
        "text": text,
        "confidence": float(confidence),
        "style_vector": style_vector.squeeze().tolist()
    }
    all_char_data.append(char_data)
    count += 1

    if count == 3:
        break

print(f"\n✅ Saved {count} characters from {image_name}")

# Save metadata
with open("char_metadata_easyocr.json", "w") as f:
    json.dump(all_char_data, f, indent=2)

print("\n✅ All EasyOCR characters and style embeddings saved.")
