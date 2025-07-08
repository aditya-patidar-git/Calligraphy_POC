import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import easyocr
import os

# Paths
sam_checkpoint = "checkpoints/sam_vit_b_01ec64.pth"  # Path to SAM checkpoint
image_path = "img.png"                               # Path to your test image
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1️⃣ Load SAM Model
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device)

# 2️⃣ Initialize the Predictor
predictor = SamPredictor(sam)

# 3️⃣ Read and Set the Image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# 4️⃣ Choose a Point for Segmentation
# Adjust these coordinates for the area you want
input_point = np.array([[100, 100]])
input_label = np.array([1])

# 5️⃣ Get the Segmentation Masks
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

mask = masks[0]

# 6️⃣ Save the Mask
mask_image = (mask * 255).astype(np.uint8)
mask_file = "mask.png"
cv2.imwrite(mask_file, mask_image)
print(f"✅ Done! Mask saved to {mask_file}")

# 7️⃣ Optional: Overlay the mask
overlay = image.copy()
overlay[mask == 1] = (0, 255, 0)  # Color segmented area in green
cv2.imwrite("mask_overlay.png", overlay)
print(f"✅ Overlay saved as mask_overlay.png.")

# 8️⃣ Extract Cropped Region
y_indices, x_indices = mask.nonzero()
x_min, x_max = x_indices.min(), x_indices.max()
y_min, y_max = y_indices.min(), y_indices.max()
cropped_img = image[y_min:y_max, x_min:x_max]

# 9️⃣ Perform OCR using EasyOCR
reader = easyocr.Reader(["en"])
results = reader.readtext(cropped_img)

# 1️⃣0️⃣ Print and Save Results
if results:
    for (bbox, text, conf) in results:
        print(f"Text: {text}, Confidence: {conf}")

    # Save results
    results_file = "results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        for (bbox, text, conf) in results:
            f.write(f"Text: {text}, Confidence: {conf}\n")
    print(f"✅ OCR results saved to {results_file}")

else:
    print("❌ No text detected.")
