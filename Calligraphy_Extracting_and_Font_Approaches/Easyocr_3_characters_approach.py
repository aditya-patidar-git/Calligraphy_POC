import cv2
import os
import numpy as np
from pathlib import Path
import easyocr
from collections import Counter, defaultdict

def extract_characters(image_path, output_dir="./output_chars"):
    """Perform OCR, find top 3 chars, and save cropped instances of those chars."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1️⃣ Read and preprocess
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2️⃣ Perform OCR
    reader = easyocr.Reader(['en', 'hi'], gpu=False)  # Adjust language as needed
    results = reader.readtext(image_path)

    if not results:
        print("❌ No text detected by OCR.")
        return

    # ✅ Print detected text and confidence
    print("\n👀 OCR Results (with confidence):")
    for (bbox, text, conf) in results:
        print(f"  Text: '{text}' | Conf: {conf}")

    # ✅ Extract text (using lower threshold for inclusion)
    detected_characters = [res[1] for res in results if res[2] > 0.3]
    all_text = ''.join(detected_characters).replace(' ', '')
    character_counts = Counter(all_text)

    if not character_counts:
        print("❌ No character counts found.")
        return

    top_characters = [char for char, cnt in character_counts.most_common(3)]
    print(f"\n🏁 Top 3 Characters: {top_characters}")

    # 3️⃣ Threshold the image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cropped_characters = []
    h_img, w_img = img.shape[:2]
    page_margin = 0.1  # Avoid edges

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        area = w * h
        aspect_ratio = w / float(h)

        # General character constraints
        if area > 100 and h > 10:
            cropped_char = img[y:y+h, x:x+w]

            # Check for connected chars (wide boxes), e.g., "tt"
            if aspect_ratio > 1.5:
                # Split using vertical projection
                gray_crop = cv2.cvtColor(cropped_char, cv2.COLOR_BGR2GRAY)
                _, thresh_crop = cv2.threshold(gray_crop, 127, 255, cv2.THRESH_BINARY_INV)

                cnts_split, _ = cv2.findContours(thresh_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt_split in cnts_split:
                    x_split, y_split, w_split, h_split = cv2.boundingRect(cnt_split)
                    if w_split * h_split > 50:
                        cropped_characters.append((x + x_split, y + y_split, w_split, h_split,
                                                  cropped_char[y_split:y_split+h_split, x_split:x_split+w_split]))
            else:
                cropped_characters.append((x, y, w, h, cropped_char))

    # 4️⃣ Identify instances of top 3 characters
    final_crops = []
    for (x, y, w, h, cropped_char) in cropped_characters:
        cropped_char_filename = output_dir / "temp.png"
        cv2.imwrite(str(cropped_char_filename), cropped_char)

        results_char = reader.readtext(str(cropped_char_filename))
        detected_char_in_crop = ''.join([res[1] for res in results_char if res[2] > 0.3])

        if any(c in top_characters for c in detected_char_in_crop):
            final_crops.append((x, y, w, h, cropped_char, detected_char_in_crop))

    # ✅ Group final crops by character
    char_groups = defaultdict(list)
    for (x, y, w, h, cropped_char, detected_char_in_crop) in final_crops:
        for ch in detected_char_in_crop:
            if ch in top_characters:
                char_groups[ch].append((x, y, w, h, cropped_char))

    # ✅ Save all instances of top 3 characters
    count = 1
    for character, instances in char_groups.items():
        instances = sorted(instances, key=lambda item: item[2] * item[3], reverse=True)  # sort by area
        for (x, y, w, h, cropped_char) in instances:
            output_file = output_dir / f"char_{count}_{character}.png"
            cv2.imwrite(str(output_file), cropped_char)
            count += 1

    print(f"✅ Final cropped instances saved in {output_dir}")

# Usage
extract_characters('./image_9.png', './output_chars')
 