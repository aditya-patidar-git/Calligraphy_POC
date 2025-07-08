import cv2
import numpy as np
import easyocr
from pathlib import Path

reader = easyocr.Reader(['en'])

def extract_text(image_path):
    result = reader.readtext(image_path) 
    text = ''.join([r[1] for r in result])
    print(f"Full OCR Text: {text}")
    return text[:3], result  # return top 3 characters and all OCR results

def segment_with_craft(image_path):
    # For now, assume CRAFT already ran and boxes are in a .txt file.
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    txt_path = r"./CRAFT-pytorch/result/res_image_2.txt"
    # txt_path = Path(image_path).with_name('res_' + Path(image_path).stem + '.txt')
    
    boxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                coords = np.array(list(map(float, parts[:8]))).reshape(4, 2).astype(np.int32)
                x, y, ww, hh = cv2.boundingRect(coords)
                crop = image[y:y+hh, x:x+ww]
                boxes.append((coords, crop))
    return boxes

def recognize_each_crop(boxes):
    crops_data = []
    for poly, crop in boxes:
        if crop.shape[0] < 5 or crop.shape[1] < 5:
            continue  # skip too small
        result = reader.readtext(crop)
        if result:
            text, conf = result[0][1], result[0][2]
            crops_data.append({
                "char": text.strip()[0] if text.strip() else '',
                "conf": conf,
                "crop": crop
            })
    return crops_data

def find_best_instances(top3_chars, crops_data):
    best_crops = {}
    for char in top3_chars:
        candidates = [c for c in crops_data if c["char"].lower() == char.lower()]
        if candidates:
            best = max(candidates, key=lambda x: x["conf"])
            best_crops[char] = best["crop"]
    return best_crops

def save_characters(char_crops, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (char, img) in enumerate(char_crops.items(), start=1):
        path = output_dir / f"char_{i}_{char}.png"
        cv2.imwrite(str(path), img)
        print(f"Saved: {path}")

def process_image(image_path, output_dir="./output_chars"):
    top3_chars, _ = extract_text(image_path)
    boxes = segment_with_craft(image_path)
    crops_data = recognize_each_crop(boxes)
    best_crops = find_best_instances(top3_chars, crops_data)
    save_characters(best_crops, output_dir)

# Example usage:
process_image("./images/image_2.png", output_dir="./output_chars")
