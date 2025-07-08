import cv2
import easyocr
import numpy as np
import os
import json

def load_polys_from_txt(txt_file):
    """Load polygons from a .txt file. Returns a list of numpy arrays."""
    polys = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            coords = line.strip().split(',')
            if len(coords) >= 8:
                points = list(map(int, coords))
                poly = np.array([[points[i], points[i + 1]] for i in range(0, len(points), 2)])
                polys.append(poly)
    return polys

def extract_text_from_polys(image, polys, lang='en'):
    """Extract text from cropped areas defined by polygons using EasyOCR."""
    reader = easyocr.Reader([lang])
    results = []
    for poly in polys:
        poly = np.array(poly).astype(np.int32)

        x_min, y_min = poly[:, 0].min(), poly[:, 1].min()
        x_max, y_max = poly[:, 0].max(), poly[:, 1].max()

        cropped_img = image[y_min:y_max, x_min:x_max]
        ocr_result = reader.readtext(cropped_img)

        text_in_box = ' '.join([res[1] for res in ocr_result])
        results.append({'box': poly.tolist(), 'text': text_in_box})
    return results

def process_image_and_txt(image_path, txt_file, lang='en'):
    """Main process for one image + its coordinate text file."""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image could not be read from path: {image_path}")

    # Load polygons
    polys = load_polys_from_txt(txt_file)

    # Extract text
    results = extract_text_from_polys(image, polys, lang)

    return results

if __name__ == '__main__':
    # Paths
    image_dir = './images'                      # Directory with images
    coords_dir = './result/text_files'           # Directory with .txt files
    output_dir = './ocr_output'                 # Directory for results
    lang = 'en'                                 # Language for OCR

    os.makedirs(output_dir, exist_ok=True)

    # Loop over text files
    for txt_file in os.listdir(coords_dir):
        if txt_file.endswith('.txt'):
            txt_file_name = os.path.splitext(txt_file)[0]  # e.g., "res_image_1"
            if txt_file_name.startswith("res_"):
                image_name = txt_file_name[len("res_"):]    # Removes "res_"
            else:
                image_name = txt_file_name
            print(f"Processing {image_name}")

            # Find image path
            image_path = None
            for ext in ['.jpg', '.png']:
                candidate = os.path.join(image_dir, image_name + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            if image_path is None:
                print(f"Warning: No image found for {txt_file}")
                continue

            # Process and save results
            try:
                results = process_image_and_txt(image_path, os.path.join(coords_dir, txt_file), lang)
                output_file = os.path.join(output_dir, image_name + '_results.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                print(f"✅ Done for {image_name}. Results saved to {output_file}")

            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")

