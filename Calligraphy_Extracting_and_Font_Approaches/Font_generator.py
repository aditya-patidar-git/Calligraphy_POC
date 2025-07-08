import os
import subprocess
from PIL import Image

INPUT_DIR = r"C:\Users\AdityaPatidar\Documents\Calligraphy_Approaches\output_images\image_2"
OUTPUT_DIR = r"C:\Users\AdityaPatidar\Documents\Calligraphy_Approaches\output_class_2_svg"
THRESHOLD = 128
POTRACE_PATH = r"C:\Users\AdityaPatidar\Documents\Calligraphy_Approaches\potrace-1.16.win64\potrace.exe"  # <-- IMPORTANT

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_image_to_pbm(image_path, pbm_path, threshold=128):
    img = Image.open(image_path).convert("L")  # grayscale
    img = img.point(lambda p: 255 if p > threshold else 0)  # apply threshold
    img.save(pbm_path, "PPM")  # PBM/PPM output

def trace_pbm_to_svg(pbm_path, svg_output_path):
    subprocess.run([POTRACE_PATH, pbm_path, "-s", "-o", svg_output_path], check=True)

def process_directory(input_dir, output_dir, threshold=128):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            base_name = os.path.splitext(filename)[0]
            input_image = os.path.join(input_dir, filename)

            pbm_file = os.path.join(output_dir, base_name + ".pbm")
            svg_file = os.path.join(output_dir, base_name + ".svg")

            convert_image_to_pbm(input_image, pbm_file, threshold)
            trace_pbm_to_svg(pbm_file, svg_file)

            os.remove(pbm_file)  # Optional clean‑up
            print(f"✅ Done: {filename} ➔ {base_name}.svg")

if __name__ == "__main__":
    process_directory(INPUT_DIR, OUTPUT_DIR, THRESHOLD)
