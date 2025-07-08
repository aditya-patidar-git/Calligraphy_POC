import cv2
import os
import argparse
import numpy as np

def extract_shapes_with_ellipses(image_path, output_dir="extracted_shapes_ellipse",
                                   threshold_value=127, min_area=100, scale_factor=1.5):
    """Extract shapes using ellipses for bounding and save them."""

    # 1️⃣ Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Could not read the image from {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2️⃣ Threshold
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # 3️⃣ Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    os.makedirs(output_dir, exist_ok=True)

    saved_count = 0
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area or len(cnt) < 5:
            continue  # skip tiny contours or those too few points for an ellipse

        # 4️⃣ Get the fitted ellipse
        (center_x, center_y), (width, height), angle = cv2.fitEllipse(cnt)

        # 5️⃣ Scale the ellipse
        center = (center_x, center_y)
        new_axes = (width * scale_factor, height * scale_factor)
        scaled_ellipse = (center, new_axes, angle)

        # 6️⃣ Create a mask for the elliptical area
        mask = np.zeros_like(gray)
        cv2.ellipse(mask, scaled_ellipse, 255, -1)

        # 7️⃣ Extract the elliptical area
        result = cv2.bitwise_and(img, img, mask=mask)

        # 8️⃣ Crop the result
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = result[y:y + h, x:x + w]

        # 9️⃣ Save the cropped elliptical shape
        save_path = os.path.join(output_dir, f"shape_ellipse_{i}.png")
        cv2.imwrite(save_path, cropped)

        # 10️⃣ Draw the scaled ellipse for visual annotation
        cv2.ellipse(img, scaled_ellipse, (0, 255, 0), 2)

        saved_count += 1

    # 🔟 Save the annotated image
    annotated_image = os.path.join(output_dir, "annotated.png")
    cv2.imwrite(annotated_image, img)

    print(f"✅ Done! Extracted and saved {saved_count} shapes to '{output_dir}/'.")

if __name__ == "__main__":
    extract_shapes_with_ellipses(
        image_path="image_2.png",
        output_dir="output_images",
        threshold_value=150,
        min_area=200,
        scale_factor=1.1
    )
