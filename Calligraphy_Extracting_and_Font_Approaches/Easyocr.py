import easyocr
reader = easyocr.Reader(['en'])
results = reader.readtext('image_2.png')
for bbox, text, conf in results:
    print(f"Text: {text} (confidence: {conf:.2f})")
