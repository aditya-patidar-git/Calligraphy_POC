from main.py import train_calligraphy_model

result = train_calligraphy_model(
    character_dir="/kaggle/input/character-simple-images",   # replace with your actual path
    word_dir="kaggle/input/word-simple-images",             # replace with your actual path
    style_name="simple",
    complex_style=False,                         # set to True for complex calligraphy
    output_dir="/kaggle/working/output-data"            # optional
)