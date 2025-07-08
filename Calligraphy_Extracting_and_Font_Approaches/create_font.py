import fontforge
import os

# Paths
font = fontforge.font()
svg_dir = os.path.normpath(r"C:\Users\AdityaPatidar\Documents\Calligraphy_Approaches\output_class_2_svg")

files = {
    # Lowercase
    "a.svg": 0x61,
    "b.svg": 0x62,
    "d.svg": 0x64,
    "e.svg": 0x65,
    "k.svg": 0x6B,
    "s.svg": 0x73,
    "u.svg": 0x75,
    "v.svg": 0x76,
    "w.svg": 0x77,

    # Uppercase
    "capital_A.svg": 0x41,
    "capital_E.svg": 0x45,
    "F.svg": 0x46,
    "I.svg": 0x49,
    "L.svg": 0x4C,
    "M.svg": 0x4D,
    "N.svg": 0x4E,

    # Special character
    "&.svg": 0x26,
}

# Import SVG files
for filename, codepoint in files.items():
    glyph = font.createChar(codepoint)
    glyph.importOutlines(os.path.join(svg_dir, filename))
    glyph.left_side_bearing = 50
    glyph.right_side_bearing = 50

# Export font
font.generate("MyCalligraphyFont.ttf")
print("✅ Done! Font generated as MyCalligraphyFont.ttf.")
