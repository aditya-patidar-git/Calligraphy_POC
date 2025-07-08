import sys
import os
import subprocess

def find_fontforge_python():
    """Find and add FontForge Python module to path"""
    
    # Common FontForge installation paths on Windows
    possible_paths = [
        r"C:\Program Files (x86)\FontForgeBuilds\lib\python3.9\site-packages",
        r"C:\Program Files\FontForgeBuilds\lib\python3.9\site-packages",
        r"C:\Program Files (x86)\FontForge\lib\python3.9\site-packages",
        r"C:\Program Files\FontForge\lib\python3.9\site-packages",
        r"C:\msys64\mingw64\lib\python3.9\site-packages",
        r"C:\msys64\usr\lib\python3.9\site-packages",
    ]
    
    # Also check different Python versions
    for base_path in [r"C:\Program Files (x86)\FontForgeBuilds", r"C:\Program Files\FontForgeBuilds"]:
        if os.path.exists(base_path):
            lib_path = os.path.join(base_path, "lib")
            if os.path.exists(lib_path):
                for item in os.listdir(lib_path):
                    if item.startswith("python3."):
                        possible_paths.append(os.path.join(lib_path, item, "site-packages"))
    
    print("Searching for FontForge Python module...")
    for path in possible_paths:
        fontforge_path = os.path.join(path, "fontforge.pyd")  # Windows Python extension
        fontforge_so = os.path.join(path, "fontforge.so")     # Alternative
        
        if os.path.exists(fontforge_path) or os.path.exists(fontforge_so):
            print(f"✅ Found FontForge module at: {path}")
            if path not in sys.path:
                sys.path.insert(0, path)
            return True
        elif os.path.exists(path):
            print(f"🔍 Checking: {path}")
            # Check if fontforge module exists in this directory
            for file in os.listdir(path):
                if file.startswith("fontforge"):
                    print(f"✅ Found FontForge file: {os.path.join(path, file)}")
                    if path not in sys.path:
                        sys.path.insert(0, path)
                    return True
    
    print("❌ FontForge Python module not found in common locations")
    return False

def test_fontforge_import():
    """Test if FontForge can be imported"""
    try:
        import fontforge
        print("✅ FontForge imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ FontForge import failed: {e}")
        return False

def create_calligraphy_font():
    """Create the calligraphy font"""
    import fontforge
    import os
    
    # Create new font
    font = fontforge.font()
    
    # Set font properties
    font.fontname = "MyCalligraphyFont"
    font.fullname = "My Calligraphy Font"
    font.familyname = "MyCalligraphyFont"
    font.weight = "Regular"
    font.version = "1.0"
    
    # Paths
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
    successful_imports = 0
    for filename, codepoint in files.items():
        svg_path = os.path.join(svg_dir, filename)
        
        if not os.path.exists(svg_path):
            print(f"⚠️  File not found: {filename}")
            continue
            
        try:
            glyph = font.createChar(codepoint)
            glyph.importOutlines(svg_path)
            glyph.left_side_bearing = 50
            glyph.right_side_bearing = 50
            
            print(f"✅ Imported: {filename}")
            successful_imports += 1
            
        except Exception as e:
            print(f"❌ Error importing {filename}: {e}")

    # Generate font
    if successful_imports > 0:
        font.generate("MyCalligraphyFont.ttf")
        print(f"✅ Font generated! ({successful_imports} glyphs)")
    else:
        print("❌ No glyphs imported, font not generated")

if __name__ == "__main__":
    # Try to find and add FontForge to path
    if find_fontforge_python():
        if test_fontforge_import():
            create_calligraphy_font()
        else:
            print("FontForge found but still can't import. Try the alternative solution.")
    else:
        print("FontForge Python module not found. Use the fontTools alternative instead.")