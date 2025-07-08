import os
import subprocess
import tempfile

def create_fontforge_script():
    """Create a FontForge script file to be executed via CLI"""
    
    # Paths
    svg_dir = r"C:\Users\AdityaPatidar\Documents\Calligraphy_Approaches\output_class_2_svg"
    
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
    
    # Create FontForge script content
    script_content = '''#!/usr/bin/env fontforge

# Create new font
New()

# Set font properties
SetFontNames("MyCalligraphyFont", "MyCalligraphyFont", "My Calligraphy Font")

# Define file mappings
'''
    
    # Add each glyph
    for filename, codepoint in files.items():
        svg_path = os.path.join(svg_dir, filename).replace('\\', '/')
        script_content += f'''
# Import {filename}
if (FileAccess("{svg_path}", 0) == 0)
    Select({codepoint:#x})
    Import("{svg_path}")
    SetLBearing(50)
    SetRBearing(50)
    Print("Imported {filename}")
else
    Print("File not found: {filename}")
endif
'''
    
    # Add font generation
    script_content += '''
# Generate font
Generate("MyCalligraphyFont.ttf")
Print("Font generated successfully!")
Quit()
'''
    
    return script_content

def run_fontforge_script():
    """Execute FontForge script via command line"""
    
    # Create the script
    script_content = create_fontforge_script()
    
    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pe', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        print("🚀 Running FontForge script...")
        print(f"Script location: {script_path}")
        
        # Execute FontForge with the script
        result = subprocess.run(
            ['fontforge', '-script', script_path],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        # Print results
        if result.stdout:
            print("📋 FontForge Output:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️ FontForge Errors/Warnings:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Font generation completed successfully!")
            if os.path.exists("MyCalligraphyFont.ttf"):
                print("✅ MyCalligraphyFont.ttf created!")
            else:
                print("⚠️ Font file not found after generation")
        else:
            print(f"❌ FontForge exited with code: {result.returncode}")
    
    except FileNotFoundError:
        print("❌ FontForge command not found. Make sure it's in your PATH.")
    except Exception as e:
        print(f"❌ Error running FontForge: {e}")
    
    finally:
        # Clean up script file
        try:
            os.unlink(script_path)
        except:
            pass

def create_standalone_script():
    """Create a standalone .pe script file for manual execution"""
    
    script_content = create_fontforge_script()
    
    with open("create_font.pe", 'w') as f:
        f.write(script_content)
    
    print("✅ Created create_font.pe script file")
    print("You can run it manually with: fontforge -script create_font.pe")

if __name__ == "__main__":
    print("FontForge CLI Font Generator")
    print("=" * 40)
    
    # Check if FontForge is available
    try:
        result = subprocess.run(['fontforge', '--version'], capture_output=True)
        if result.returncode == 0:
            print("✅ FontForge found in PATH")
            
            # Try to run the script
            run_fontforge_script()
            
            # Also create standalone script for manual use
            create_standalone_script()
            
        else:
            print("❌ FontForge not properly configured")
    except FileNotFoundError:
        print("❌ FontForge command not found in PATH")
        create_standalone_script()
        print("Please run: fontforge -script create_font.pe")