# Calligraphy Generation Pipeline using Stable Diffusion + ControlNet
# Designed for Google Colab with GPU runtime

import os
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from transformers import pipeline
import json
from pathlib import Path
import shutil

# =============================================================================
# 1. SETUP AND INSTALLATION (Run in Colab)
# =============================================================================

def setup_environment():
    """Install required packages for Colab"""
    commands = [
        "pip install diffusers transformers accelerate",
        "pip install controlnet-aux",
        "pip install xformers",
        "pip install opencv-python",
        "pip install pandas openpyxl",
        "pip install matplotlib seaborn",
        "pip install Pillow==9.5.0",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    ]
    
    for cmd in commands:
        os.system(cmd)
    
    print("Environment setup complete!")

# =============================================================================
# 2. DATA PREPARATION AND PREPROCESSING
# =============================================================================

class CalligraphyDataProcessor:
    def __init__(self, stroke_images_path="stroke_images/"):
        self.stroke_images_path = stroke_images_path
        self.processed_data = []
        
    def load_stroke_images(self):
        """Load and process individual stroke images"""
        stroke_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            stroke_files.extend(Path(self.stroke_images_path).glob(ext))
        
        strokes = []
        for file_path in stroke_files:
            img = Image.open(file_path).convert('RGB')
            # Resize to standard size for training
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            strokes.append({
                'image': img,
                'filename': file_path.name,
                'character': self.extract_character_from_filename(file_path.name)
            })
        
        return strokes
    
    def extract_character_from_filename(self, filename):
        """Extract character from filename - customize based on your naming convention"""
        # Assuming filenames like 'stroke_A.jpg', 'stroke_B.jpg', etc.
        base_name = filename.split('.')[0]
        if '_' in base_name:
            return base_name.split('_')[-1]
        return base_name[-1]  # fallback to last character
    
    def create_edge_maps(self, images):
        """Create edge maps for ControlNet conditioning"""
        edge_maps = []
        for img in images:
            # Convert PIL to numpy
            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Convert back to PIL
            edge_img = Image.fromarray(edges).convert('RGB')
            edge_maps.append(edge_img)
        
        return edge_maps

# =============================================================================
# 3. FINE-TUNING SETUP
# =============================================================================

class CalligraphyTrainer:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.controlnet = None
        self.pipeline = None
        
    def setup_controlnet(self):
        """Setup ControlNet for edge-based conditioning"""
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_name,
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        if self.device == "cuda":
            self.pipeline = self.pipeline.to("cuda")
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.enable_xformers_memory_efficient_attention()
        
        # Use DDIM scheduler for better quality
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        
        print("ControlNet pipeline setup complete!")
    
    def create_training_prompts(self, characters):
        """Create descriptive prompts for each character"""
        base_prompt = """elegant calligraphy handwriting, expressive line drawing, 
        emotional wildness and fluidity, spontaneous and alive lettering, 
        sweeping loops, dramatic curves, confident irregularities, 
        motion and mood, illustrative presence, bold yet graceful, 
        visual rhythm, dynamic imaginative aesthetic, high contrast black ink on white paper"""
        
        prompts = []
        for char in characters:
            char_prompt = f"letter '{char}', {base_prompt}"
            prompts.append(char_prompt)
        
        return prompts

# =============================================================================
# 4. TEXT-TO-CALLIGRAPHY GENERATION
# =============================================================================

class CalligraphyGenerator:
    def __init__(self, trained_pipeline):
        self.pipeline = trained_pipeline
        self.base_prompt = """elegant calligraphy handwriting, expressive line drawing, 
        emotional wildness and fluidity, spontaneous and alive lettering, 
        sweeping loops, dramatic curves, confident irregularities, 
        motion and mood, illustrative presence, bold yet graceful, 
        visual rhythm, dynamic imaginative aesthetic, high contrast black ink on white paper"""
    
    def create_text_layout(self, text, image_size=(1024, 1024)):
        """Create a basic text layout for ControlNet conditioning"""
        img = Image.new('RGB', image_size, 'white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a script-like font, fallback to default
        try:
            font_size = max(60, min(120, image_size[0] // len(text)))
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (image_size[0] - text_width) // 2
        y = (image_size[1] - text_height) // 2
        
        draw.text((x, y), text, fill='black', font=font)
        
        return img
    
    def text_to_edge_map(self, text, image_size=(1024, 1024)):
        """Convert text to edge map for ControlNet"""
        text_img = self.create_text_layout(text, image_size)
        
        # Convert to edge map
        img_np = np.array(text_img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to make them more prominent
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        edge_img = Image.fromarray(edges).convert('RGB')
        return edge_img
    
    def generate_calligraphy(self, text, num_inference_steps=20, guidance_scale=7.5, controlnet_conditioning_scale=1.0):
        """Generate calligraphy for given text"""
        # Create control image
        control_image = self.text_to_edge_map(text)
        
        # Create prompt
        prompt = f"calligraphy writing '{text}', {self.base_prompt}"
        negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, extra limbs, poorly drawn"
        
        # Generate image
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                width=1024,
                height=1024
            )
        
        return result.images[0], control_image

# =============================================================================
# 5. BATCH PROCESSING FROM SPREADSHEET
# =============================================================================

class BatchProcessor:
    def __init__(self, generator, output_dir="generated_calligraphy/"):
        self.generator = generator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_spreadsheet(self, file_path):
        """Load names and addresses from spreadsheet"""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        return df
    
    def process_row(self, row_data, row_index):
        """Process a single row from the spreadsheet"""
        # Combine name and address
        name = str(row_data.get('name', ''))
        address = str(row_data.get('address', ''))
        
        # Create full text
        if name and address:
            full_text = f"{name}\n{address}"
        elif name:
            full_text = name
        elif address:
            full_text = address
        else:
            full_text = f"Row {row_index}"
        
        # Generate calligraphy
        generated_img, control_img = self.generator.generate_calligraphy(full_text)
        
        # Save high-resolution image
        output_path = self.output_dir / f"calligraphy_{row_index:04d}.png"
        
        # Upscale to high DPI (1200 DPI equivalent)
        high_res_size = (4800, 4800)  # 4 inches at 1200 DPI
        generated_img = generated_img.resize(high_res_size, Image.Resampling.LANCZOS)
        
        generated_img.save(output_path, dpi=(1200, 1200))
        
        # Also save control image for reference
        control_path = self.output_dir / f"control_{row_index:04d}.png"
        control_img.save(control_path)
        
        return output_path
    
    def process_spreadsheet(self, file_path):
        """Process entire spreadsheet"""
        df = self.load_spreadsheet(file_path)
        results = []
        
        for index, row in df.iterrows():
            try:
                output_path = self.process_row(row, index)
                results.append({
                    'row_index': index,
                    'status': 'success',
                    'output_path': str(output_path),
                    'original_data': dict(row)
                })
                print(f"Processed row {index}: {output_path}")
            except Exception as e:
                results.append({
                    'row_index': index,
                    'status': 'error',
                    'error': str(e),
                    'original_data': dict(row)
                })
                print(f"Error processing row {index}: {e}")
        
        # Save results log
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / "processing_results.csv", index=False)
        
        return results

# =============================================================================
# 6. MAIN WORKFLOW
# =============================================================================

def main_workflow():
    """Complete workflow for calligraphy generation"""
    
    print("=== Calligraphy Generation Pipeline ===")
    
    # Step 1: Setup environment (run once in Colab)
    # setup_environment()
    
    # Step 2: Process stroke images
    print("\n1. Processing stroke images...")
    processor = CalligraphyDataProcessor()
    strokes = processor.load_stroke_images()
    print(f"Loaded {len(strokes)} stroke images")
    
    # Step 3: Setup and prepare model
    print("\n2. Setting up ControlNet...")
    trainer = CalligraphyTrainer()
    trainer.setup_controlnet()
    
    # Step 4: Create generator
    print("\n3. Creating calligraphy generator...")
    generator = CalligraphyGenerator(trainer.pipeline)
    
    # Step 5: Test single generation
    print("\n4. Testing single generation...")
    test_text = "John Smith"
    test_img, test_control = generator.generate_calligraphy(test_text)
    
    # Display test result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(test_control)
    ax1.set_title("Control Image")
    ax1.axis('off')
    ax2.imshow(test_img)
    ax2.set_title("Generated Calligraphy")
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Step 6: Process spreadsheet (when ready)
    print("\n5. Ready for batch processing!")
    print("Upload your spreadsheet and run:")
    print("batch_processor = BatchProcessor(generator)")
    print("results = batch_processor.process_spreadsheet('your_file.csv')")
    
    return generator

# =============================================================================
# 7. COLAB-SPECIFIC HELPER FUNCTIONS
# =============================================================================

def upload_stroke_images():
    """Helper function to upload stroke images in Colab"""
    from google.colab import files
    
    print("Please upload your stroke images...")
    uploaded = files.upload()
    
    # Create directory and save files
    os.makedirs("stroke_images", exist_ok=True)
    for filename, data in uploaded.items():
        with open(f"stroke_images/{filename}", "wb") as f:
            f.write(data)
    
    print(f"Uploaded {len(uploaded)} stroke images to stroke_images/")

def upload_spreadsheet():
    """Helper function to upload spreadsheet in Colab"""
    from google.colab import files
    
    print("Please upload your spreadsheet (CSV or Excel)...")
    uploaded = files.upload()
    
    filename = list(uploaded.keys())[0]
    print(f"Uploaded spreadsheet: {filename}")
    return filename

# =============================================================================
# 8. EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # For Colab usage, uncomment these lines:
    # upload_stroke_images()
    # generator = main_workflow()
    # 
    # spreadsheet_file = upload_spreadsheet()
    # batch_processor = BatchProcessor(generator)
    # results = batch_processor.process_spreadsheet(spreadsheet_file)
    
    print("Calligraphy generation pipeline ready!")
    print("\nTo use in Colab:")
    print("1. Run setup_environment()")
    print("2. Upload stroke images with upload_stroke_images()")
    print("3. Run main_workflow()")
    print("4. Upload spreadsheet and process with BatchProcessor")