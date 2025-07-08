# %%writefile calligraphy_generator.py
#!/usr/bin/env python3
"""
Calligraphy Generation Script using trained LoRA model
"""

import torch
import pandas as pd
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
import argparse
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2
import json
from typing import List, Dict, Tuple
import re

class CalligraphyGenerator:
    def __init__(self, model_path: str, lora_path: str, device: str = "auto"):
        """
        Initialize the calligraphy generator
        
        Args:
            model_path: Path to base Stable Diffusion model
            lora_path: Path to trained LoRA weights
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.device = self._setup_device(device)
        self.model_path = model_path
        self.lora_path = lora_path
        
        print(f"Loading models on {self.device}...")
        self._load_models()
        
    def _setup_device(self, device: str) -> str:
        """Setup and return the appropriate device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_models(self):
        """Load the base model and LoRA weights"""
        # Load base pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Load LoRA weights
        if Path(self.lora_path).exists():
            print(f"Loading LoRA weights from: {self.lora_path}")
            self.pipe.unet = PeftModel.from_pretrained(
                self.pipe.unet, 
                self.lora_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
        else:
            raise FileNotFoundError(f"LoRA weights not found at: {self.lora_path}")
        
        # Optimize pipeline
        self.pipe = self.pipe.to(self.device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Enable memory efficient attention if available
        if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        
        # Enable attention slicing for memory efficiency
        self.pipe.enable_attention_slicing()
        
        print("Models loaded successfully!")
    
    def _create_calligraphy_prompt(self, text: str, style_emphasis: str = "medium") -> str:
        """
        Create an optimized prompt for calligraphy generation
        
        Args:
            text: The text to be written in calligraphy
            style_emphasis: Level of style emphasis ('light', 'medium', 'strong')
        """
        # Base prompt templates matching your training data
        base_templates = [
            f"brush lettering text '{text}', elegant calligraphy style",
            f"handwritten text '{text}' in flowing brush calligraphy",
            f"artistic brush lettering '{text}', expressive calligraphy",
            f"elegant handwritten text '{text}' with brush pen strokes",
            f"flowing calligraphy text '{text}', artistic brush style"
        ]
        
        # Style modifiers based on the artist's distinctive characteristics
        style_modifiers = {
            "light": [
                ", clean black ink on white paper",
                ", elegant and flowing strokes",
                ", artistic calligraphy style"
            ],
            "medium": [
                ", dramatic curves and sweeping loops",
                ", expressive brush strokes, black ink on white paper",
                ", flowing calligraphy with confident irregularities",
                ", artistic hand lettering, dynamic curves"
            ],
            "strong": [
                ", bold expressive calligraphy, dramatic sweeping loops",
                ", wild flowing brush strokes, confident irregularities",
                ", artistic calligraphy with emotional fluidity and motion",
                ", dynamic brush lettering, illustrative presence, black ink on white"
            ]
        }
        
        # Quality enhancers
        quality_terms = [
            ", high contrast, sharp details",
            ", professional calligraphy, clear strokes",
            ", high resolution, crisp lines",
            ", detailed brush work, artistic quality"
        ]
        
        # Negative prompt elements
        negative_elements = [
            "blurry", "low quality", "pixelated", "distorted", "multiple copies",
            "duplicate text", "cropped", "worst quality", "jpeg artifacts",
            "watermark", "signature", "username", "text", "logo", "copyright"
        ]
        
        # Construct the prompt
        import random
        base = random.choice(base_templates)
        style = random.choice(style_modifiers.get(style_emphasis, style_modifiers["medium"]))
        quality = random.choice(quality_terms)
        
        prompt = base + style + quality
        negative_prompt = ", ".join(negative_elements)
        
        return prompt, negative_prompt
    
    def _post_process_image(self, image: Image.Image, target_dpi: int = 1200) -> Image.Image:
        """
        Post-process the generated image for better quality
        
        Args:
            image: Generated PIL Image
            target_dpi: Target DPI for output
        """
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        
        # Convert to grayscale if needed, then back to RGB for consistency
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply additional contrast enhancement using OpenCV
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Convert back to PIL
        image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # Set DPI
        image.info['dpi'] = (target_dpi, target_dpi)
        
        return image
    
    def generate_single_text(self, 
                           text: str, 
                           style_emphasis: str = "medium",
                           num_inference_steps: int = 50,
                           guidance_scale: float = 7.5,
                           height: int = 768,
                           width: int = 1024,
                           seed: int = None) -> Image.Image:
        """
        Generate calligraphy for a single text input
        
        Args:
            text: Text to convert to calligraphy
            style_emphasis: Style emphasis level
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            height: Output height
            width: Output width
            seed: Random seed for reproducibility
        """
        # Create prompt
        prompt, negative_prompt = self._create_calligraphy_prompt(text, style_emphasis)
        
        print(f"Generating calligraphy for: '{text}'")
        print(f"Prompt: {prompt}")
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Generate image
        with torch.autocast("cuda" if self.device == "cuda" else "cpu"):
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1
            )
        
        image = result.images[0]
        
        # Post-process
        image = self._post_process_image(image)
        
        return image
    
    def generate_from_csv(self, 
                         csv_path: str, 
                         output_dir: str,
                         text_column: str = "text",
                         name_column: str = "name",
                         **generation_kwargs) -> List[Dict]:
        """
        Generate calligraphy from CSV file
        
        Args:
            csv_path: Path to CSV file
            output_dir: Directory to save generated images
            text_column: Column name containing text to generate
            name_column: Column name for output filename (optional)
            **generation_kwargs: Additional arguments for generation
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for idx, row in df.iterrows():
            text = str(row[text_column])
            
            # Generate filename
            if name_column in df.columns and pd.notna(row[name_column]):
                filename = f"{idx:03d}_{self._clean_filename(str(row[name_column]))}.png"
            else:
                filename = f"{idx:03d}_{self._clean_filename(text)}.png"
            
            filepath = output_path / filename
            
            try:
                # Generate image
                image = self.generate_single_text(text, **generation_kwargs)
                
                # Save image
                image.save(filepath, "PNG", dpi=(1200, 1200))
                
                result = {
                    "index": idx,
                    "text": text,
                    "filename": filename,
                    "filepath": str(filepath),
                    "status": "success"
                }
                
                print(f"Generated: {filename}")
                
            except Exception as e:
                result = {
                    "index": idx,
                    "text": text,
                    "filename": filename,
                    "filepath": str(filepath),
                    "status": "error",
                    "error": str(e)
                }
                print(f"Error generating {filename}: {e}")
            
            results.append(result)
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save results summary
        results_path = output_path / "generation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nGeneration complete! Results saved to: {output_path}")
        print(f"Summary: {sum(1 for r in results if r['status'] == 'success')}/{len(results)} successful")
        
        return results
    
    def _clean_filename(self, text: str) -> str:
        """Clean text for use as filename"""
        # Remove or replace invalid filename characters
        text = re.sub(r'[<>:"/\\|?*]', '_', text)
        text = re.sub(r'\s+', '_', text)
        return text[:50]  # Limit length
    
    def generate_batch_samples(self, 
                             texts: List[str], 
                             output_dir: str,
                             variations_per_text: int = 3,
                             **generation_kwargs) -> List[Dict]:
        """
        Generate multiple variations for each text
        
        Args:
            texts: List of texts to generate
            output_dir: Output directory
            variations_per_text: Number of variations per text
            **generation_kwargs: Generation parameters
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for text_idx, text in enumerate(texts):
            for var_idx in range(variations_per_text):
                filename = f"text_{text_idx:03d}_var_{var_idx:02d}_{self._clean_filename(text)}.png"
                filepath = output_path / filename
                
                try:
                    # Use different seeds for variations
                    seed = text_idx * 1000 + var_idx if 'seed' not in generation_kwargs else None
                    
                    image = self.generate_single_text(
                        text, 
                        seed=seed,
                        **generation_kwargs
                    )
                    
                    image.save(filepath, "PNG", dpi=(1200, 1200))
                    
                    result = {
                        "text_index": text_idx,
                        "variation": var_idx,
                        "text": text,
                        "filename": filename,
                        "filepath": str(filepath),
                        "status": "success"
                    }
                    
                    print(f"Generated: {filename}")
                    
                except Exception as e:
                    result = {
                        "text_index": text_idx,
                        "variation": var_idx,
                        "text": text,
                        "filename": filename,
                        "filepath": str(filepath),
                        "status": "error",
                        "error": str(e)
                    }
                    print(f"Error generating {filename}: {e}")
                
                results.append(result)
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Generate calligraphy using trained LoRA model")
    parser.add_argument("--lora_path", required=True, help="Path to trained LoRA weights")
    parser.add_argument("--model_path", default="runwayml/stable-diffusion-v1-5", help="Base model path")
    parser.add_argument("--output_dir", default="./generated_calligraphy", help="Output directory")
    
    # Input options
    parser.add_argument("--text", type=str, help="Single text to generate")
    parser.add_argument("--csv_path", type=str, help="Path to CSV file with texts")
    parser.add_argument("--text_column", default="text", help="Column name for text in CSV")
    parser.add_argument("--name_column", default="name", help="Column name for names in CSV")
    
    # Generation parameters
    parser.add_argument("--style_emphasis", default="medium", choices=["light", "medium", "strong"])
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--variations", type=int, default=1, help="Number of variations per text")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CalligraphyGenerator(
        model_path=args.model_path,
        lora_path=args.lora_path
    )
    
    # Generation parameters
    gen_kwargs = {
        "style_emphasis": args.style_emphasis,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed": args.seed
    }
    
    if args.text:
        # Generate single text
        print(f"Generating calligraphy for: '{args.text}'")
        image = generator.generate_single_text(args.text, **gen_kwargs)
        
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"{generator._clean_filename(args.text)}.png"
        filepath = output_path / filename
        
        image.save(filepath, "PNG", dpi=(1200, 1200))
        print(f"Saved: {filepath}")
        
    elif args.csv_path:
        # Generate from CSV
        print(f"Generating calligraphy from CSV: {args.csv_path}")
        results = generator.generate_from_csv(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            text_column=args.text_column,
            name_column=args.name_column,
            **gen_kwargs
        )
        
        print(f"Generated {len(results)} images")
        
    else:
        # Demo mode - generate some example texts
        demo_texts = [
            "Hello Beautiful",
            "Wedding Invitation", 
            "Thank You",
            "Save the Date",
            "Congratulations"
        ]
        
        print("Demo mode - generating sample texts...")
        results = generator.generate_batch_samples(
            texts=demo_texts,
            output_dir=args.output_dir,
            variations_per_text=args.variations,
            **gen_kwargs
        )
        
        print(f"Generated {len(results)} sample images")


if __name__ == "__main__":
    main()