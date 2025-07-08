# %%writefile /kaggle/working/setup.py
# Cell 1: Setup and Imports
# Run this first to install dependencies and import required libraries

# Install required packages
# !pip install diffusers==0.21.4
# !pip install transformers==4.35.0
# !pip install accelerate==0.24.1
# !pip install xformers==0.0.22
# !pip install opencv-python==4.8.1.78
# !pip install pillow==10.0.1
# !pip install datasets==2.14.6
# !pip install peft==0.6.2
# !pip install bitsandbytes==0.41.2.post2

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import random
import re
import shutil
from datetime import datetime

# Deep learning imports
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from diffusers import (
    StableDiffusionPipeline, 
    UNet2DConditionModel, 
    DDPMScheduler,
    AutoencoderKL
)
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)