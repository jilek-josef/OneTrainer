#!/usr/bin/env python3
"""
Test diffusers Cosmos2TextToImagePipeline with local model files.
"""
import os
import sys

sys.path.insert(0, '/home/pc/KimiProjects/ZImage-Loli/OneTrainer')
from scripts.util.import_util import script_imports
script_imports()

import torch
from PIL import Image
import numpy as np

from diffusers import Cosmos2TextToImagePipeline, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2Tokenizer

# Model paths
TRANSFORMER_PATH = "/home/pc/Stažené/anima-preview3-base.safetensors"
TEXT_ENCODER_PATH = "/home/pc/Stažené/qwen_3_06b_base.safetensors"
VAE_PATH = "/home/pc/Stažené/qwen_image_vae.safetensors"
HF_COSMOS_REPO = "nvidia/Cosmos-Predict2-2B-Text2Image"

DEVICE = "cuda"
PROMPT = "masterpiece, best quality, score_7, safe, 1girl, anime style, high quality"
NEGATIVE_PROMPT = "worst quality, low quality"
HEIGHT = 1024
WIDTH = 1024
STEPS = 35
CFG = 4.5
SEED = 42

OUT_DIR = "/home/pc/KimiProjects/ZImage-Loli/OneTrainer/test_images"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("Diffusers Cosmos2TextToImagePipeline Test")
print("=" * 60)

# Try loading from the base repo first
print("\nAttempting to load pipeline from HuggingFace...")
try:
    pipe = Cosmos2TextToImagePipeline.from_pretrained(
        HF_COSMOS_REPO,
        torch_dtype=torch.bfloat16,
    )
    print("Loaded from HuggingFace successfully!")
except Exception as e:
    print(f"Failed to load from HuggingFace: {e}")
    print("\nTrying to load components individually...")
    
    # Load components individually
    from diffusers import AutoencoderKLWan, CosmosTransformer3DModel
    from transformers import Qwen3Model
    
    # Load tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    
    # Load text encoder
    text_encoder = Qwen3Model.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
    )
    
    # Load VAE
    vae = AutoencoderKLWan.from_pretrained(
        HF_COSMOS_REPO,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    
    # Load transformer
    transformer = CosmosTransformer3DModel.from_pretrained(
        HF_COSMOS_REPO,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    
    # Create scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        HF_COSMOS_REPO,
        subfolder="scheduler",
    )
    
    # Create pipeline
    pipe = Cosmos2TextToImagePipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    print("Loaded components individually!")

pipe = pipe.to(DEVICE)

# Generate image
print(f"\nGenerating image...")
print(f"  Prompt: {PROMPT}")
print(f"  Size: {WIDTH}x{HEIGHT}, Steps: {STEPS}, CFG: {CFG}, Seed: {SEED}")

generator = torch.Generator(device=DEVICE).manual_seed(SEED)

image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    height=HEIGHT,
    width=WIDTH,
    num_inference_steps=STEPS,
    guidance_scale=CFG,
    generator=generator,
).images[0]

image_path = os.path.join(OUT_DIR, "test_diffusers_pipeline_1024.png")
image.save(image_path)

arr = np.array(image)
print(f"\nSaved to: {image_path}")
print(f"Stats: shape={arr.shape} mean={arr.mean():.1f} std={arr.std():.1f}")
print("Done!")
