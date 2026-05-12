#!/usr/bin/env python3
"""
Test with exact same code as test_anima_inference.py but different configs.
"""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.util.import_util import script_imports
script_imports()

import torch
from PIL import Image
import numpy as np

from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.AnimaModelLoader import AnimaModelLoader
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType

TRANSFORMER_PATH = "/home/pc/Stažené/anima-preview3-base.safetensors"
TEXT_ENCODER_PATH = "/home/pc/Stažené/qwen_3_06b_base.safetensors"
VAE_PATH = "/home/pc/Stažené/qwen_image_vae.safetensors"
HF_COSMOS_REPO = "nvidia/Cosmos-Predict2-2B-Text2Image"

DEVICE = "cuda"
PROMPT = "masterpiece, best quality, score_7, safe, 1girl, anime style, high quality"
NEGATIVE_PROMPT = "worst quality, low quality"
HEIGHT = 1024
WIDTH = 1024
SEED = 42

OUT_DIR = "/home/pc/KimiProjects/ZImage-Loli/OneTrainer/test_images"
os.makedirs(OUT_DIR, exist_ok=True)

def hf_download(repo_id, filename, local_path):
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=os.path.dirname(local_path), local_dir_use_symlinks=False)
    except Exception as e:
        return None

def prepare_base_model_dir():
    tmpdir = tempfile.mkdtemp(prefix="anima_base_")
    os.makedirs(os.path.join(tmpdir, "scheduler"), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, "tokenizer"), exist_ok=True)
    hf_download(HF_COSMOS_REPO, "scheduler/scheduler_config.json", os.path.join(tmpdir, "scheduler_config.json"))
    hf_download("Qwen/Qwen3-0.6B", "tokenizer.json", os.path.join(tmpdir, "tokenizer", "tokenizer.json"))
    hf_download("Qwen/Qwen3-0.6B", "tokenizer_config.json", os.path.join(tmpdir, "tokenizer", "tokenizer_config.json"))
    return tmpdir

print("=" * 60)
print("Exact test_anima_inference code with different configs")
print("=" * 60)

base_model_dir = prepare_base_model_dir()

model = AnimaModel(ModelType.ANIMA)
model_weight_dtypes = ModelWeightDtypes.from_single_dtype(DataType.BFLOAT_16)
config = TrainConfig.default_values()
config.train_device = DEVICE
config.model_type = ModelType.ANIMA

loader = AnimaModelLoader()
model_names = ModelNames(
    base_model=base_model_dir,
    transformer_model=TRANSFORMER_PATH,
    vae_model=VAE_PATH,
    text_encoder_model=TEXT_ENCODER_PATH,
)

try:
    loader.load(
        model=model,
        model_type=ModelType.ANIMA,
        model_names=model_names,
        weight_dtypes=model_weight_dtypes,
        quantization=config.quantization,
    )
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    shutil.rmtree(base_model_dir, ignore_errors=True)
    sys.exit(1)

model.to(torch.device(DEVICE))
model.eval()

# Encode text
with torch.no_grad():
    text_encoder_output = model.encode_text(
        train_device=torch.device(DEVICE),
        text=[PROMPT],
        batch_size=1,
    )
    text_encoder_output_neg = model.encode_text(
        train_device=torch.device(DEVICE),
        text=[NEGATIVE_PROMPT],
        batch_size=1,
    )

generator = torch.Generator(device=DEVICE)
generator.manual_seed(SEED)

vae_scale_factor_temporal = 2 ** sum(model.vae.temperal_downsample)
vae_scale_factor_spatial = 2 ** len(model.vae.temperal_downsample)
num_latent_frames = (1 - 1) // vae_scale_factor_temporal + 1
latent_height = HEIGHT // vae_scale_factor_spatial
latent_width = WIDTH // vae_scale_factor_spatial
latent_shape = (1, model.transformer.config.in_channels, num_latent_frames, latent_height, latent_width)

latents = torch.randn(latent_shape, generator=generator, device=DEVICE, dtype=torch.float32)

from diffusers import FlowMatchEulerDiscreteScheduler

# Test 1: 35 steps, CFG=4.5 (same as test_anima_inference.py)
print("\nTest 1: 35 steps, CFG=4.5")
noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
noise_scheduler.set_timesteps(35, device=DEVICE)
timesteps = noise_scheduler.timesteps

padding_mask = latents.new_zeros(1, 1, HEIGHT, WIDTH, dtype=model.transformer.dtype)
transformer_dtype = model.transformer.dtype

latents_copy = latents.clone()
with torch.no_grad():
    for i, t in enumerate(timesteps):
        sigma = noise_scheduler.sigmas[i]
        timestep = sigma.expand(latents_copy.shape[0]).to(transformer_dtype)
        latent_model_input = latents_copy.to(transformer_dtype)

        velocity_cond = model.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_encoder_output.to(transformer_dtype),
            padding_mask=padding_mask,
            return_dict=False,
        )[0].float()

        velocity_uncond = model.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_encoder_output_neg.to(transformer_dtype),
            padding_mask=padding_mask,
            return_dict=False,
        )[0].float()

        velocity = velocity_uncond + 4.5 * (velocity_cond - velocity_uncond)
        latents_copy = noise_scheduler.step(velocity, t, latents_copy, return_dict=False)[0]

z_dim = model.vae.config.z_dim
latents_mean = torch.tensor(model.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(DEVICE, latents_copy.dtype)
latents_std = (1.0 / torch.tensor(model.vae.config.latents_std)).view(1, z_dim, 1, 1, 1).to(DEVICE, latents_copy.dtype)
latents_for_decode = latents_copy / latents_std + latents_mean
image = model.vae.decode(latents_for_decode.to(model.vae.dtype), return_dict=False)[0]
if image.ndim == 5:
    image = image.squeeze(2)
image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).float().numpy()
pil_image = Image.fromarray((image[0] * 255).astype("uint8"))
pil_image.save(os.path.join(OUT_DIR, "test_exact_35steps_cfg45.png"))
arr = np.array(pil_image)
print(f"  Saved: mean={arr.mean():.1f}, std={arr.std():.1f}")

# Test 2: 20 steps, CFG=1.0
print("\nTest 2: 20 steps, CFG=1.0")
noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
noise_scheduler.set_timesteps(20, device=DEVICE)
timesteps = noise_scheduler.timesteps

latents_copy = latents.clone()
with torch.no_grad():
    for i, t in enumerate(timesteps):
        sigma = noise_scheduler.sigmas[i]
        timestep = sigma.expand(latents_copy.shape[0]).to(transformer_dtype)
        latent_model_input = latents_copy.to(transformer_dtype)

        velocity = model.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_encoder_output.to(transformer_dtype),
            padding_mask=padding_mask,
            return_dict=False,
        )[0].float()

        latents_copy = noise_scheduler.step(velocity, t, latents_copy, return_dict=False)[0]

latents_for_decode = latents_copy / latents_std + latents_mean
image = model.vae.decode(latents_for_decode.to(model.vae.dtype), return_dict=False)[0]
if image.ndim == 5:
    image = image.squeeze(2)
image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).float().numpy()
pil_image = Image.fromarray((image[0] * 255).astype("uint8"))
pil_image.save(os.path.join(OUT_DIR, "test_exact_20steps_cfg1.png"))
arr = np.array(pil_image)
print(f"  Saved: mean={arr.mean():.1f}, std={arr.std():.1f}")

shutil.rmtree(base_model_dir, ignore_errors=True)
print("\nDone!")
