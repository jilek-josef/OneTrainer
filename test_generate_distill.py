#!/usr/bin/env python3
"""
Generate test images using DistillLoRA sampling code.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.util.import_util import script_imports
script_imports()

import torch
import torch.nn.functional as F
from PIL import Image
import copy

from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.AnimaModelLoader import AnimaModelLoader
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType

TRANSFORMER_PATH = "/home/pc/Stažené/anima-preview3-base.safetensors"
TEXT_ENCODER_PATH = "/home/pc/Stažené/qwen_3_06b_base.safetensors"
VAE_PATH = "/home/pc/Stažené/qwen_image_vae.safetensors"
HF_COSMOS_REPO = "nvidia/Cosmos-Predict2-2B-Text2Image"

DEVICE = "cuda"
PROMPT = "masterpiece, best quality, score_7, safe, 1girl, anime style, high quality"
HEIGHT = 512
WIDTH = 512
SEED = 42

# Output paths
OUT_DIR = "/home/pc/KimiProjects/ZImage-Loli/OneTrainer/test_images"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("DistillLoRA Sampling Test")
print("=" * 60)

# Prepare base model dir
base_model_dir = tempfile.mkdtemp(prefix="anima_base_")
os.makedirs(os.path.join(base_model_dir, "scheduler"), exist_ok=True)
os.makedirs(os.path.join(base_model_dir, "tokenizer"), exist_ok=True)

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id=HF_COSMOS_REPO, filename="scheduler/scheduler_config.json", local_dir=base_model_dir)
hf_hub_download(repo_id="Qwen/Qwen3-0.6B", filename="tokenizer.json", local_dir=os.path.join(base_model_dir, "tokenizer"))
hf_hub_download(repo_id="Qwen/Qwen3-0.6B", filename="tokenizer_config.json", local_dir=os.path.join(base_model_dir, "tokenizer"))

# Create model
print("\n[1/5] Creating model...")
model = AnimaModel(model_type=ModelType.ANIMA)

# Load weights
print("[2/5] Loading weights...")
model_names = ModelNames(
    base_model=base_model_dir,
    transformer_model=TRANSFORMER_PATH,
    text_encoder_model=TEXT_ENCODER_PATH,
    vae_model=VAE_PATH,
)
weight_dtypes = ModelWeightDtypes(
    train_dtype=DataType.BFLOAT_16,
    fallback_train_dtype=DataType.BFLOAT_16,
    unet=DataType.NONE,
    prior=DataType.NONE,
    transformer=DataType.BFLOAT_16,
    text_encoder=DataType.BFLOAT_16,
    text_encoder_2=DataType.NONE,
    text_encoder_3=DataType.NONE,
    text_encoder_4=DataType.NONE,
    vae=DataType.BFLOAT_16,
    effnet_encoder=DataType.NONE,
    decoder=DataType.NONE,
    decoder_text_encoder=DataType.NONE,
    decoder_vqgan=DataType.NONE,
    lora=DataType.NONE,
    embedding=DataType.NONE,
)

loader = AnimaModelLoader()
from modules.util.quantization_util import QuantizationConfig
loader.load(
    model=model,
    model_type=ModelType.ANIMA,
    model_names=model_names,
    weight_dtypes=weight_dtypes,
    quantization=QuantizationConfig.default_values(),
)

print("[3/5] Moving to device...")
model.to(torch.device(DEVICE))
model.eval()

# Encode text
print("[4/5] Encoding text...")
text_encoder_output = model.encode_text(
    train_device=torch.device(DEVICE),
    batch_size=1,
    rand=__import__('random').Random(SEED),
    text=[PROMPT],
)
print(f"  Text embed: mean={text_encoder_output.mean().item():.4f}, std={text_encoder_output.std().item():.4f}")

# Generate noise
generator = torch.Generator(device=DEVICE)
generator.manual_seed(SEED)

vae_scale_factor_temporal = 2 ** sum(model.vae.temperal_downsample)
vae_scale_factor_spatial = 2 ** len(model.vae.temperal_downsample)
num_latent_frames = (1 - 1) // vae_scale_factor_temporal + 1
latent_height = HEIGHT // vae_scale_factor_spatial
latent_width = WIDTH // vae_scale_factor_spatial
latent_shape = (1, model.transformer.config.in_channels, num_latent_frames, latent_height, latent_width)

noise = torch.randn(latent_shape, generator=generator, device=DEVICE, dtype=torch.float32)
print(f"  Noise: mean={noise.mean().item():.4f}, std={noise.std().item():.4f}")

# ---- TEST 1: Student sampling (20 steps, CFG=1.0, enable_grad) ----
print("\n[5/5] Test 1: Student sampling (20 steps, CFG=1.0)...")

student_scheduler = copy.deepcopy(model.noise_scheduler)
student_scheduler.set_timesteps(20, device=DEVICE)
student_timesteps = student_scheduler.timesteps
student_sigmas = student_scheduler.sigmas.to(DEVICE)

student_latent = noise.clone()
b, c, t, h, w = student_latent.shape
padding_mask = student_latent.new_zeros(1, 1, HEIGHT, WIDTH, dtype=model.transformer.dtype)
text_encoder_output_dtype = text_encoder_output.to(dtype=model.transformer.dtype)

with torch.no_grad():
    for i, t_step in enumerate(student_timesteps):
        sigma = student_sigmas[i]
        timestep_t = sigma.expand(student_latent.shape[0]).to(model.transformer.dtype)
        latent_input = student_latent.to(dtype=model.transformer.dtype)
        
        velocity = model.transformer(
            hidden_states=latent_input,
            timestep=timestep_t,
            encoder_hidden_states=text_encoder_output_dtype,
            padding_mask=padding_mask,
            return_dict=False,
        )[0].float()
        
        student_latent = student_scheduler.step(velocity, t_step, student_latent, return_dict=False)[0]

# Decode using DistillLoRA method
print("  Decoding student...")
z_dim = model.vae.config.z_dim
latents_mean = (
    torch.tensor(model.vae.config.latents_mean)
    .view(1, z_dim, 1, 1, 1)
    .to(DEVICE, student_latent.dtype)
)
latents_std = (
    1.0 / torch.tensor(model.vae.config.latents_std)
    .view(1, z_dim, 1, 1, 1)
    .to(DEVICE, student_latent.dtype)
)
latents_for_decode = student_latent / latents_std + latents_mean
with torch.no_grad():
    student_image = model.vae.decode(latents_for_decode.to(model.vae.dtype), return_dict=False)[0]

if student_image.ndim == 5:
    student_image = student_image.squeeze(2)
student_image = (student_image / 2 + 0.5).clamp(0, 1)

student_pil = Image.fromarray((student_image[0].permute(1, 2, 0).cpu().to(torch.float32).numpy() * 255).astype("uint8"))
student_path = os.path.join(OUT_DIR, "test_student_20steps_cfg1.png")
student_pil.save(student_path)
print(f"  Saved: {student_path}")

# ---- TEST 2: Teacher sampling (30 steps, CFG=5.0, no_grad) ----
print("\nTest 2: Teacher sampling (30 steps, CFG=5.0)...")

# Encode negative prompt
text_encoder_output_neg = model.encode_text(
    train_device=torch.device(DEVICE),
    batch_size=1,
    rand=__import__('random').Random(SEED),
    text=["worst quality, low quality"],
)
text_encoder_output_neg_dtype = text_encoder_output_neg.to(dtype=model.transformer.dtype)

teacher_scheduler = copy.deepcopy(model.noise_scheduler)
teacher_scheduler.set_timesteps(30, device=DEVICE)
teacher_timesteps = teacher_scheduler.timesteps
teacher_sigmas = teacher_scheduler.sigmas.to(DEVICE)

teacher_latent = noise.clone()

with torch.no_grad():
    for i, t_step in enumerate(teacher_timesteps):
        sigma = teacher_sigmas[i]
        timestep_t = sigma.expand(teacher_latent.shape[0]).to(model.transformer.dtype)
        latent_input = teacher_latent.to(dtype=model.transformer.dtype)
        
        velocity_cond = model.transformer(
            hidden_states=latent_input,
            timestep=timestep_t,
            encoder_hidden_states=text_encoder_output_dtype,
            padding_mask=padding_mask,
            return_dict=False,
        )[0].float()
        
        velocity_uncond = model.transformer(
            hidden_states=latent_input,
            timestep=timestep_t,
            encoder_hidden_states=text_encoder_output_neg_dtype,
            padding_mask=padding_mask,
            return_dict=False,
        )[0].float()
        
        velocity = velocity_uncond + 5.0 * (velocity_cond - velocity_uncond)
        teacher_latent = teacher_scheduler.step(velocity, t_step, teacher_latent, return_dict=False)[0]

print("  Decoding teacher...")
latents_for_decode = teacher_latent / latents_std + latents_mean
with torch.no_grad():
    teacher_image = model.vae.decode(latents_for_decode.to(model.vae.dtype), return_dict=False)[0]

if teacher_image.ndim == 5:
    teacher_image = teacher_image.squeeze(2)
teacher_image = (teacher_image / 2 + 0.5).clamp(0, 1)

teacher_pil = Image.fromarray((teacher_image[0].permute(1, 2, 0).cpu().to(torch.float32).numpy() * 255).astype("uint8"))
teacher_path = os.path.join(OUT_DIR, "test_teacher_30steps_cfg5.png")
teacher_pil.save(teacher_path)
print(f"  Saved: {teacher_path}")

print("\n" + "=" * 60)
print("Done! Check the images in:", OUT_DIR)
print("=" * 60)
