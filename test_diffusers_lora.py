#!/usr/bin/env python3
"""
Test with diffusers preconditioning + proper LoRA loading.
"""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, '/home/pc/KimiProjects/ZImage-Loli/OneTrainer')
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
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.enum.ModelType import PeftType
from diffusers import FlowMatchEulerDiscreteScheduler

TRANSFORMER_PATH = "/home/pc/Stažené/anima-preview3-base.safetensors"
TEXT_ENCODER_PATH = "/home/pc/Stažené/qwen_3_06b_base.safetensors"
VAE_PATH = "/home/pc/Stažené/qwen_image_vae.safetensors"
LORA_PATH = "/home/pc/KimiProjects/ZImage-Loli/OneTrainer/workspace/run/backup/2026-05-11_11-10-24-backup-27-0-27/lora/lora.safetensors"
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

def hf_download(repo_id, filename, local_path):
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=os.path.dirname(local_path), local_dir_use_symlinks=False)
    except:
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
print("Diffusers Preconditioning + LoRA Test")
print("=" * 60)

base_model_dir = prepare_base_model_dir()

model = AnimaModel(ModelType.ANIMA)
model_weight_dtypes = ModelWeightDtypes.from_single_dtype(DataType.BFLOAT_16)
config = TrainConfig.default_values()
config.train_device = DEVICE
config.model_type = ModelType.ANIMA

# Load LoRA weights BEFORE loading the model
from safetensors.torch import load_file
model.lora_state_dict = load_file(LORA_PATH)
print(f"Loaded LoRA weights: {len(model.lora_state_dict)} keys")

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
    print(f"[ERROR] {e}")
    shutil.rmtree(base_model_dir, ignore_errors=True)
    sys.exit(1)

# Set up LoRA manually
from modules.module.LoRAModule import LoRAModuleWrapper

config.training_method = TrainingMethod.LORA
config.peft_type = PeftType.LORA
config.lora_rank = 16
config.lora_alpha = 1.0
config.dropout_probability = 0.0
config.layer_filter = ""

model.transformer_lora = LoRAModuleWrapper(
    model.transformer, "transformer", config, None
)

if model.lora_state_dict:
    model.transformer_lora.load_state_dict(model.lora_state_dict)
    model.lora_state_dict = None

model.transformer_lora.prune()
model.transformer_lora.set_dropout(0.0)
model.transformer_lora.to(dtype=torch.bfloat16)
model.transformer_lora.hook_to_module()

model.text_encoder_to("cpu")
model.vae_to("cpu")
model.transformer_to(DEVICE)
model.text_encoder.eval()
model.vae.eval()
model.transformer.eval()

model.to(torch.device(DEVICE))
model.eval()

print(f"LoRA hooked: {model.transformer_lora is not None}")
print(f"LoRA modules: {len(model.transformer_lora.lora_modules)}")

# Encode text
print("Encoding text...")
model.text_encoder_to(DEVICE)
with torch.no_grad():
    prompt_embeds = model.encode_text(
        train_device=torch.device(DEVICE),
        batch_size=1,
        text=[PROMPT],
    )
    negative_prompt_embeds = model.encode_text(
        train_device=torch.device(DEVICE),
        batch_size=1,
        text=[NEGATIVE_PROMPT],
    )
model.text_encoder_to("cpu")
torch.cuda.empty_cache()

# Prepare latents
print("Preparing latents...")
generator = torch.Generator(device=DEVICE)
generator.manual_seed(SEED)

vae_scale_factor_temporal = 2 ** sum(model.vae.temperal_downsample)
vae_scale_factor_spatial = 2 ** len(model.vae.temperal_downsample)
num_latent_frames = (1 - 1) // vae_scale_factor_temporal + 1
latent_height = HEIGHT // vae_scale_factor_spatial
latent_width = WIDTH // vae_scale_factor_spatial
num_channels_latents = model.transformer.config.in_channels

latents = torch.randn(
    (1, num_channels_latents, num_latent_frames, latent_height, latent_width),
    generator=generator,
    device=DEVICE,
    dtype=torch.float32,
)

# Use fresh scheduler
scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
scheduler.set_timesteps(STEPS, device=DEVICE)
timesteps = scheduler.timesteps

padding_mask = latents.new_zeros(1, 1, HEIGHT, WIDTH, dtype=torch.bfloat16)
transformer_dtype = model.transformer.dtype

# Move transformer to device
model.transformer_to(DEVICE)

# Denoising loop with DIFFUSERS PRECONDITIONING + LoRA
print(f"Sampling {STEPS} steps with diffusers preconditioning + LoRA...")
with torch.no_grad():
    for i, t in enumerate(timesteps):
        current_sigma = scheduler.sigmas[i]
        
        # Diffusers preconditioning
        current_t = current_sigma / (current_sigma + 1)
        c_in = 1 - current_t
        c_skip = 1 - current_t
        c_out = -current_t
        
        timestep = current_t.expand(latents.shape[0]).to(transformer_dtype)
        latent_model_input = latents * c_in
        latent_model_input = latent_model_input.to(transformer_dtype)

        noise_pred = model.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
            padding_mask=padding_mask,
            return_dict=False,
        )[0]
        noise_pred = (c_skip * latents + c_out * noise_pred.float()).to(transformer_dtype)
        
        if CFG > 1.0:
            noise_pred_uncond = model.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds.to(dtype=transformer_dtype),
                padding_mask=padding_mask,
                return_dict=False,
            )[0]
            noise_pred_uncond = (c_skip * latents + c_out * noise_pred_uncond.float()).to(transformer_dtype)
            noise_pred = noise_pred + CFG * (noise_pred - noise_pred_uncond)
        
        velocity = (latents - noise_pred.float()) / current_sigma
        latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
        
        if (i + 1) % 5 == 0:
            print(f"  Step {i + 1}/{len(timesteps)}")

model.transformer_to("cpu")
torch.cuda.empty_cache()

# Decode
print("Decoding...")
model.vae_to(DEVICE)

z_dim = model.vae.config.z_dim
latents_mean = (
    torch.tensor(model.vae.config.latents_mean)
    .view(1, z_dim, 1, 1, 1)
    .to(DEVICE, latents.dtype)
)
latents_std = (
    1.0 / torch.tensor(model.vae.config.latents_std)
    .view(1, z_dim, 1, 1, 1)
    .to(DEVICE, latents.dtype)
)
latents_for_decode = latents / latents_std + latents_mean

with torch.no_grad():
    video = model.vae.decode(latents_for_decode.to(model.vae.dtype), return_dict=False)[0]

image = video.squeeze(2)
image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).float().numpy()
pil_image = Image.fromarray((image[0] * 255).astype("uint8"))

pil_image.save(os.path.join(OUT_DIR, "test_diffusers_lora_1024.png"))
arr = np.array(pil_image)
print(f"Saved: mean={arr.mean():.1f}, std={arr.std():.1f}")

model.vae_to("cpu")
shutil.rmtree(base_model_dir, ignore_errors=True)
print("Done!")
