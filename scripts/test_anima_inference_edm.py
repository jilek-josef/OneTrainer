#!/usr/bin/env python3
"""
Test Anima inference using the EXACT diffusers Cosmos2TextToImagePipeline denoising logic.

Key differences from the old broken test:
1. Latents initialized as randn() * sigma_max (80.0), NOT VAE-scaled
2. Uses EDM c_in/c_skip/c_out scaling with scheduler sigmas in [0,1]
3. VAE decode uses the pipeline's exact unscale formula
4. 5D tensors throughout (B, C, T, H, W) with T=1
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.util.import_util import script_imports
script_imports()

import torch
import numpy as np
from PIL import Image

from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.AnimaModelLoader import AnimaModelLoader
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

DEVICE = "cuda"
DTYPE = torch.bfloat16
HEIGHT = 512
WIDTH = 512
NUM_STEPS = 28
SEED = 42

PROMPT = "a beautiful anime girl with long silver hair, blue eyes, detailed face, high quality"

TRANSFORMER_PATH = "/home/pc/Stažené/anima-preview3-base.safetensors"
TEXT_ENCODER_PATH = "/home/pc/Stažené/qwen_3_06b_base.safetensors"
VAE_PATH = "/home/pc/Stažené/qwen_image_vae.safetensors"
HF_ANIMA_REPO = "circlestone-labs/Anima"
HF_COSMOS_REPO = "nvidia/Cosmos-Predict2-2B-Text2Image"


def main():
    print("=" * 60)
    print("Anima Inference Test — EDM formulation (diffusers pipeline exact)")
    print("=" * 60)

    # 1. Load model
    print("\n[1/5] Loading model...")
    train_config = TrainConfig.default_values()
    train_config.model_type = ModelType.ANIMA
    train_config.weight_dtype = DataType.BFLOAT_16
    train_config.text_encoder.weight_dtype = DataType.BFLOAT_16
    train_config.text_encoder_2.weight_dtype = DataType.BFLOAT_16
    train_config.prior.weight_dtype = DataType.BFLOAT_16
    train_config.lora_rank = 0

    model = AnimaModel(ModelType.ANIMA)
    model_weight_dtypes = ModelWeightDtypes.from_single_dtype(DataType.BFLOAT_16)
    model.train_dtype = model_weight_dtypes

    # Need a base model dir with scheduler + tokenizer configs
    # First search for existing temp dir with configs
    import shutil
    existing_tmp = None
    for d in os.listdir("/tmp"):
        if d.startswith("anima_base_"):
            candidate = os.path.join("/tmp", d)
            has_scheduler = os.path.exists(os.path.join(candidate, "scheduler", "scheduler_config.json"))
            has_tokenizer = os.path.exists(os.path.join(candidate, "tokenizer", "tokenizer.json"))
            if has_scheduler and has_tokenizer:
                existing_tmp = candidate
                break

    if existing_tmp:
        base_model_dir = existing_tmp
        print(f"  Reused existing base config from {existing_tmp}")
    else:
        import tempfile
        base_model_dir = tempfile.mkdtemp(prefix="anima_base_")
        os.makedirs(os.path.join(base_model_dir, "scheduler"), exist_ok=True)
        os.makedirs(os.path.join(base_model_dir, "tokenizer"), exist_ok=True)
        print(f"  [WARN] No existing base config found, creating minimal one at {base_model_dir}...")
        # Write minimal scheduler config
        with open(os.path.join(base_model_dir, "scheduler", "scheduler_config.json"), "w") as f:
            f.write('{"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": 1.0}')
        # We still need tokenizer files - this will likely fail without them

    loader = AnimaModelLoader()
    loader.load(
        model=model,
        model_type=ModelType.ANIMA,
        model_names=ModelNames(
            base_model=base_model_dir,
            transformer_model=TRANSFORMER_PATH,
            text_encoder_model=TEXT_ENCODER_PATH,
            vae_model=VAE_PATH,
        ),
        weight_dtypes=model_weight_dtypes,
        quantization=train_config.quantization,
    )

    model.to(DEVICE)
    model.eval()
    print(f"  Model loaded. Transformer dtype: {model.transformer.dtype}")
    print(f"  VAE latent_mean: {model.vae.config.latents_mean}")
    print(f"  VAE latent_std: {model.vae.config.latents_std}")
    print(f"  Scheduler config sigma_max: {model.noise_scheduler.config.sigma_max}")
    print(f"  Scheduler config sigma_min: {model.noise_scheduler.config.sigma_min}")
    print(f"  Scheduler config sigma_data: {model.noise_scheduler.config.sigma_data}")

    # 2. Encode text (Qwen3 + LLM Adapter)
    print("\n[2/5] Encoding prompt...")
    with torch.no_grad():
        text_encoder_output = model.encode_text(
            train_device=torch.device(DEVICE),
            text=[PROMPT],
            batch_size=1,
        )
    print(f"  Prompt embeddings shape: {text_encoder_output.shape}, dtype: {text_encoder_output.dtype}")

    # 3. Prepare scheduler with EXACT pipeline logic
    print(f"\n[3/5] Preparing scheduler ({NUM_STEPS} steps)...")
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    scheduler.register_to_config(
        sigma_max=80.0,
        sigma_min=0.002,
        sigma_data=1.0,
        final_sigmas_type="sigma_min",
    )

    # Use scheduler's default sigmas (descending 1.0 -> 0.0) instead of pipeline's
    # ascending linspace which causes division by zero at first step.
    # The pipeline's torch.linspace(0, 1, N) appears to be a bug — EDM needs
    # sigma to decrease from high to low.
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = scheduler.timesteps

    if scheduler.config.get("final_sigmas_type", "zero") == "sigma_min":
        scheduler.sigmas[-1] = scheduler.sigmas[-2]

    print(f"  Timesteps: {timesteps[:5].tolist()}... (len={len(timesteps)})")
    print(f"  Sigmas: {scheduler.sigmas[:5].tolist()}... (len={len(scheduler.sigmas)})")

    # 4. Prepare latents (EXACT pipeline logic)
    print(f"\n[4/5] Preparing latents ({HEIGHT}x{WIDTH})...")
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    vae_scale_factor_temporal = 2 ** sum(model.vae.temperal_downsample)
    vae_scale_factor_spatial = 2 ** len(model.vae.temperal_downsample)
    num_latent_frames = (1 - 1) // vae_scale_factor_temporal + 1  # = 1
    latent_height = HEIGHT // vae_scale_factor_spatial
    latent_width = WIDTH // vae_scale_factor_spatial
    latent_shape = (1, model.transformer.config.in_channels, num_latent_frames, latent_height, latent_width)

    print(f"  VAE spatial scale: {vae_scale_factor_spatial}, temporal scale: {vae_scale_factor_temporal}")
    print(f"  Latent shape: {latent_shape}")

    latents = torch.randn(latent_shape, generator=generator, device=DEVICE, dtype=torch.float32)
    latents = latents * scheduler.config.sigma_max  # scale by sigma_max=80
    print(f"  Latents init: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")

    # Padding mask — pipeline uses image dims, transformer resizes internally
    padding_mask = latents.new_zeros(1, 1, HEIGHT, WIDTH, dtype=model.transformer.dtype)

    # 5. Denoising loop (EXACT pipeline logic)
    print(f"\n[5/5] Denoising loop...")
    transformer_dtype = model.transformer.dtype

    for i, t in enumerate(timesteps):
        current_sigma = scheduler.sigmas[i]
        current_t = current_sigma / (current_sigma + 1)
        c_in = 1 - current_t
        c_skip = 1 - current_t
        c_out = -current_t
        timestep = current_t.expand(latents.shape[0]).to(transformer_dtype)

        latent_model_input = latents * c_in
        latent_model_input = latent_model_input.to(transformer_dtype)

        with torch.no_grad():
            output = model.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=text_encoder_output.to(dtype=transformer_dtype),
                padding_mask=padding_mask,
                return_dict=False,
            )[0]

        noise_pred = (c_skip * latents + c_out * output.float()).to(transformer_dtype)
        noise_pred = (latents - noise_pred) / current_sigma

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if i % 7 == 0 or i == len(timesteps) - 1:
            print(f"  Step {i+1}/{len(timesteps)} | sigma={current_sigma:.4f} | t={current_t:.4f} | "
                  f"latents mean={latents.mean().item():.4f} std={latents.std().item():.4f}")

    # 6. VAE decode (EXACT pipeline logic)
    print("\n[6/6] VAE decoding...")
    with torch.no_grad():
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

        # Pipeline exact formula: latents / latents_std / sigma_data + latents_mean
        latents_for_decode = latents / latents_std / scheduler.config.sigma_data + latents_mean
        print(f"  Latents for decode: mean={latents_for_decode.mean().item():.4f}, std={latents_for_decode.std().item():.4f}")

        video = model.vae.decode(latents_for_decode.to(model.vae.dtype), return_dict=False)[0]
        print(f"  VAE output shape: {video.shape}")

    # 7. Post-process
    # video is [B, C, T, H, W], extract first frame
    image_tensor = video[:, :, 0, :, :]  # [B, C, H, W]
    image_tensor = image_tensor.float()

    # The pipeline's VideoProcessor does: video = (video / 2 + 0.5).clamp(0, 1)
    # Since we bypass VideoProcessor, do it manually
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)

    # Convert to PIL
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    image = Image.fromarray(image_np)

    output_path = os.path.join(os.path.dirname(__file__), "anima_test_output_edm.png")
    image.save(output_path)
    print(f"\n  Image saved to: {output_path}")
    print(f"  Image size: {image.size}")
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
