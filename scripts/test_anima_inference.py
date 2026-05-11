#!/usr/bin/env python3
"""
Test inference for Anima model using EDM formulation.

Usage:
    cd OneTrainer
    source venv/bin/activate
    python scripts/test_anima_inference.py
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

from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.AnimaModelLoader import AnimaModelLoader
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType

# ── Configurable paths ──────────────────────────────────────────────────────
TRANSFORMER_PATH = "/home/pc/Stažené/anima-preview3-base.safetensors"
TEXT_ENCODER_PATH = "/home/pc/Stažené/qwen_3_06b_base.safetensors"
VAE_PATH = "/home/pc/Stažené/qwen_image_vae.safetensors"

HF_ANIMA_REPO = "circlestone-labs/Anima"
HF_COSMOS_REPO = "nvidia/Cosmos-Predict2-2B-Text2Image"

OUTPUT_PATH = "/home/pc/Stažené/anima_test_output.png"

# ── Generation settings ─────────────────────────────────────────────────────
DEVICE = "cuda"
PROMPT = "masterpiece, best quality, score_7, safe, 1girl, anime style, high quality"
NEGATIVE_PROMPT = "worst quality, low quality, score_1, score_2, score_3"
HEIGHT = 1024
WIDTH = 1024
STEPS = 35
CFG_SCALE = 4.5
SEED = 42


def check_file(path, name):
    if os.path.exists(path):
        print(f"  [OK] {name}: {path}")
        return True
    print(f"  [MISSING] {name}: {path}")
    return False


def hf_download(repo_id, filename, local_path):
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"\n  [DOWNLOAD] {repo_id}/{filename} -> {local_path}")
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=os.path.dirname(local_path),
            local_dir_use_symlinks=False,
        )
        print(f"  [OK] Downloaded to {downloaded}")
        return downloaded
    except Exception as e:
        print(f"  [ERROR] Failed to download: {e}")
        return None


def prepare_base_model_dir():
    tmpdir = tempfile.mkdtemp(prefix="anima_base_")
    print(f"\n  [SETUP] Creating temporary base model dir: {tmpdir}")

    scheduler_dir = os.path.join(tmpdir, "scheduler")
    os.makedirs(scheduler_dir, exist_ok=True)
    hf_download(HF_COSMOS_REPO, "scheduler/scheduler_config.json",
                os.path.join(tmpdir, "scheduler_config.json"))

    tokenizer_src = "Qwen/Qwen3-0.6B"
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]
    tokenizer_dir = os.path.join(tmpdir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    for f in tokenizer_files:
        try:
            hf_download(tokenizer_src, f, os.path.join(tokenizer_dir, f))
        except Exception as e:
            print(f"  [WARN] Could not download {f}: {e}")

    try:
        from huggingface_hub import list_repo_files
        for f in list_repo_files(tokenizer_src):
            if f.startswith("tokenizer") or f in ["vocab.json", "merges.txt", "special_tokens_map.json"]:
                local = os.path.join(tokenizer_dir, f)
                if not os.path.exists(local):
                    hf_download(tokenizer_src, f, local)
    except Exception as e:
        print(f"  [WARN] Could not list tokenizer files: {e}")

    return tmpdir


print("=" * 60)
print("Anima Inference Test — Flow Match + CONST preconditioning")
print("=" * 60)

print("\n[0/5] Checking local files...")
have_transformer = check_file(TRANSFORMER_PATH, "Transformer")
have_te = check_file(TEXT_ENCODER_PATH, "Text Encoder")
have_vae = check_file(VAE_PATH, "VAE")

if not have_transformer:
    print("\n[!] Transformer file is missing. Please set TRANSFORMER_PATH.")
    sys.exit(1)

if not have_te:
    TEXT_ENCODER_PATH = hf_download(HF_ANIMA_REPO, "split_files/text_encoders/qwen_3_06b_base.safetensors", TEXT_ENCODER_PATH)
    if not TEXT_ENCODER_PATH or not os.path.exists(TEXT_ENCODER_PATH):
        print("[!] Text encoder is missing and could not be downloaded.")
        sys.exit(1)

if not have_vae:
    VAE_PATH = hf_download(HF_ANIMA_REPO, "split_files/vae/qwen_image_vae.safetensors", VAE_PATH)
    if not VAE_PATH or not os.path.exists(VAE_PATH):
        print("[!] VAE is missing and could not be downloaded.")
        sys.exit(1)

print("\n[1/5] Preparing base model directory...")
base_model_dir = prepare_base_model_dir()

print("\n[2/5] Creating model...")
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

print("[3/5] Loading weights...")
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
    import traceback
    traceback.print_exc()
    shutil.rmtree(base_model_dir, ignore_errors=True)
    sys.exit(1)

print(f"  Transformer:   {sum(p.numel() for p in model.transformer.parameters()):,} params")
if model.llm_adapter:
    print(f"  LLM Adapter:   {sum(p.numel() for p in model.llm_adapter.parameters()):,} params")
print(f"  Text Encoder:  {sum(p.numel() for p in model.text_encoder.parameters()):,} params")
print(f"  VAE:           {sum(p.numel() for p in model.vae.parameters()):,} params")

print(f"\n[4/5] Moving model to {DEVICE} (dtype=torch.bfloat16)...")
model.to(torch.device(DEVICE))
model.eval()

print(f"[5/5] Generating image...")
print(f"  Prompt: {PROMPT}")
print(f"  Size: {WIDTH}x{HEIGHT}, Steps: {STEPS}, CFG: {CFG_SCALE}, Seed: {SEED}")

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

    print(f"  Text embed (cond): mean={text_encoder_output.mean().item():.4f}, std={text_encoder_output.std().item():.4f}")
    print(f"  Text embed (uncond): mean={text_encoder_output_neg.mean().item():.4f}, std={text_encoder_output_neg.std().item():.4f}")

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(SEED)

    vae_scale_factor_temporal = 2 ** sum(model.vae.temperal_downsample)
    vae_scale_factor_spatial = 2 ** len(model.vae.temperal_downsample)
    num_latent_frames = (1 - 1) // vae_scale_factor_temporal + 1
    latent_height = HEIGHT // vae_scale_factor_spatial
    latent_width = WIDTH // vae_scale_factor_spatial
    latent_shape = (1, model.transformer.config.in_channels, num_latent_frames, latent_height, latent_width)

    latents = torch.randn(latent_shape, generator=generator, device=DEVICE, dtype=torch.float32)
    print(f"  Latents init: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")

    from diffusers import FlowMatchEulerDiscreteScheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
    )
    noise_scheduler.set_timesteps(STEPS, device=DEVICE)
    timesteps = noise_scheduler.timesteps

    padding_mask = latents.new_zeros(1, 1, HEIGHT, WIDTH, dtype=model.transformer.dtype)
    transformer_dtype = model.transformer.dtype

    for i, t in enumerate(timesteps):
        sigma = noise_scheduler.sigmas[i]
        timestep = sigma.expand(latents.shape[0]).to(transformer_dtype)
        latent_model_input = latents.to(transformer_dtype)

        velocity_cond = model.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_encoder_output.to(dtype=transformer_dtype),
            padding_mask=padding_mask,
            return_dict=False,
        )[0].float()

        velocity_uncond = model.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_encoder_output_neg.to(dtype=transformer_dtype),
            padding_mask=padding_mask,
            return_dict=False,
        )[0].float()

        velocity = velocity_uncond + CFG_SCALE * (velocity_cond - velocity_uncond)
        latents = noise_scheduler.step(velocity, t, latents, return_dict=False)[0]

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Step {i + 1}/{len(timesteps)} | sigma={sigma:.4f}")

    print("  Decoding...")

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
    image = model.vae.decode(latents_for_decode.to(model.vae.dtype), return_dict=False)[0]

    if image.ndim == 5:
        image = image.squeeze(2)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    pil_image = Image.fromarray((image[0] * 255).astype("uint8"))

    pil_image.save(OUTPUT_PATH)
    print(f"\n[OK] Image saved to: {OUTPUT_PATH}")

shutil.rmtree(base_model_dir, ignore_errors=True)
