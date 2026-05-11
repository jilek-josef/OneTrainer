#!/usr/bin/env python3
"""Test if the transformer responds to different inputs."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.util.import_util import script_imports
script_imports()

import torch
from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.AnimaModelLoader import AnimaModelLoader
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.config.TrainConfig import TrainConfig

DEVICE = "cuda"

train_config = TrainConfig.default_values()
train_config.model_type = ModelType.ANIMA

model = AnimaModel(ModelType.ANIMA)
model.train_dtype = ModelWeightDtypes.from_single_dtype(DataType.BFLOAT_16)

loader = AnimaModelLoader()
loader.load(
    model=model,
    model_type=ModelType.ANIMA,
    model_names=ModelNames(
        base_model='/tmp/anima_base_cfg43nxh',
        transformer_model='/home/pc/Stažené/anima-preview3-base.safetensors',
        text_encoder_model='/home/pc/Stažené/qwen_3_06b_base.safetensors',
        vae_model='/home/pc/Stažené/qwen_image_vae.safetensors',
    ),
    weight_dtypes=model.train_dtype,
    quantization=train_config.quantization,
)
model.to(DEVICE)
model.eval()

print("Model loaded successfully")
print(f"Transformer dtype: {model.transformer.dtype}")

# Test 1: Different latent inputs should produce different outputs
latent_a = torch.randn(1, 16, 1, 64, 64, device=DEVICE, dtype=torch.float32)
latent_b = torch.randn(1, 16, 1, 64, 64, device=DEVICE, dtype=torch.float32)

timestep = torch.tensor([0.5], device=DEVICE, dtype=model.transformer.dtype)
text_embeds = torch.randn(1, 512, 1024, device=DEVICE, dtype=model.transformer.dtype)
padding_mask = latent_a.new_zeros(1, 1, 512, 512, dtype=model.transformer.dtype)

with torch.no_grad():
    out_a = model.transformer(
        hidden_states=latent_a.to(model.transformer.dtype),
        timestep=timestep,
        encoder_hidden_states=text_embeds,
        padding_mask=padding_mask,
        return_dict=False,
    )[0]
    out_b = model.transformer(
        hidden_states=latent_b.to(model.transformer.dtype),
        timestep=timestep,
        encoder_hidden_states=text_embeds,
        padding_mask=padding_mask,
        return_dict=False,
    )[0]

print(f"\nTest 1: Different inputs")
print(f"  out_a mean={out_a.float().mean().item():.6f} std={out_a.float().std().item():.6f}")
print(f"  out_b mean={out_b.float().mean().item():.6f} std={out_b.float().std().item():.6f}")
print(f"  diff mean={(out_a - out_b).float().mean().item():.6f} std={(out_a - out_b).float().std().item():.6f}")
print(f"  max diff={(out_a - out_b).float().abs().max().item():.6f}")

# Test 2: Different timesteps should produce different outputs
with torch.no_grad():
    out_t0 = model.transformer(
        hidden_states=latent_a.to(model.transformer.dtype),
        timestep=torch.tensor([0.0], device=DEVICE, dtype=model.transformer.dtype),
        encoder_hidden_states=text_embeds,
        padding_mask=padding_mask,
        return_dict=False,
    )[0]
    out_t1 = model.transformer(
        hidden_states=latent_a.to(model.transformer.dtype),
        timestep=torch.tensor([0.5], device=DEVICE, dtype=model.transformer.dtype),
        encoder_hidden_states=text_embeds,
        padding_mask=padding_mask,
        return_dict=False,
    )[0]

print(f"\nTest 2: Different timesteps")
print(f"  out_t0 mean={out_t0.float().mean().item():.6f} std={out_t0.float().std().item():.6f}")
print(f"  out_t1 mean={out_t1.float().mean().item():.6f} std={out_t1.float().std().item():.6f}")
print(f"  diff mean={(out_t0 - out_t1).float().mean().item():.6f} std={(out_t0 - out_t1).float().std().item():.6f}")
print(f"  max diff={(out_t0 - out_t1).float().abs().max().item():.6f}")

# Test 3: Different text embeddings should produce different outputs
text_a = torch.randn(1, 512, 1024, device=DEVICE, dtype=model.transformer.dtype)
text_b = torch.randn(1, 512, 1024, device=DEVICE, dtype=model.transformer.dtype)

with torch.no_grad():
    out_text_a = model.transformer(
        hidden_states=latent_a.to(model.transformer.dtype),
        timestep=timestep,
        encoder_hidden_states=text_a,
        padding_mask=padding_mask,
        return_dict=False,
    )[0]
    out_text_b = model.transformer(
        hidden_states=latent_a.to(model.transformer.dtype),
        timestep=timestep,
        encoder_hidden_states=text_b,
        padding_mask=padding_mask,
        return_dict=False,
    )[0]

print(f"\nTest 3: Different text embeddings")
print(f"  out_text_a mean={out_text_a.float().mean().item():.6f} std={out_text_a.float().std().item():.6f}")
print(f"  out_text_b mean={out_text_b.float().mean().item():.6f} std={out_text_b.float().std().item():.6f}")
print(f"  diff mean={(out_text_a - out_text_b).float().mean().item():.6f} std={(out_text_a - out_text_b).float().std().item():.6f}")
print(f"  max diff={(out_text_a - out_text_b).float().abs().max().item():.6f}")

# Test 4: Check if patch_embed weights are actually loaded
patch_embed = model.transformer.patch_embed
print(f"\nTest 4: patch_embed")
print(f"  proj weight mean={patch_embed.proj.weight.float().mean().item():.6f} std={patch_embed.proj.weight.float().std().item():.6f}")
print(f"  proj weight shape={patch_embed.proj.weight.shape}")

# Test 5: Check first transformer block weights
block0 = model.transformer.transformer_blocks[0]
print(f"\nTest 5: Block 0")
print(f"  attn1.to_q weight mean={block0.attn1.to_q.weight.float().mean().item():.6f}")
print(f"  attn2.to_q weight mean={block0.attn2.to_q.weight.float().mean().item():.6f}")
print(f"  ff.net.0.proj weight mean={block0.ff.net[0].proj.weight.float().mean().item():.6f}")
