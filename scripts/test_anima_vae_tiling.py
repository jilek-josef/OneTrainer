#!/usr/bin/env python3
"""Test Anima VAE decode with tiling to debug grey output issue."""

import sys
sys.path.insert(0, "/home/pc/KimiProjects/ZImage-Loli/OneTrainer")

import torch
from PIL import Image
from diffusers import AutoencoderKLWan

# Load VAE
print("Loading Wan VAE...")
vae = AutoencoderKLWan.from_pretrained(
    "CalamitousFelicitousness/Anima-Preview-3-sdnext-diffusers",
    subfolder="vae",
)
vae = vae.to("cuda", dtype=torch.float16)
vae.eval()

# Wrap VAE for 4D images (add/remove T=1)
orig_decode = vae.decode

def decode_4d(z, **kwargs):
    squeeze = z.ndim == 4
    if squeeze:
        z = z.unsqueeze(2)
    out = orig_decode(z, return_dict=False)
    if isinstance(out, tuple):
        out = out[0]
    return out.squeeze(2) if squeeze else out

vae.decode = decode_4d

# Test at different resolutions
resolutions = [
    (1280, 832),   # user's actual training resolution
    (1024, 1024),  # square
    (512, 512),    # small
]

for w, h in resolutions:
    print(f"\n{'='*60}")
    print(f"Resolution: {w}x{h}")
    latent_h = h // vae.spatial_compression_ratio
    latent_w = w // vae.spatial_compression_ratio
    latent = torch.randn(1, 16, latent_h, latent_w, device="cuda", dtype=torch.float16)
    print(f"Latent shape: {latent.shape}")

    # Test 1: No tiling
    print("\n--- No tiling ---")
    vae.disable_tiling()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            dec = vae.decode(latent)
            if isinstance(dec, tuple):
                dec = dec[0]
        dec = (dec / 2.0 + 0.5).clamp(0, 1)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  OK! shape={dec.shape} min={dec.min():.4f} max={dec.max():.4f} peak={peak:.2f}GB")
        out = (dec[0].cpu().permute(1, 2, 0).to(torch.float32).numpy() * 255).astype("uint8")
        Image.fromarray(out).save(f"/tmp/anima_vae_{w}x{h}_no_tiling.png")
        print(f"  Saved: /tmp/anima_vae_{w}x{h}_no_tiling.png")
    except torch.OutOfMemoryError as e:
        print(f"  OOM: {e}")

    # Test 2: Default tiling (256x256)
    print("\n--- Tiling 256x256 ---")
    vae.enable_tiling()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            dec = vae.decode(latent)
            if isinstance(dec, tuple):
                dec = dec[0]
        dec = (dec / 2.0 + 0.5).clamp(0, 1)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  OK! shape={dec.shape} min={dec.min():.4f} max={dec.max():.4f} peak={peak:.2f}GB")
        out = (dec[0].cpu().permute(1, 2, 0).to(torch.float32).numpy() * 255).astype("uint8")
        Image.fromarray(out).save(f"/tmp/anima_vae_{w}x{h}_tiling256.png")
        print(f"  Saved: /tmp/anima_vae_{w}x{h}_tiling256.png")
    except torch.OutOfMemoryError as e:
        print(f"  OOM: {e}")

    # Test 3: Large tiling (512x512)
    print("\n--- Tiling 512x512 ---")
    vae.enable_tiling(
        tile_sample_min_height=512,
        tile_sample_min_width=512,
        tile_sample_stride_height=384,
        tile_sample_stride_width=384,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            dec = vae.decode(latent)
            if isinstance(dec, tuple):
                dec = dec[0]
        dec = (dec / 2.0 + 0.5).clamp(0, 1)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  OK! shape={dec.shape} min={dec.min():.4f} max={dec.max():.4f} peak={peak:.2f}GB")
        out = (dec[0].cpu().permute(1, 2, 0).to(torch.float32).numpy() * 255).astype("uint8")
        Image.fromarray(out).save(f"/tmp/anima_vae_{w}x{h}_tiling512.png")
        print(f"  Saved: /tmp/anima_vae_{w}x{h}_tiling512.png")
    except torch.OutOfMemoryError as e:
        print(f"  OOM: {e}")

print("\nAll tests completed.")
