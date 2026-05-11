#!/usr/bin/env python3
"""Test custom tiled decode output quality."""

import sys
sys.path.insert(0, "/home/pc/KimiProjects/ZImage-Loli/OneTrainer")

import torch
from torch.utils.checkpoint import checkpoint
from diffusers import AutoencoderKLWan
from PIL import Image

print("Loading Wan VAE...")
vae = AutoencoderKLWan.from_pretrained(
    "CalamitousFelicitousness/Anima-Preview-3-sdnext-diffusers",
    subfolder="vae",
)
vae = vae.to("cuda", dtype=torch.float16)
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

# Wrap decode for 4D
orig_decode = vae.decode

def decode_4d(z, **kwargs):
    squeeze = z.ndim == 4
    if squeeze:
        z = z.unsqueeze(2)
    out = orig_decode(z, return_dict=False)
    out = out[0]
    return out.squeeze(2) if squeeze else out

vae.decode = decode_4d

# Create a structured test latent (not random — use a pattern so we can visually verify)
w, h = 1280, 832
latent_w = w // vae.spatial_compression_ratio
latent_h = h // vae.spatial_compression_ratio

# Create latent with a clear pattern: different values in different quadrants
z = torch.zeros(1, 16, latent_h, latent_w, device="cuda", dtype=torch.float16)
z[:, :, :latent_h//2, :latent_w//2] = 1.0   # top-left: high
z[:, :, :latent_h//2, latent_w//2:] = -1.0  # top-right: low
z[:, :, latent_h//2:, :latent_w//2] = 0.5   # bottom-left: medium
z[:, :, latent_h//2:, latent_w//2:] = -0.5  # bottom-right: medium-low
z.requires_grad = True

print(f"Latent: {z.shape} pattern=[TL=1, TR=-1, BL=0.5, BR=-0.5]")

# Test 1: No tiling (no_grad) as ground truth
print("\n=== Ground truth: no tiling, no_grad ===")
vae.disable_tiling()
with torch.no_grad():
    out_gt = vae.decode(z)
out_gt = (out_gt / 2.0 + 0.5).clamp(0, 1)
print(f"  shape={out_gt.shape} min={out_gt.min():.4f} max={out_gt.max():.4f}")
img_gt = (out_gt[0].cpu().permute(1,2,0).to(torch.float32).numpy() * 255).astype("uint8")
Image.fromarray(img_gt).save("/tmp/anima_custom_gt.png")
print("  Saved: /tmp/anima_custom_gt.png")

# Test 2: Custom tiled decode with gradients
print("\n=== Custom tiled decode WITH grad ===")
vae.enable_tiling(
    tile_sample_min_height=256,
    tile_sample_min_width=256,
    tile_sample_stride_height=192,
    tile_sample_stride_width=192,
)

def _vae_decode_tiled(z_in):
    _, _, num_frames, height, width = z_in.shape
    sc = vae.spatial_compression_ratio
    tile_h = vae.tile_sample_min_height // sc
    tile_w = vae.tile_sample_min_width // sc
    stride_h = vae.tile_sample_stride_height // sc
    stride_w = vae.tile_sample_stride_width // sc
    blend_h = vae.tile_sample_min_height - vae.tile_sample_stride_height
    blend_w = vae.tile_sample_min_width - vae.tile_sample_stride_width

    rows = []
    for i in range(0, height, stride_h):
        row = []
        for j in range(0, width, stride_w):
            vae.clear_cache()
            tile = z_in[:, :, 0:1, i : i + tile_h, j : j + tile_w]
            tile = vae.post_quant_conv(tile)

            def _decode_tile(t):
                vae.clear_cache()
                vae._conv_idx = [0]
                return vae.decoder(
                    t, feat_cache=vae._feat_map, feat_idx=vae._conv_idx, first_chunk=True
                )

            decoded = checkpoint(_decode_tile, tile, use_reentrant=False)
            row.append(decoded)
        rows.append(row)

    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = vae.blend_v(rows[i - 1][j], tile, blend_h)
            if j > 0:
                tile = vae.blend_h(row[j - 1], tile, blend_w)
            result_row.append(tile[:, :, :, : stride_h * sc, : stride_w * sc])
        result_rows.append(torch.cat(result_row, dim=-1))
    dec = torch.cat(result_rows, dim=3)[:, :, :, : height * sc, : width * sc]
    dec = torch.clamp(dec, min=-1.0, max=1.0)
    return dec

z_copy = z.detach().requires_grad_(True)
out_custom = _vae_decode_tiled(z_copy.unsqueeze(2))
out_custom = (out_custom / 2.0 + 0.5).clamp(0, 1)
if out_custom.dim() == 5:
    out_custom = out_custom.squeeze(2)

print(f"  shape={out_custom.shape} min={out_custom.min():.4f} max={out_custom.max():.4f}")

# Compare
print(f"\n=== Comparison ===")
print(f"MSE(custom vs gt): {torch.nn.functional.mse_loss(out_gt, out_custom).item():.8f}")
print(f"Max diff: {(out_gt - out_custom).abs().max().item():.6f}")

# Save custom output
img_custom = (out_custom[0].detach().cpu().permute(1,2,0).to(torch.float32).numpy() * 255).astype("uint8")
Image.fromarray(img_custom).save("/tmp/anima_custom_grad.png")
print("  Saved: /tmp/anima_custom_grad.png")

# Also test backward
loss = out_custom.mean()
loss.backward()
print(f"  loss={loss.item():.6f} grad_norm={z_copy.grad.norm():.4f}")

print("\nDone.")
