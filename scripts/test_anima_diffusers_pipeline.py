#!/usr/bin/env python3
"""
Test Anima inference using the actual diffusers Cosmos2TextToImagePipeline.
This validates whether the issue is with the model components or the inference code.
"""
import os
import sys
import json
import shutil
import tempfile
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Insert repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from PIL import Image

from diffusers import Cosmos2TextToImagePipeline, FlowMatchEulerDiscreteScheduler
from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.AnimaModelLoader import AnimaModelLoader
from modules.util.quantization_util import QuantizationConfig
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.ModelNames import ModelNames
from modules.util.enum.ModelType import ModelType
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
import numpy as np


class DummyCosmosSafetyChecker(ModelMixin, ConfigMixin):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_device_tracker", torch.zeros(1, dtype=torch.float32), persistent=False)

    def check_text_safety(self, prompt: str) -> bool:
        return True

    def check_video_safety(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def to(self, device: str | torch.device = None, dtype: torch.dtype = None):
        module = super().to(device=device, dtype=dtype)
        return module

    @property
    def device(self) -> torch.device:
        return self._device_tracker.device

    @property
    def dtype(self) -> torch.dtype:
        return self._device_tracker.dtype


# Paths
TRANSFORMER_PATH = "/home/pc/Stažené/anima-preview3-base.safetensors"
TEXT_ENCODER_PATH = "/home/pc/Stažené/qwen_3_06b_base.safetensors"
VAE_PATH = "/home/pc/Stažené/qwen_image_vae.safetensors"
OUTPUT_PATH = "/home/pc/Stažené/anima_test_diffusers_pipeline.png"

DEVICE = "cuda"
PROMPT = "masterpiece, best quality, score_7, safe, 1girl, anime style, high quality"
NEGATIVE_PROMPT = "low quality, blurry, deformed"
WIDTH, HEIGHT = 1024, 1024
STEPS = 35
CFG_SCALE = 4.5
SEED = 42

print("=" * 60)
print("Anima Inference Test — Using Diffusers Pipeline Directly")
print("=" * 60)

# Prepare base model dir for tokenizer
base_model_dir = tempfile.mkdtemp(prefix="anima_base_")
os.makedirs(os.path.join(base_model_dir, "scheduler"), exist_ok=True)
os.makedirs(os.path.join(base_model_dir, "tokenizer"), exist_ok=True)

from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="nvidia/Cosmos-Predict2-2B-Text2Image",
    filename="scheduler/scheduler_config.json",
    local_dir=base_model_dir,
    local_dir_use_symlinks=False,
)
hf_hub_download(
    repo_id="Qwen/Qwen3-0.6B",
    filename="tokenizer.json",
    local_dir=os.path.join(base_model_dir, "tokenizer"),
    local_dir_use_symlinks=False,
)
hf_hub_download(
    repo_id="Qwen/Qwen3-0.6B",
    filename="tokenizer_config.json",
    local_dir=os.path.join(base_model_dir, "tokenizer"),
    local_dir_use_symlinks=False,
)

# Create and load model
print("\n[1/3] Loading model components...")
model = AnimaModel(ModelType.ANIMA)
model_names = ModelNames(
    base_model=base_model_dir,
    transformer_model=TRANSFORMER_PATH,
    text_encoder_model=TEXT_ENCODER_PATH,
    vae_model=VAE_PATH,
)

loader = AnimaModelLoader()
weight_dtypes = ModelWeightDtypes(
    train_dtype=torch.bfloat16,
    fallback_train_dtype=torch.bfloat16,
    unet=torch.bfloat16,
    prior=torch.bfloat16,
    transformer=torch.bfloat16,
    text_encoder=torch.bfloat16,
    text_encoder_2=torch.bfloat16,
    text_encoder_3=torch.bfloat16,
    text_encoder_4=torch.bfloat16,
    vae=torch.bfloat16,
    effnet_encoder=torch.bfloat16,
    decoder=torch.bfloat16,
    decoder_text_encoder=torch.bfloat16,
    decoder_vqgan=torch.bfloat16,
    lora=torch.bfloat16,
    embedding=torch.bfloat16,
)
loader.load(
    model_type=ModelType.ANIMA,
    model_names=model_names,
    weight_dtypes=weight_dtypes,
    quantization_config=QuantizationConfig(quantization_mode=None),
)

print(f"  Transformer:   {sum(p.numel() for p in model.transformer.parameters()):,} params")
print(f"  Text Encoder:  {sum(p.numel() for p in model.text_encoder.parameters()):,} params")
print(f"  VAE:           {sum(p.numel() for p in model.vae.parameters()):,} params")

# Move to device
model.to(torch.device(DEVICE))
model.eval()

# Create pipeline with our components
print("\n[2/3] Creating diffusers pipeline...")

# We need a T5 tokenizer for the pipeline. Try to load one.
from transformers import T5TokenizerFast
try:
    t5_tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
except Exception as e:
    print(f"  Failed to load t5-small tokenizer: {e}")
    t5_tokenizer = None

# The pipeline expects T5EncoderModel, but we have Qwen3ForCausalLM + LLM Adapter
# We can't directly use the pipeline's encode_prompt because it expects T5.
# Instead, we'll pre-compute the text embeddings and pass them to the pipeline.

print("\n[3/3] Encoding prompt and running inference...")
with torch.no_grad():
    prompt_embeds = model.encode_text(
        train_device=torch.device(DEVICE),
        text=[PROMPT],
        batch_size=1,
    )
    negative_prompt_embeds = model.encode_text(
        train_device=torch.device(DEVICE),
        text=[NEGATIVE_PROMPT],
        batch_size=1,
    )

print(f"  Prompt embeds: {prompt_embeds.shape}")

# Create scheduler from config
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_dir, subfolder="scheduler")

# Create a minimal pipeline that bypasses the text encoder
# We'll monkey-patch the pipeline's encode_prompt to return our precomputed embeddings
class PatchedPipeline(Cosmos2TextToImagePipeline):
    def __init__(self, transformer, vae, scheduler, prompt_embeds, negative_prompt_embeds):
        # Skip parent __init__ to avoid needing tokenizer/text_encoder
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline
        DiffusionPipeline.__init__(self)
        
        self.register_modules(
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            transformer=transformer,
            scheduler=scheduler,
            safety_checker=DummyCosmosSafetyChecker(),
        )
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        from diffusers.video_processor import VideoProcessor
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        
        self.sigma_max = 80.0
        self.sigma_min = 0.002
        self.sigma_data = 1.0
        self.final_sigmas_type = "sigma_min"
        if self.scheduler is not None:
            self.scheduler.register_to_config(
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                sigma_data=self.sigma_data,
                final_sigmas_type=self.final_sigmas_type,
            )
        
        self._precomputed_prompt_embeds = prompt_embeds
        self._precomputed_negative_prompt_embeds = negative_prompt_embeds
    
    def _get_t5_prompt_embeds(self, **kwargs):
        return self._precomputed_prompt_embeds
    
    def encode_prompt(self, prompt=None, negative_prompt=None, do_classifier_free_guidance=True, **kwargs):
        prompt_embeds = self._precomputed_prompt_embeds
        negative_prompt_embeds = self._precomputed_negative_prompt_embeds
        
        if do_classifier_free_guidance and negative_prompt_embeds is not None:
            # duplicate for num_images_per_prompt
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
        
        return prompt_embeds, negative_prompt_embeds

pipe = PatchedPipeline(
    transformer=model.transformer,
    vae=model.vae,
    scheduler=scheduler,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=True)

generator = torch.Generator(device=DEVICE).manual_seed(SEED)

print(f"  Running with: size={WIDTH}x{HEIGHT}, steps={STEPS}, cfg={CFG_SCALE}, seed={SEED}")

# Run inference
output = pipe(
    prompt="dummy",  # Not used since we override encode_prompt
    negative_prompt="dummy",
    height=HEIGHT,
    width=WIDTH,
    num_inference_steps=STEPS,
    guidance_scale=CFG_SCALE,
    generator=generator,
    output_type="pil",
).images[0]

output.save(OUTPUT_PATH)
print(f"\n[OK] Image saved to: {OUTPUT_PATH}")

shutil.rmtree(base_model_dir, ignore_errors=True)
