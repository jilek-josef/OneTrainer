# Copy this file to: modules/modelSetup/ZImageEmbeddingLoRASetup.py
#
# Z-Image Embedding LoRA Setup
# =============================
# Trains Z-Image (or compatible DiT diffusion models) using L2 embedding loss
# with LoRA adapters.
#
# Instead of the standard flow-matching loss (noise prediction), this setup:
#   1. Generates images through full Euler sampling (30 steps) with gradient checkpointing
#   2. Decodes latents to pixels with VAE
#   3. Extracts features using EVA-02-Large (runs on CPU by default)
#   4. Computes L2 loss between generated and reference image embeddings
#
# The EVA-02 feature extractor runs on CPU to save VRAM. Gradients flow from
# CPU features back to GPU transformer through PyTorch's cross-device autograd.
#
# Usage:
#   1. Add TrainingMethod.EMBEDDING_LORA to modules/util/enum/TrainingMethod.py
#   2. Copy this file to modules/modelSetup/ZImageEmbeddingLoRASetup.py
#   3. Select training method: EMBEDDING_LORA
#   4. Select model type: Z_IMAGE

import gc
import math
from pathlib import Path
from random import Random
from typing import Callable

import timm
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
from tqdm import tqdm

from transformers import AutoModel, CLIPImageProcessor

from modules.model.ZImageModel import ZImageModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseZImageSetup import BaseZImageSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress


class ZImageEmbeddingLoRASetup(BaseZImageSetup):
    """
    Z-Image training setup that uses L2 embedding loss instead of flow matching.

    Full sampling -> VAE decode -> EVA-02 feature extraction -> L2 embedding loss
    """

    # Feature extractor settings
    FEATURE_EXTRACTOR_NAME: str = "hf-hub:animetimm/eva02_large_patch14_448.dbv4-full"
    FEATURE_EXTRACTOR_RESOLUTION: int = 448
    FEATURE_DEVICE: str = "cpu"  # Run feature extractor on CPU to save VRAM
    RADIO_WEIGHT = 0.7
    EVA_WEIGHT = 0.3

    # Sampling settings
    NUM_INFERENCE_STEPS: int = 30  # Euler steps during training

    # Cached state
    _feature_extractor: torch.nn.Module | None = None
    _ref_embedding: Tensor | None = None
    _ref_embedding_computed: bool = False
    _image_transform: Callable | None = None

    _radio_model = None
    _radio_processor = None

    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    # ======================================================================
    # LoRA Setup (copied from ZImageLoRASetup)
    # ======================================================================

    def create_parameters(
            self,
            model: ZImageModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()
        self._create_model_part_parameters(
            parameter_group_collection, "transformer",
            model.transformer_lora, config.transformer,
        )
        return parameter_group_collection

    def __setup_requires_grad(self, model: ZImageModel, config: TrainConfig):
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)
        self._setup_model_part_requires_grad(
            "transformer", model.transformer_lora,
            config.transformer, model.train_progress,
        )

    def setup_model(self, model: ZImageModel, config: TrainConfig):
        # Setup LoRA
        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "transformer", config,
            config.layer_filter.split(","),
        )
        if model.lora_state_dict:
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

        # Initialize feature extractor (on CPU)
        self._init_feature_extractor()

    def setup_train_device(self, model: ZImageModel, config: TrainConfig):
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = not config.latent_caching

        model.text_encoder_to(
            self.train_device if text_encoder_on_train_device else self.temp_device
        )
        model.vae_to(
            self.train_device if vae_on_train_device else self.temp_device
        )
        model.transformer_to(self.train_device)

        model.text_encoder.eval()
        model.vae.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self, model: ZImageModel, config: TrainConfig,
            train_progress: TrainProgress,
    ):
        self.__setup_requires_grad(model, config)

    # ======================================================================
    # Feature Extractor (EVA-02 on CPU)
    # ======================================================================

    def _init_feature_extractor(self):
        """Load EVA-02-Large feature extractor on CPU."""
        if self._feature_extractor is not None:
            return

        print("EmbeddingLoRA: Loading EVA-02-Large feature extractor on CPU...")
        self._feature_extractor = timm.create_model(
            self.FEATURE_EXTRACTOR_NAME,
            pretrained=True,
            num_classes=0,  # Remove classification head
        )
        self._feature_extractor.to(self.FEATURE_DEVICE)
        self._feature_extractor.eval()
        for p in self._feature_extractor.parameters():
            p.requires_grad = False

        # Image transform for feature extractor
        self._image_transform = transforms.Compose([
            transforms.Resize(
                self.FEATURE_EXTRACTOR_RESOLUTION,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(self.FEATURE_EXTRACTOR_RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        print("EmbeddingLoRA: Feature extractor ready (CPU).")
        print("EmbeddingLoRA: Loading RADIO feature extractor on CPU...")
        hf_repo = "nvidia/C-RADIOv2-g"
        self._radio_processor = CLIPImageProcessor.from_pretrained(hf_repo)
        self._radio_model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
        self._radio_model.eval().to(self.FEATURE_DEVICE)


    # ======================================================================
    # Predict (Full Sampling + VAE Decode)
    # ======================================================================

    def _debug_save_images(
            self,
            images: Tensor,
            prompts: list[str] | None,
            batch_idx: int | None = None,
            output_dir: str | None = None,
            prefix: str = "gen",
    ):
        """
        Save generated images to disk for debugging.

        Args:
            images: Tensor [B, 3, H, W] in [0, 1]
            prompts: List of prompts (optional, saved as text files)
            batch_idx: Batch index for naming
            output_dir: Directory to save to
            prefix: Filename prefix
        """
        if not hasattr(self, 'DEBUG_SAVE_COUNTER'):
            self.DEBUG_SAVE_COUNTER = 0
        if output_dir is None:
            output_dir = "./debug_images"

        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        images_np = images.detach().to(torch.float32).cpu().numpy()

        for i in range(images_np.shape[0]):
            # Build filename
            idx = self.DEBUG_SAVE_COUNTER
            fname = f"{prefix}_{idx:06d}"
            if batch_idx is not None:
                fname += f"_batch{batch_idx}"
            fname += f"_sample{i}"

            # Save image
            img_arr = (images_np[i].transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
            Image.fromarray(img_arr).save(save_dir / f"{fname}.png")

            # Save prompt if available
            if prompts and i < len(prompts):
                with open(save_dir / f"{fname}.txt", "w", encoding="utf-8") as f:
                    f.write(prompts[i])

            self.DEBUG_SAVE_COUNTER += 1

        if batch_idx is not None and batch_idx % 10 == 0:
            print(f"  [DEBUG] Saved {images_np.shape[0]} images to {save_dir} "
                  f"(total: {self.DEBUG_SAVE_COUNTER})")

    def predict(
            self,
            model: ZImageModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        """
        Generate images through full Euler sampling + VAE decode.
        
        Returns images in [0, 1] range instead of flow predictions.
        """
        with model.autocast_context:
            batch_seed = batch.get("seed", None)
            generator = torch.Generator(device=self.train_device)
            if batch_seed is not None:
                generator.manual_seed(batch_seed)
            else:
                generator.seed()

            # Encode text
            batch_seed = 0 if deterministic else train_progress.global_step
            rand = Random(batch_seed)
            text_embeddings = model.encode_text(
                train_device=self.train_device,
                batch_size=batch["latent_image"].shape[0],
                rand=rand,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_output=batch.get("text_encoder_hidden_state"),
                text_encoder_dropout_probability=config.text_encoder.dropout_probability if not deterministic else None,
            )

            # Create noise latents
            # Get latent dimensions from the actual latent tensor
            latent_shape = batch["latent_image"].shape
            latent_height, latent_width = latent_shape[-2], latent_shape[-1]
            #noise = self._create_noise(
            #    source_tensor=batch["latent_image"],
            #    config=config,
            #    generator=generator,
            #)

            noise = torch.randn(
                latent_shape,
                device=self.train_device,
                dtype=torch.float32,
                generator=generator,
            )

            # Full Euler sampling (matches ZImageSampler exactly)
            scheduler: FlowMatchEulerDiscreteScheduler = model.noise_scheduler
            scheduler.set_timesteps(self.NUM_INFERENCE_STEPS, device=self.train_device)
            timesteps = scheduler.timesteps
            sigmas = scheduler.sigmas.to(self.train_device)

            # Denoising loop: state stays float32, cast to train_dtype for transformer
            # CRITICAL: Use raw scheduler timesteps (NOT multiplied by timestep_shift).
            # timestep_shift is for noise scaling only, not for transformer input.
            with model.autocast_context:
                latent = noise
                for i in range(len(timesteps)):
                    t = timesteps[i]
                    t_batch = t.unsqueeze(0).expand(latent.shape[0])

                    def euler_step(latent_in, t_val):
                        latent_input = latent_in.unsqueeze(2).to(
                            dtype=model.train_dtype.torch_dtype()
                        )
                        latent_input_list = list(latent_input.unbind(dim=0))

                        output_list = model.transformer(
                            latent_input_list,
                            (1000 - t_val) / 1000,
                            text_embeddings,
                            return_dict=True,
                        ).sample

                        flow_pred = -torch.stack(output_list, dim=0).squeeze(2)
                        return latent_in + (sigmas[i + 1] - sigmas[i]) * flow_pred

                    if config.gradient_checkpointing.enabled() and i < len(timesteps) - 1:
                        latent = checkpoint(
                            euler_step, latent, t_batch,
                            use_reentrant=False,
                        )
                    else:
                        latent = euler_step(latent, t_batch)

            
            # VAE decode: unscale latents first (ZImageSampler convention)
            with model.autocast_context:
                model.vae_to(self.train_device)
                latents_for_vae = model.unscale_latents(latent)
                # Force gradients through VAE decode (VAE is frozen, but input grads must flow)
                with torch.enable_grad():
                    images = model.vae.decode(latents_for_vae, return_dict=False)[0]

            # Normalize to [0, 1]
            images = (images / 2.0 + 0.5).clamp(0, 1)

            if True:
                prompts = batch.get("prompt", None)
                if isinstance(prompts, str):
                    prompts = [prompts]
                self._debug_save_images(
                    images=images,
                    prompts=prompts,
                    batch_idx=batch.get("idx", None),
                    prefix="train_gen",
                )


            return {
                "images": images,  # [B, 3, H, W] in [0, 1]
            }

    # ======================================================================
    # Calculate Loss (L2 Embedding Loss)
    # ======================================================================

    def calculate_loss(
            self,
            model: ZImageModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        """
        Compute L2 embedding loss between generated images and REAL images.

        Real images are loaded from disk using batch["image_path"], then
        both generated and real images go through EVA-02 on CPU.
        No VAE decode needed for the reference — we use the original pixels.
        """
        gen_images = data["images"]  # [B, 3, H, W] in [0, 1], on GPU

        # Extract bucket resolution (H, W) from batch — aspect ratio is preserved
        crop_res = batch.get("crop_resolution", None)
        target_resolution = None
        if crop_res is not None:
            if torch.is_tensor(crop_res):
                target_resolution = (
                    int(crop_res[0][0] if crop_res.dim() > 1 else crop_res[0]),
                    int(crop_res[0][1] if crop_res.dim() > 1 else crop_res[1]),
                )
            elif isinstance(crop_res, list) and len(crop_res) > 0:
                first = crop_res[0]
                target_resolution = (
                    int(first[0] if isinstance(first, (list, tuple)) else first),
                    int(first[1] if isinstance(first, (list, tuple)) else first),
                )

        # ---- Generated images: tensor -> embed (WITH gradients) ----
        gen_features = self._embed_tensor(gen_images)

        # ---- Reference images: load from disk -> bucket -> embed ----
        image_paths = batch["image_path"]
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        ref_pil = [Image.open(p).convert("RGB") for p in image_paths]

        with torch.no_grad():
            ref_features = self._embed(ref_pil, bucket_size=target_resolution[0] if target_resolution else None)

        loss_eva = F.mse_loss(gen_features, ref_features)

        # Convert generated tensor to PIL for RADIO (RADIO uses its own processor)
        gen_pil = []
        gen_np = gen_images.detach().to(torch.float32).cpu().numpy()
        for i in range(gen_np.shape[0]):
            arr = (gen_np[i].transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
            gen_pil.append(Image.fromarray(arr))

        gen_features_radio = self._embed_pil_radio(gen_pil, target_resolution=target_resolution)
        ref_features_radio = self._embed_pil_radio(ref_pil, target_resolution=target_resolution)

        loss_radio = F.mse_loss(gen_features_radio, ref_features_radio)
        total_loss = self.EVA_WEIGHT * loss_eva + self.RADIO_WEIGHT * loss_radio
        print(f"EVA {loss_eva} RAD {loss_radio}")
        return total_loss


    def _embed(self, pil_images: list[Image.Image], bucket_size: int | None = None) -> Tensor:
        """
        Convert PIL images to EVA-02 embeddings.

        Args:
            pil_images: List of PIL Image objects (RGB mode)
            bucket_size: Target bucket dimension (e.g. 1216). Images are
                         resized so the smaller side equals bucket_size
                         (preserving aspect ratio), then center cropped.

        Returns:
            Feature embeddings [B, D] on CPU
        """
        processed = []
        for img in pil_images:
            if bucket_size is not None:
                # Resize preserving aspect ratio, smaller side = bucket_size
                w, h = img.size
                scale = bucket_size / min(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h), Image.BICUBIC)
                # Center crop to square bucket
                left = (new_w - bucket_size) // 2
                top = (new_h - bucket_size) // 2
                img = img.crop((left, top, left + bucket_size, top + bucket_size))
            processed.append(self._image_transform(img))

        batch = torch.stack(processed).to(self.FEATURE_DEVICE)  # [B, 3, 448, 448]

        # ImageNet normalization
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=batch.device,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=batch.device,
        ).view(1, 3, 1, 1)
        batch = (batch - mean) / std

        with torch.no_grad():
            features = self._feature_extractor(batch)

        return features  # [B, D] on CPU
    
    def _embed_tensor(self, images: Tensor) -> Tensor:
        """
        Convert image tensors [B, 3, H, W] in [0, 1] to EVA-02 embeddings.
        Gradients flow BACK through to the generator. EVA-02 is frozen,
        so no grads accumulate on it — they just pass through.
        """
        # Move to CPU feature extractor device (KEEP gradients!)
        batch = images.to(self.FEATURE_DEVICE)

        _, _, h, w = batch.shape
        target = self.FEATURE_EXTRACTOR_RESOLUTION

        # Resize preserving aspect ratio, smaller side = target
        scale = target / min(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        batch = F.interpolate(batch, size=(new_h, new_w), mode="bilinear", align_corners=False)

        # Center crop to target x target
        top = (new_h - target) // 2
        left = (new_w - target) // 2
        batch = batch[:, :, top:top + target, left:left + target]

        # ImageNet normalization
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=batch.device,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=batch.device,
        ).view(1, 3, 1, 1)
        batch = (batch - mean) / std

        # Feature extractor is frozen — grads pass through to input
        features = self._feature_extractor(batch)

        return features  # [B, D] on CPU, connected to generator graph

    def _embed_pil_radio(
            self,
            pil_images: list[Image.Image],
            target_resolution: tuple[int, int] | None = None,
    ) -> Tensor | None:
        """RADIO embed from PIL images. Resize (cover) + center crop to RADIO's step.
        Always frozen — no gradients. Used for both generated and reference."""

        # Resize to bucket resolution first so RADIO doesn't process huge original images.
        # target_resolution is (H, W) — we resize to the exact bucket dims.
        resized = []
        for img in pil_images:
            if target_resolution is not None:
                target_h, target_w = target_resolution
                img = img.resize((target_w, target_h), Image.BICUBIC)
            resized.append(img)

        # RADIO needs resolution as multiple of its step
        w, h = resized[0].size
        nearest = self._radio_model.get_nearest_supported_resolution(h, w)
        target_w, target_h = nearest.width, nearest.height

        processed = []
        for img in resized:
            iw, ih = img.size
            # Cover resize: scale so both dims >= target, then center crop
            scale = max(target_w / iw, target_h / ih)
            new_w = max(int(iw * scale), target_w)
            new_h = max(int(ih * scale), target_h)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            # Center crop to exact target
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            img = img.crop((left, top, left + target_w, top + target_h))
            processed.append(img)

        pixel_values = self._radio_processor(
            images=processed, return_tensors="pt", do_resize=True
        ).pixel_values.to(self.FEATURE_DEVICE)
        with torch.no_grad():
            summary, _ = self._radio_model(pixel_values)
        return summary  # [B, D] — only the summary tensor


# ======================================================================
# Factory Registration
# ======================================================================

factory.register(
    BaseModelSetup, ZImageEmbeddingLoRASetup,
    ModelType.Z_IMAGE, TrainingMethod.EMBEDDING_LORA,
)

from modules.modelLoader.ZImageModelLoader import ZImageLoRAModelLoader
from modules.modelLoader.BaseModelLoader import BaseModelLoader
factory.register(
    BaseModelLoader, ZImageLoRAModelLoader,
    ModelType.Z_IMAGE, TrainingMethod.EMBEDDING_LORA,
)

from modules.modelSampler.ZImageSampler import ZImageSampler
from modules.modelSampler.BaseModelSampler import BaseModelSampler
factory.register(
    BaseModelSampler, ZImageSampler,
    ModelType.Z_IMAGE, TrainingMethod.EMBEDDING_LORA,
)

# Create proper LoRA saver for EMBEDDING_LORA
# (matches GenericLoRAModelSaver pattern used for regular LORA)
from modules.modelSaver.zImage.ZImageLoRASaver import ZImageLoRASaver
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin

class EmbeddingLoRASaver(BaseModelSaver, InternalModelSaverMixin):
    """Saver for Embedding LoRA — matches BaseModelSaver.save() interface."""
    def save(self, model, model_type, output_model_format, output_model_destination, dtype):
        # Save LoRA weights
        lora_saver = ZImageLoRASaver()
        lora_saver.save(model, output_model_format, output_model_destination, dtype)
        # Save internal data if backing up
        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)

factory.register(
    BaseModelSaver, EmbeddingLoRASaver,
    ModelType.Z_IMAGE, TrainingMethod.EMBEDDING_LORA,
)