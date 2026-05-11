import copy
import gc
from random import Random

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

from modules.model.AnimaModel import AnimaModel
from modules.modelSetup.BaseAnimaSetup import BaseAnimaSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress


# ==============================================================================
# Tiled VAE Decode with Memory-Efficient Backward
# ==============================================================================
# The Wan VAE decoder at 1024x1024 uses ~13GB of activation memory during
# backward (with torch.enable_grad()).  This OOMs even on 24GB cards.
# TiledVAEDecode solves this by:
#   1. Decoding the full latent in forward WITHOUT grad (fast, low memory).
#   2. During backward, decoding overlapping tiles WITH grad and checkpointing,
#      accumulating gradients per-tile.  Peak memory is bounded to one tile.
# ==============================================================================

class TiledVAEDecode(torch.autograd.Function):
    """
    Custom autograd function for memory-efficient VAE decode.
    
    Forward:  decodes full latent without grad (uses VAE's built-in tiling).
    Backward: re-decodes overlapping tiles with checkpointing, accumulating
              gradients.  Peak memory is bounded to ~one tile's activations.
    """

    @staticmethod
    def forward(ctx, z: Tensor, vae, tile_latent_size: int = 32, overlap: int = 4):
        ctx.save_for_backward(z)
        ctx.vae = vae
        ctx.tile_latent_size = tile_latent_size
        ctx.overlap = overlap
        # Forward: no grad needed — VAE's own tiled_decode handles large inputs
        with torch.no_grad():
            out = vae.decode(z, return_dict=False)[0]
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        z, = ctx.saved_tensors
        vae = ctx.vae
        tile_latent_size = ctx.tile_latent_size
        overlap = ctx.overlap

        b, c, t, h, w = z.shape
        grad_z = torch.zeros_like(z)
        stride = tile_latent_size - overlap

        for i in range(0, h, stride):
            for j in range(0, w, stride):
                i_end = min(i + tile_latent_size, h)
                j_end = min(j + tile_latent_size, w)

                # Extract tile and make it require grad
                tile = z[:, :, :, i:i_end, j:j_end].detach().clone().requires_grad_(True)

                def decode_tile(t):
                    return vae.decode(t, return_dict=False)[0]

                with torch.enable_grad():
                    tile_out = checkpoint(decode_tile, tile, use_reentrant=False)

                # Corresponding output gradient region
                tile_grad_out = grad_output[:, :, :, i * 8 : i_end * 8, j * 8 : j_end * 8]

                # Compute gradient w.r.t. this tile
                g = torch.autograd.grad(
                    tile_out, tile,
                    grad_outputs=tile_grad_out,
                    retain_graph=False,
                    create_graph=False,
                )[0]

                # Accumulate (overlap regions get summed — acceptable approximation)
                grad_z[:, :, :, i:i_end, j:j_end] += g

                del tile_out, g, tile
                torch_gc()

        return grad_z, None, None, None


class AnimaEmbeddingLoRASetup(BaseAnimaSetup):
    """
    Anima Embedding LoRA Setup.

    Trains Anima using L2 embedding loss with LoRA adapters instead of
    standard flow-matching loss.

    Full Euler sampling -> VAE decode -> EVA-02/RADIO feature extraction -> L2 loss
    """

    FEATURE_EXTRACTOR_NAME: str = "hf-hub:animetimm/eva02_large_patch14_448.dbv4-full"
    FEATURE_EXTRACTOR_RESOLUTION: int = 448
    FEATURE_DEVICE: str = "cpu"
    RADIO_WEIGHT = 0.7
    EVA_WEIGHT = 0.3
    NUM_INFERENCE_STEPS: int = 30
    CFG_SCALE: float = 1.0  # MUST be 1.0 during training.  CFG > 1 trains the unconditional path to match the reference from an empty prompt, which is impossible and corrupts gradients.  Use CFG during inference, not training.

    _feature_extractor: torch.nn.Module | None = None
    _image_transform: transforms.Compose | None = None
    _radio_model = None
    _radio_processor = None

    @staticmethod
    def _log_vram(label: str):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_alloc = torch.cuda.max_memory_allocated() / 1e9
            print(f"[VRAM] {label}: allocated={alloc:.2f}GB reserved={reserved:.2f}GB max={max_alloc:.2f}GB")

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

    # ==================================================================
    # LoRA Setup
    # ==================================================================

    def create_parameters(
            self,
            model: AnimaModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()
        self._create_model_part_parameters(
            parameter_group_collection, "transformer",
            model.transformer_lora, config.transformer,
        )
        return parameter_group_collection

    def __setup_requires_grad(self, model: AnimaModel, config: TrainConfig):
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)
        if model.llm_adapter is not None:
            model.llm_adapter.requires_grad_(False)

        self._setup_model_part_requires_grad(
            "transformer", model.transformer_lora,
            config.transformer, model.train_progress,
        )

    def setup_model(self, model: AnimaModel, config: TrainConfig):
        # Strip empty/whitespace patterns so an empty layer_filter trains all layers
        layer_filters = [
            p.strip() for p in config.layer_filter.split(",")
            if p.strip()
        ] or None

        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "transformer", config,
            layer_filters,
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

        self._init_feature_extractor()

    def setup_train_device(self, model: AnimaModel, config: TrainConfig):
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
            self, model: AnimaModel, config: TrainConfig,
            train_progress: TrainProgress,
    ):
        self.__setup_requires_grad(model, config)

    # ==================================================================
    # Feature Extractor (EVA-02 + RADIO on CPU)
    # ==================================================================

    def _init_feature_extractor(self):
        if self._feature_extractor is not None:
            return

        print("EmbeddingLoRA: Loading EVA-02-Large feature extractor on CPU...")
        self._feature_extractor = timm.create_model(
            self.FEATURE_EXTRACTOR_NAME,
            pretrained=True,
            num_classes=0,
        )
        self._feature_extractor.to(self.FEATURE_DEVICE)
        self._feature_extractor.eval()
        for p in self._feature_extractor.parameters():
            p.requires_grad = False

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

    # ==================================================================
    # Predict (Full Sampling + VAE Decode)
    # ==================================================================

    def predict(
            self,
            model: AnimaModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step
            generator = torch.Generator(device=self.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            self._log_vram("predict start")
            text_encoder_output = model.encode_text(
                train_device=self.train_device,
                batch_size=batch["latent_image"].shape[0],
                rand=rand,
                text=batch.get("prompt"),
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_output=batch.get("text_encoder_hidden_state"),
                text_encoder_dropout_probability=config.text_encoder.dropout_probability if not deterministic else None,
            )
            self._log_vram("after text encode")

            latent_shape = batch["latent_image"].shape
            noise = torch.randn(
                latent_shape,
                device=self.train_device,
                dtype=torch.float32,
                generator=generator,
            )
            print(f"[DEBUG] noise min={noise.min():.4f} max={noise.max():.4f} mean={noise.mean():.4f} std={noise.std():.4f}")

            # Deep-copy scheduler so set_timesteps() doesn't mutate the model's shared
            # scheduler (matches the inference sampler).
            scheduler = copy.deepcopy(model.noise_scheduler)
            scheduler.set_timesteps(self.NUM_INFERENCE_STEPS, device=self.train_device)
            timesteps = scheduler.timesteps
            sigmas = scheduler.sigmas.to(self.train_device)

            vae_scale_factor_temporal = 2 ** sum(model.vae.temperal_downsample)
            num_latent_frames = (1 - 1) // vae_scale_factor_temporal + 1
            latent = noise.unsqueeze(2).expand(-1, -1, num_latent_frames, -1, -1)

            padding_mask = latent.new_zeros(
                1, 1,
                latent_shape[-2] * (2 ** len(model.vae.temperal_downsample)),
                latent_shape[-1] * (2 ** len(model.vae.temperal_downsample)),
                dtype=model.train_dtype.torch_dtype(),
            )

            self._log_vram("before sampling loop")

            # Compute unconditional text embeddings for CFG.
            # The conditional embeddings may be cached; unconditional are computed on-the-fly.
            do_cfg = self.CFG_SCALE > 1.0
            if do_cfg:
                model.text_encoder_to(self.train_device)
                with model.autocast_context:
                    uncond_text_encoder_output = model.encode_text(
                        train_device=self.train_device,
                        batch_size=batch["latent_image"].shape[0],
                        text=[""],
                    )
                model.text_encoder_to(self.temp_device)
                torch_gc()

            # The trainer calls setup_train_device() before each step, which puts the
            # transformer in train() mode.  Sampling must be done in eval() mode —
            # otherwise dropout (and any other train-only behaviour) destroys the image.
            # We leave it in eval() for the rest of the step so that step-level
            # checkpoint recomputation during backward sees the same deterministic
            # state (no dropout mask mismatch).
            model.transformer.eval()

            # Explicitly disable LoRA dropout during sampling.  model.transformer.eval()
            # does NOT propagate to the hooked LoRA modules, so we must set dropout
            # to 0 manually and restore it after sampling.
            original_dropout = config.dropout_probability
            model.transformer_lora.set_dropout(0.0)

            # Pre-cast text embeddings to training dtype to avoid repeated conversions
            text_encoder_output_dtype = text_encoder_output.to(dtype=model.train_dtype.torch_dtype())
            if do_cfg:
                uncond_text_encoder_output_dtype = uncond_text_encoder_output.to(dtype=model.train_dtype.torch_dtype())

            # Step-level checkpoint wrapper: this collapses the entire transformer forward
            # of one CFG pass into a single checkpoint node.  Nested with the per-block
            # checkpoints inside the transformer, peak backward memory is bounded to the
            # footprint of one sampling step instead of growing linearly with steps.
            def _transformer_step(hidden_states, timestep, encoder_hidden_states, padding_mask):
                return model.transformer(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]

            with model.autocast_context:
                for i, t in enumerate(timesteps):
                    sigma = sigmas[i]
                    timestep_t = sigma.expand(latent.shape[0]).to(model.train_dtype.torch_dtype())
                    latent_input = latent.to(dtype=model.train_dtype.torch_dtype())

                    velocity_cond = checkpoint(
                        _transformer_step,
                        latent_input,
                        timestep_t,
                        text_encoder_output_dtype,
                        padding_mask,
                        use_reentrant=False,
                    ).float()

                    if do_cfg:
                        velocity_uncond = checkpoint(
                            _transformer_step,
                            latent_input,
                            timestep_t,
                            uncond_text_encoder_output_dtype,
                            padding_mask,
                            use_reentrant=False,
                        ).float()
                        velocity = velocity_uncond + self.CFG_SCALE * (velocity_cond - velocity_uncond)
                    else:
                        velocity = velocity_cond

                    print(f"[DEBUG] step {i} velocity min={velocity.min():.4f} max={velocity.max():.4f} mean={velocity.mean():.4f} std={velocity.std():.4f}")

                    latent = scheduler.step(velocity, t, latent, return_dict=False)[0]
                    print(f"[DEBUG] step {i} latent min={latent.min():.4f} max={latent.max():.4f} mean={latent.mean():.4f} std={latent.std():.4f}")

            # Restore LoRA dropout for the backward pass / next step
            model.transformer_lora.set_dropout(original_dropout)

            self._log_vram("after sampling loop")
            print(f"[LATENT] sampled latent min={latent.min():.4f} max={latent.max():.4f} mean={latent.mean():.4f} std={latent.std():.4f}")

            # Do NOT restore train mode here.  The backward pass will recompute
            # the step-level checkpoints, and dropout must stay disabled to avoid
            # a forward/backward mismatch.  setup_train_device() will restore
            # train mode at the start of the next step if needed.

            # Free unconditional embeddings — no longer needed.
            if do_cfg:
                del uncond_text_encoder_output
                del uncond_text_encoder_output_dtype
                torch_gc()

            with model.autocast_context:
                model.vae_to(self.train_device)
                latents_for_vae = model.unscale_latents(latent.squeeze(2))
                vae_input = latents_for_vae.unsqueeze(2).to(model.vae.dtype)
                print(f"[SHAPE] VAE decode input: {vae_input.shape} (dtype={vae_input.dtype})")
                print(f"[DEBUG] vae_input min={vae_input.min():.4f} max={vae_input.max():.4f} mean={vae_input.mean():.4f} std={vae_input.std():.4f}")
                self._log_vram("before VAE decode")

                # Ensure VAE params don't accumulate gradients (saves memory + compute)
                for p in model.vae.parameters():
                    p.requires_grad = False

                # Use TiledVAEDecode for memory-efficient VAE decode at high resolution.
                # The Wan VAE's built-in tiling works for forward, but backward through
                # the full decoder OOMs at 1024x1024.  TiledVAEDecode decodes without
                # grad in forward, then re-decodes overlapping tiles with checkpointing
                # during backward, keeping peak memory bounded to one tile.
                images = TiledVAEDecode.apply(vae_input, model.vae, 32, 4)

                self._log_vram("after VAE decode")
                print(f"[SHAPE] VAE decode output: {images.shape}")
                print(f"[DEBUG] images min={images.min():.4f} max={images.max():.4f} mean={images.mean():.4f} std={images.std():.4f}")

            images = (images / 2.0 + 0.5).clamp(0, 1)
            if images.dim() == 5:
                images = images.squeeze(2)  # Wan VAE returns 5D; remove T=1
            self._log_vram("predict end")

            # NOTE: Do NOT move VAE off GPU here.  The VAE decode is part of the
            # computation graph (torch.enable_grad() was active), so the VAE
            # weights must stay on the same device as the saved activations until
            # backward() finishes.  Moving the VAE to temp_device now would break
            # backward with a cross-device error.
            torch_gc()

            # Debug: save generated image to disk every step
            try:
                from PIL import Image
                import os
                debug_dir = os.path.join(config.debug_dir, "embedding_lora_gen")
                os.makedirs(debug_dir, exist_ok=True)
                img_np = images[0].detach().cpu().to(torch.float32).permute(1, 2, 0).numpy()
                img_np = (img_np * 255).clip(0, 255).astype("uint8")
                Image.fromarray(img_np).save(
                    os.path.join(debug_dir, f"step_{train_progress.global_step:06d}.png")
                )
            except Exception as e:
                print(f"[DEBUG] Failed to save gen image: {e}")

            return {"images": images}

    # ==================================================================
    # Calculate Loss (L2 Embedding Loss)
    # ==================================================================

    def calculate_loss(
            self,
            model: AnimaModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        gen_images = data["images"]
        self._log_vram("calculate_loss start")

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

        gen_features = self._embed_tensor(gen_images)
        self._log_vram("after EVA gen features")

        image_paths = batch["image_path"]
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        ref_pil = [Image.open(p).convert("RGB") for p in image_paths]

        with torch.no_grad():
            ref_features = self._embed(ref_pil, bucket_size=target_resolution[0] if target_resolution else None)
        self._log_vram("after EVA ref features")

        loss_eva = F.mse_loss(gen_features, ref_features)

        gen_pil = []
        gen_np = gen_images.detach().to(torch.float32).cpu().numpy()
        for i in range(gen_np.shape[0]):
            arr = (gen_np[i].transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
            gen_pil.append(Image.fromarray(arr))

        gen_features_radio = self._embed_pil_radio(gen_pil, target_resolution=target_resolution)
        self._log_vram("after RADIO gen features")
        ref_features_radio = self._embed_pil_radio(ref_pil, target_resolution=target_resolution)
        self._log_vram("after RADIO ref features")

        loss_radio = F.mse_loss(gen_features_radio, ref_features_radio)
        total_loss = self.EVA_WEIGHT * loss_eva + self.RADIO_WEIGHT * loss_radio

        print(f"[LOSS] EVA(raw)={loss_eva.item():.4f} RADIO(raw)={loss_radio.item():.4f} "
              f"EVA(w={self.EVA_WEIGHT})={self.EVA_WEIGHT * loss_eva.item():.4f} "
              f"RADIO(w={self.RADIO_WEIGHT})={self.RADIO_WEIGHT * loss_radio.item():.4f} "
              f"TOTAL={total_loss.item():.4f}")

        # Debug: save reference image alongside generated image every step
        try:
            import os
            debug_dir = os.path.join(config.debug_dir, "embedding_lora_ref")
            os.makedirs(debug_dir, exist_ok=True)
            ref_name = os.path.splitext(os.path.basename(image_paths[0]))[0]
            ref_pil[0].save(
                os.path.join(debug_dir, f"{ref_name}.png")
            )
        except Exception as e:
            print(f"[DEBUG] Failed to save ref image: {e}")

        self._log_vram("calculate_loss end")
        return total_loss

    def _embed(self, pil_images: list[Image.Image], bucket_size: int | None = None) -> Tensor:
        processed = []
        for img in pil_images:
            if bucket_size is not None:
                w, h = img.size
                scale = bucket_size / min(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h), Image.BICUBIC)
                left = (new_w - bucket_size) // 2
                top = (new_h - bucket_size) // 2
                img = img.crop((left, top, left + bucket_size, top + bucket_size))
            processed.append(self._image_transform(img))

        batch = torch.stack(processed).to(self.FEATURE_DEVICE)
        # _image_transform already applies Normalize(mean, std) — do NOT normalize again

        with torch.no_grad():
            features = self._feature_extractor(batch)
        return features

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
        # Resize to bucket resolution first so RADIO doesn't process huge original images.
        # target_resolution is (H, W) — we resize to the exact bucket dims (aspect ratio
        # is already preserved by the dataloader's aspect bucketing).
        resized = []
        for img in pil_images:
            if target_resolution is not None:
                target_h, target_w = target_resolution
                img = img.resize((target_w, target_h), Image.BICUBIC)
            resized.append(img)

        w, h = resized[0].size
        nearest = self._radio_model.get_nearest_supported_resolution(h, w)
        target_w, target_h = nearest.width, nearest.height

        processed = []
        for img in resized:
            iw, ih = img.size
            scale = max(target_w / iw, target_h / ih)
            new_w = max(int(iw * scale), target_w)
            new_h = max(int(ih * scale), target_h)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            img = img.crop((left, top, left + target_w, top + target_h))
            processed.append(img)

        pixel_values = self._radio_processor(
            images=processed, return_tensors="pt", do_resize=True
        ).pixel_values.to(self.FEATURE_DEVICE)
        with torch.no_grad():
            summary, _ = self._radio_model(pixel_values)
        return summary


# ======================================================================
# Factory Registration
# ======================================================================

factory.register(
    BaseModelSetup, AnimaEmbeddingLoRASetup,
    ModelType.ANIMA, TrainingMethod.EMBEDDING_LORA,
)

from modules.modelLoader.AnimaModelLoader import AnimaLoRAModelLoader
from modules.modelLoader.BaseModelLoader import BaseModelLoader
factory.register(
    BaseModelLoader, AnimaLoRAModelLoader,
    ModelType.ANIMA, TrainingMethod.EMBEDDING_LORA,
)

from modules.modelSampler.AnimaSampler import AnimaSampler
from modules.modelSampler.BaseModelSampler import BaseModelSampler
factory.register(
    BaseModelSampler, AnimaSampler,
    ModelType.ANIMA, TrainingMethod.EMBEDDING_LORA,
)

from modules.modelSaver.anima.AnimaLoRASaver import AnimaLoRASaver
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin


class AnimaEmbeddingLoRASaver(BaseModelSaver, InternalModelSaverMixin):
    def save(self, model, model_type, output_model_format, output_model_destination, dtype):
        lora_saver = AnimaLoRASaver()
        lora_saver.save(model, output_model_format, output_model_destination, dtype)
        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)


factory.register(
    BaseModelSaver, AnimaEmbeddingLoRASaver,
    ModelType.ANIMA, TrainingMethod.EMBEDDING_LORA,
)
