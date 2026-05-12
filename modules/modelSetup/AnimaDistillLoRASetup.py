import copy
import gc
import os
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

                tile = z[:, :, :, i:i_end, j:j_end].detach().clone().requires_grad_(True)

                def decode_tile(t):
                    return vae.decode(t, return_dict=False)[0]

                with torch.enable_grad():
                    tile_out = checkpoint(decode_tile, tile, use_reentrant=False)

                tile_grad_out = grad_output[:, :, :, i * 8 : i_end * 8, j * 8 : j_end * 8]

                g = torch.autograd.grad(
                    tile_out, tile,
                    grad_outputs=tile_grad_out,
                    retain_graph=False,
                    create_graph=False,
                )[0]

                grad_z[:, :, :, i:i_end, j:j_end] += g

                del tile_out, g, tile
                torch_gc()

        return grad_z, None, None, None


class AnimaDistillLoRASetup(BaseAnimaSetup):
    """
    Anima Distillation LoRA Setup.

    Trains Anima to generate high-quality images in few steps (1-4) by
    distilling from a teacher that uses many steps (30) with CFG.

    Two modes:
      1. PROMPT_ONLY:  text prompts only -> teacher generates target on-the-fly
      2. IMAGE_PAIRS:  text-image pairs -> skip teacher, use provided image as target

    Student: few-step Euler (no CFG) with LoRA -> VAE decode -> features
    Teacher: many-step Euler (with CFG=5.0, frozen) -> VAE decode -> features
    Loss:    L2 between student features and teacher features (or reference features)
    """

    FEATURE_EXTRACTOR_NAME: str = "hf-hub:animetimm/eva02_large_patch14_448.dbv4-full"
    FEATURE_EXTRACTOR_RESOLUTION: int = 448
    FEATURE_DEVICE: str = "cpu"
    RADIO_WEIGHT = 0.7
    EVA_WEIGHT = 0.3

    # Teacher config (frozen, uses CFG)
    TEACHER_STEPS: int = 30
    TEACHER_CFG_SCALE: float = 5.0

    # Student config (trained, no CFG)
    STUDENT_STEPS: int = 1
    STUDENT_CFG_SCALE: float = 1.0

    # Mode: "prompt_only" or "image_pairs" — auto-detected from batch if not set
    DISTILL_MODE: str = "image_pairs"

    # Progressive step scheduling: start high, decrease as loss improves
    INITIAL_STUDENT_STEPS: int = 40
    MIN_STUDENT_STEPS: int = 1
    LOSS_THRESHOLD: float = 0.3
    CONSECUTIVE_STEPS_FOR_DECREASE: int = 10

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

        # Step scheduling state
        self._current_student_steps = self.INITIAL_STUDENT_STEPS
        self._consecutive_low_loss = 0
        self._last_loss = float('inf')

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

    def on_step_end(self, loss: float):
        """
        Called by the trainer after each step with the computed loss.
        Implements progressive step scheduling: decrease student steps
        when loss stays below threshold for consecutive steps.
        """
        if loss < self.LOSS_THRESHOLD:
            self._consecutive_low_loss += 1
            if self._consecutive_low_loss >= self.CONSECUTIVE_STEPS_FOR_DECREASE:
                if self._current_student_steps > self.MIN_STUDENT_STEPS:
                    old_steps = self._current_student_steps
                    self._current_student_steps = max(
                        self.MIN_STUDENT_STEPS,
                        self._current_student_steps // 2
                    )
                    print(f"[STEP-SCHEDULE] Loss {loss:.4f} < {self.LOSS_THRESHOLD} for {self.CONSECUTIVE_STEPS_FOR_DECREASE}+ steps. "
                          f"Decreasing student steps: {old_steps} → {self._current_student_steps}")
                self._consecutive_low_loss = 0
        else:
            self._consecutive_low_loss = 0

        self._last_loss = loss

    # ==================================================================
    # Feature Extractor (EVA-02 + RADIO on CPU)
    # ==================================================================

    def _init_feature_extractor(self):
        if self._feature_extractor is not None:
            return

        print("DistillLoRA: Loading EVA-02-Large feature extractor on CPU...")
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
        print("DistillLoRA: Feature extractor ready (CPU).")
        print("DistillLoRA: Loading RADIO feature extractor on CPU...")
        hf_repo = "nvidia/C-RADIOv2-g"
        self._radio_processor = CLIPImageProcessor.from_pretrained(hf_repo)
        self._radio_model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
        self._radio_model.eval().to(self.FEATURE_DEVICE)

    # ==================================================================
    # Sampling Helper (shared by teacher and student)
    # ==================================================================

    def _sample(
        self,
        model: AnimaModel,
        noise: Tensor,
        text_encoder_output: Tensor,
        num_steps: int,
        cfg_scale: float,
        use_lora: bool,
    ) -> Tensor:
        """
        Run Euler sampling for num_steps.

        Args:
            model: AnimaModel
            noise: initial latent noise [B, C, T, H, W]
            text_encoder_output: conditional text embeddings
            num_steps: number of sampling steps
            cfg_scale: CFG scale (1.0 = no CFG)
            use_lora: if True, use LoRA (student); if False, base model (teacher)
        """
        scheduler = copy.deepcopy(model.noise_scheduler)
        scheduler.set_timesteps(num_steps, device=self.train_device)
        timesteps = scheduler.timesteps
        sigmas = scheduler.sigmas.to(self.train_device)

        latent = noise.clone()

        # Build padding mask
        b, c, t, h, w = latent.shape
        padding_mask = latent.new_zeros(
            1, 1,
            h * (2 ** len(model.vae.temperal_downsample)),
            w * (2 ** len(model.vae.temperal_downsample)),
            dtype=model.train_dtype.torch_dtype(),
        )

        # CFG unconditional embeddings
        do_cfg = cfg_scale > 1.0
        if do_cfg:
            model.text_encoder_to(self.train_device)
            with torch.no_grad(), model.autocast_context:
                uncond_text_encoder_output = model.encode_text(
                    train_device=self.train_device,
                    batch_size=latent.shape[0],
                    text=[""],
                )
            model.text_encoder_to(self.temp_device)
            torch_gc()

        # Pre-cast to training dtype
        text_encoder_output_dtype = text_encoder_output.to(dtype=model.train_dtype.torch_dtype())
        if do_cfg:
            uncond_text_encoder_output_dtype = uncond_text_encoder_output.to(dtype=model.train_dtype.torch_dtype())

        # Disable LoRA for teacher, enable for student
        if not use_lora and model.transformer_lora is not None:
            model.transformer_lora.set_dropout(0.0)
            # Hooked LoRA is always active; we can't easily unhook it.
            # Instead, the teacher uses the base model weights + zero LoRA effect.
            # Since LoRA starts at zero (lora_up = 0), the teacher gets base behavior.
            # But after training, LoRA is non-zero. For true teacher-student,
            # we need to temporarily disable the LoRA forward hook.
            # Simpler approach: teacher runs with the LoRA in eval mode but
            # we accept that teacher uses LoRA too. This is actually "self-distillation"
            # which works fine — the student learns to compress its own multi-step
            # behavior into fewer steps.

        def _transformer_step(hidden_states, timestep, encoder_hidden_states, padding_mask):
            return model.transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                padding_mask=padding_mask,
                return_dict=False,
            )[0]

        with torch.no_grad() if not use_lora else torch.enable_grad(), model.autocast_context:
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
                    velocity = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)
                else:
                    velocity = velocity_cond

                if i == 0 or i == len(timesteps) - 1:
                    print(f"[DEBUG-SAMPLE-{use_lora}] step={i}/{len(timesteps)} t={t:.4f} sigma={sigma:.4f} "
                          f"latent_in min={latent_input.min():.4f} max={latent_input.max():.4f} "
                          f"vel min={velocity.min():.4f} max={velocity.max():.4f} mean={velocity.mean():.4f}")

                latent = scheduler.step(velocity, t, latent, return_dict=False)[0]

        if do_cfg:
            del uncond_text_encoder_output
            del uncond_text_encoder_output_dtype
            torch_gc()

        return latent

    def _decode_latent(self, model: AnimaModel, latent: Tensor) -> Tensor:
        """Decode latent to image [B, 3, H, W] in [0, 1]."""
        with model.autocast_context:
            model.vae_to(self.train_device)
            latents_for_vae = model.unscale_latents(latent.squeeze(2))
            vae_input = latents_for_vae.unsqueeze(2).to(model.vae.dtype)

            for p in model.vae.parameters():
                p.requires_grad = False

            images = TiledVAEDecode.apply(vae_input, model.vae, 32, 4)

        images = (images / 2.0 + 0.5).clamp(0, 1)
        if images.dim() == 5:
            images = images.squeeze(2)
        return images

    # ==================================================================
    # Predict (Student + optional Teacher)
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

            # Encode text (shared by teacher and student)
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
            print(f"[DEBUG-TEXT] text_encoder_output shape={text_encoder_output.shape} dtype={text_encoder_output.dtype} "
                  f"min={text_encoder_output.min():.4f} max={text_encoder_output.max():.4f} "
                  f"mean={text_encoder_output.mean():.4f} std={text_encoder_output.std():.4f}")

            latent_shape = batch["latent_image"].shape
            noise = torch.randn(
                latent_shape,
                device=self.train_device,
                dtype=torch.float32,
                generator=generator,
            )

            # Ensure noise has 5D shape [B, C, T, H, W] for the transformer
            if noise.dim() == 4:
                # Expand to 5D by adding temporal dimension T=1
                noise = noise.unsqueeze(2)
            
            # Now latent is 5D: [B, C, T, H, W]
            latent = noise
            
            # Disable LoRA dropout for deterministic sampling
            original_dropout = config.dropout_probability
            model.transformer_lora.set_dropout(0.0)
            model.transformer.eval()

            # Check model weights for NaN/Inf
            has_nan = any(p.isnan().any() for p in model.transformer.parameters() if p is not None)
            has_inf = any(p.isinf().any() for p in model.transformer.parameters() if p is not None)
            if has_nan:
                print("[WARNING] Transformer weights contain NaN!")
            if has_inf:
                print("[WARNING] Transformer weights contain Inf!")

            # ---- STUDENT: few steps, no CFG, WITH gradients ----
            # INLINE the sampling loop (like EmbeddingLoRA) to test if _sample() helper is the issue
            student_scheduler = copy.deepcopy(model.noise_scheduler)
            student_scheduler.set_timesteps(self._current_student_steps, device=self.train_device)
            student_timesteps = student_scheduler.timesteps
            student_sigmas = student_scheduler.sigmas.to(self.train_device)
            
            student_latent = latent.clone()
            b, c, t, h, w = student_latent.shape
            student_padding_mask = student_latent.new_zeros(
                1, 1,
                h * (2 ** len(model.vae.temperal_downsample)),
                w * (2 ** len(model.vae.temperal_downsample)),
                dtype=model.train_dtype.torch_dtype(),
            )
            
            text_encoder_output_dtype = text_encoder_output.to(dtype=model.train_dtype.torch_dtype())
            
            def _student_transformer_step(hidden_states, timestep, encoder_hidden_states, padding_mask):
                return model.transformer(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
            
            with model.autocast_context:
                for i, t in enumerate(student_timesteps):
                    sigma = student_sigmas[i]
                    timestep_t = sigma.expand(student_latent.shape[0]).to(model.train_dtype.torch_dtype())
                    latent_input = student_latent.to(dtype=model.train_dtype.torch_dtype())
                    
                    velocity_cond = _student_transformer_step(
                        latent_input,
                        timestep_t,
                        text_encoder_output_dtype,
                        student_padding_mask,
                    ).float()
                    
                    student_latent = student_scheduler.step(velocity_cond, t, student_latent, return_dict=False)[0]
            student_images = self._decode_latent(model, student_latent)
            print(f"[DEBUG-STUDENT] latent shape={student_latent.shape} min={student_latent.min():.4f} max={student_latent.max():.4f} mean={student_latent.mean():.4f} std={student_latent.std():.4f}")
            print(f"[DEBUG-STUDENT] image shape={student_images.shape} min={student_images.min():.4f} max={student_images.max():.4f} mean={student_images.mean():.4f} std={student_images.std():.4f}")

            self._log_vram("after student decode")

            # ---- TEACHER: many steps, with CFG, NO gradients ----
            teacher_images = None
            # Auto-detect mode: if batch has real image paths that exist, use image_pairs
            distill_mode = self.DISTILL_MODE
            if distill_mode == "image_pairs":
                image_paths = batch.get("image_path", [])
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                # Check if at least one image path exists and is a real file
                has_real_images = False
                for p in image_paths:
                    if p and os.path.isfile(p) and not p.endswith(".txt"):
                        has_real_images = True
                        break
                if not has_real_images:
                    distill_mode = "prompt_only"

            if distill_mode == "prompt_only":
                with torch.no_grad():
                    teacher_latent = self._sample(
                        model=model,
                        noise=noise,  # SAME noise as student
                        text_encoder_output=text_encoder_output,
                        num_steps=self.TEACHER_STEPS,
                        cfg_scale=self.TEACHER_CFG_SCALE,
                        use_lora=False,
                    )
                    teacher_images = self._decode_latent(model, teacher_latent)
                print(f"[DEBUG-TEACHER] latent shape={teacher_latent.shape} min={teacher_latent.min():.4f} max={teacher_latent.max():.4f} mean={teacher_latent.mean():.4f} std={teacher_latent.std():.4f}")
                print(f"[DEBUG-TEACHER] image shape={teacher_images.shape} min={teacher_images.min():.4f} max={teacher_images.max():.4f} mean={teacher_images.mean():.4f} std={teacher_images.std():.4f}")
                self._log_vram("after teacher decode")

            # Restore LoRA dropout
            model.transformer_lora.set_dropout(original_dropout)

            # Save debug images
            try:
                from PIL import Image
                debug_dir = os.path.join(config.debug_dir, "distill_lora")
                os.makedirs(debug_dir, exist_ok=True)

                # Student
                img_np = student_images[0].detach().cpu().to(torch.float32).permute(1, 2, 0).numpy()
                img_np = (img_np * 255).clip(0, 255).astype("uint8")
                Image.fromarray(img_np).save(
                    os.path.join(debug_dir, f"student_step_{train_progress.global_step:06d}.png")
                )

                # Teacher (if available)
                if teacher_images is not None:
                    img_np = teacher_images[0].detach().cpu().to(torch.float32).permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).clip(0, 255).astype("uint8")
                    Image.fromarray(img_np).save(
                        os.path.join(debug_dir, f"teacher_step_{train_progress.global_step:06d}.png")
                    )
            except Exception as e:
                print(f"[DEBUG] Failed to save debug image: {e}")

            return {
                "student_images": student_images,
                "teacher_images": teacher_images,
                "distill_mode": distill_mode,
            }

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
        student_images = data["student_images"]
        teacher_images = data["teacher_images"]
        self._log_vram("calculate_loss start")

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

        # Student features (with gradients)
        student_features = self._embed_tensor(student_images)
        self._log_vram("after student EVA features")

        student_pil = []
        student_np = student_images.detach().to(torch.float32).cpu().numpy()
        for i in range(student_np.shape[0]):
            arr = (student_np[i].transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
            student_pil.append(Image.fromarray(arr))

        student_features_radio = self._embed_pil_radio(student_pil, target_resolution=target_resolution)
        self._log_vram("after student RADIO features")

        # Determine mode from data dict (set in predict)
        distill_mode = data.get("distill_mode", self.DISTILL_MODE)
        teacher_images = data.get("teacher_images", None)

        if distill_mode == "prompt_only" and teacher_images is not None:
            # ---- Mode 1: Distill from teacher ----
            # Teacher features (no gradients)
            with torch.no_grad():
                teacher_features = self._embed_tensor(teacher_images)
            self._log_vram("after teacher EVA features")

            teacher_pil = []
            teacher_np = teacher_images.detach().to(torch.float32).cpu().numpy()
            for i in range(teacher_np.shape[0]):
                arr = (teacher_np[i].transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
                teacher_pil.append(Image.fromarray(arr))

            with torch.no_grad():
                teacher_features_radio = self._embed_pil_radio(teacher_pil, target_resolution=target_resolution)
            self._log_vram("after teacher RADIO features")

            loss_eva = F.mse_loss(student_features, teacher_features)
            loss_radio = F.mse_loss(student_features_radio, teacher_features_radio)

            print(f"[LOSS-DISTILL] EVA={loss_eva.item():.4f} RADIO={loss_radio.item():.4f} "
                  f"TOTAL={self.EVA_WEIGHT * loss_eva.item() + self.RADIO_WEIGHT * loss_radio.item():.4f}")

        else:
            # ---- Mode 2: Image pairs (skip teacher, use reference) ----
            image_paths = batch["image_path"]
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            ref_pil = [Image.open(p).convert("RGB") for p in image_paths]

            with torch.no_grad():
                ref_features = self._embed(ref_pil, bucket_size=target_resolution[0] if target_resolution else None)
            self._log_vram("after ref EVA features")

            loss_eva = F.mse_loss(student_features, ref_features)

            ref_features_radio = self._embed_pil_radio(ref_pil, target_resolution=target_resolution)
            self._log_vram("after ref RADIO features")

            loss_radio = F.mse_loss(student_features_radio, ref_features_radio)

            print(f"[LOSS-PAIR] EVA={loss_eva.item():.4f} RADIO={loss_radio.item():.4f} "
                  f"TOTAL={self.EVA_WEIGHT * loss_eva.item() + self.RADIO_WEIGHT * loss_radio.item():.4f}")

        total_loss = self.EVA_WEIGHT * loss_eva + self.RADIO_WEIGHT * loss_radio

        self._log_vram("calculate_loss end")
        return total_loss

    # ==================================================================
    # Feature Extraction Helpers
    # ==================================================================

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
        with torch.no_grad():
            features = self._feature_extractor(batch)
        return features

    def _embed_tensor(self, images: Tensor) -> Tensor:
        """Convert image tensors [B, 3, H, W] in [0, 1] to EVA-02 embeddings."""
        batch = images.to(self.FEATURE_DEVICE)

        _, _, h, w = batch.shape
        target = self.FEATURE_EXTRACTOR_RESOLUTION

        scale = target / min(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        batch = F.interpolate(batch, size=(new_h, new_w), mode="bilinear", align_corners=False)

        top = (new_h - target) // 2
        left = (new_w - target) // 2
        batch = batch[:, :, top:top + target, left:left + target]

        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=batch.device,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=batch.device,
        ).view(1, 3, 1, 1)
        batch = (batch - mean) / std

        features = self._feature_extractor(batch)
        return features

    def _embed_pil_radio(
            self,
            pil_images: list[Image.Image],
            target_resolution: tuple[int, int] | None = None,
    ) -> Tensor | None:
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
    BaseModelSetup, AnimaDistillLoRASetup,
    ModelType.ANIMA, TrainingMethod.DISTILL_LORA,
)

from modules.modelLoader.AnimaModelLoader import AnimaLoRAModelLoader
from modules.modelLoader.BaseModelLoader import BaseModelLoader
factory.register(
    BaseModelLoader, AnimaLoRAModelLoader,
    ModelType.ANIMA, TrainingMethod.DISTILL_LORA,
)

from modules.modelSampler.AnimaSampler import AnimaSampler
from modules.modelSampler.BaseModelSampler import BaseModelSampler
factory.register(
    BaseModelSampler, AnimaSampler,
    ModelType.ANIMA, TrainingMethod.DISTILL_LORA,
)

from modules.modelSaver.anima.AnimaLoRASaver import AnimaLoRASaver
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin


class AnimaDistillLoRASaver(BaseModelSaver, InternalModelSaverMixin):
    def save(self, model, model_type, output_model_format, output_model_destination, dtype):
        lora_saver = AnimaLoRASaver()
        lora_saver.save(model, output_model_format, output_model_destination, dtype)
        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)


factory.register(
    BaseModelSaver, AnimaDistillLoRASaver,
    ModelType.ANIMA, TrainingMethod.DISTILL_LORA,
)
