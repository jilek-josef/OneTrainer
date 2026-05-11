import copy
from collections.abc import Callable

from modules.model.AnimaModel import AnimaModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch

from tqdm import tqdm


class AnimaSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: AnimaModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            negative_prompt: str,
            height: int,
            width: int,
            seed: int,
            random_seed: bool,
            diffusion_steps: int,
            cfg_scale: float,
            noise_scheduler: NoiseScheduler,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            scheduler = copy.deepcopy(self.model.noise_scheduler)
            transformer = self.model.transformer
            vae = self.model.vae

            vae_scale_factor_temporal = 2 ** sum(vae.temperal_downsample)
            vae_scale_factor_spatial = 2 ** len(vae.temperal_downsample)
            num_latent_frames = (1 - 1) // vae_scale_factor_temporal + 1
            latent_height = height // vae_scale_factor_spatial
            latent_width = width // vae_scale_factor_spatial
            num_channels_latents = transformer.config.in_channels

            # Encode prompt
            self.model.text_encoder_to(self.train_device)
            try:
                prompt_embeds = self.model.encode_text(
                    text=[prompt],
                    batch_size=1,
                    train_device=self.train_device,
                )
                negative_prompt_embeds = self.model.encode_text(
                    text=[negative_prompt],
                    batch_size=1,
                    train_device=self.train_device,
                )
            finally:
                self.model.text_encoder_to(self.temp_device)
                torch_gc()

            # Prepare latents
            latents = torch.randn(
                (1, num_channels_latents, num_latent_frames, latent_height, latent_width),
                generator=generator,
                device=self.train_device,
                dtype=torch.float32,
            )

            # Prepare timesteps
            scheduler.set_timesteps(diffusion_steps, device=self.train_device)
            timesteps = scheduler.timesteps

            padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer.dtype)
            transformer_dtype = transformer.dtype
            do_cfg = cfg_scale > 1.0

            self.model.transformer_to(self.train_device)

            # Denoising loop — CONST preconditioning (flow matching)
            for i, t in enumerate(tqdm(timesteps, desc="sampling")):
                sigma = scheduler.sigmas[i]
                timestep = sigma.expand(latents.shape[0]).to(transformer_dtype)
                latent_model_input = latents.to(transformer_dtype)

                velocity_cond = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0].float()

                if do_cfg:
                    velocity_uncond = transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds.to(dtype=transformer_dtype),
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0].float()
                    velocity = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)
                else:
                    velocity = velocity_cond

                latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
                on_update_progress(i + 1, len(timesteps))

            self.model.transformer_to(self.temp_device)
            torch_gc()
            self.model.vae_to(self.train_device)

            # VAE decode
            z_dim = vae.config.z_dim
            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, z_dim, 1, 1, 1)
                .to(self.train_device, latents.dtype)
            )
            latents_std = (
                1.0 / torch.tensor(vae.config.latents_std)
                .view(1, z_dim, 1, 1, 1)
                .to(self.train_device, latents.dtype)
            )
            latents_for_decode = latents / latents_std + latents_mean
            video = vae.decode(latents_for_decode.to(vae.dtype), return_dict=False)[0]

            # Extract image from video (B, C, T, H, W) -> PIL
            image = video.squeeze(2)  # Remove temporal dim
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            from PIL import Image
            pil_image = Image.fromarray((image[0] * 255).astype("uint8"))

            self.model.vae_to(self.temp_device)
            torch_gc()

            return ModelSamplerOutput(
                file_type=FileType.IMAGE,
                data=pil_image,
            )

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=self.quantize_resolution(sample_config.height, 64),
            width=self.quantize_resolution(sample_config.width, 64),
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)


factory.register(BaseModelSampler, AnimaSampler, ModelType.ANIMA)
