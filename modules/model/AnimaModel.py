import math
from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKLWan,
    CosmosTransformer3DModel,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen2Tokenizer, Qwen3Model, T5TokenizerFast

PROMPT_MAX_LENGTH = 512


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LLMAdapterAttention(torch.nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()

        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = torch.nn.RMSNorm(self.head_dim, eps=1e-6)

        self.k_proj = torch.nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = torch.nn.RMSNorm(self.head_dim, eps=1e-6)

        self.v_proj = torch.nn.Linear(context_dim, inner_dim, bias=False)

        self.o_proj = torch.nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            assert position_embeddings_context is not None
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states = apply_rotary_pos_emb(key_states, cos, sin)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=mask
        )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

    def init_weights(self):
        torch.nn.init.zeros_(self.o_proj.weight)


class LLMAdapterTransformerBlock(torch.nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, use_self_attn=True):
        super().__init__()
        self.use_self_attn = use_self_attn

        if self.use_self_attn:
            self.norm_self_attn = torch.nn.RMSNorm(model_dim, eps=1e-6)
            self.self_attn = LLMAdapterAttention(
                query_dim=model_dim,
                context_dim=model_dim,
                n_heads=num_heads,
                head_dim=model_dim // num_heads,
            )

        self.norm_cross_attn = torch.nn.RMSNorm(model_dim, eps=1e-6)
        self.cross_attn = LLMAdapterAttention(
            query_dim=model_dim,
            context_dim=source_dim,
            n_heads=num_heads,
            head_dim=model_dim // num_heads,
        )

        self.norm_mlp = torch.nn.RMSNorm(model_dim, eps=1e-6)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(model_dim, int(model_dim * mlp_ratio)),
            torch.nn.GELU(),
            torch.nn.Linear(int(model_dim * mlp_ratio), model_dim),
        )

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None, position_embeddings=None, position_embeddings_context=None):
        if self.use_self_attn:
            normed = self.norm_self_attn(x)
            attn_out = self.self_attn(normed, mask=target_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings)
            x = x + attn_out

        normed = self.norm_cross_attn(x)
        attn_out = self.cross_attn(normed, mask=source_attention_mask, context=context, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        x = x + attn_out

        x = x + self.mlp(self.norm_mlp(x))
        return x

    def init_weights(self):
        torch.nn.init.zeros_(self.mlp[2].weight)
        self.cross_attn.init_weights()


class LLMAdapter(torch.nn.Module):
    def __init__(
        self,
        source_dim=1024,
        target_dim=1024,
        model_dim=1024,
        num_layers=6,
        num_heads=16,
        use_self_attn=True,
    ):
        super().__init__()

        self.embed = torch.nn.Embedding(32128, target_dim)
        if model_dim != target_dim:
            self.in_proj = torch.nn.Linear(target_dim, model_dim)
        else:
            self.in_proj = torch.nn.Identity()
        self.rotary_emb = RotaryEmbedding(model_dim // num_heads)
        self.blocks = torch.nn.ModuleList([
            LLMAdapterTransformerBlock(source_dim, model_dim, num_heads=num_heads, use_self_attn=use_self_attn)
            for _ in range(num_layers)
        ])
        self.out_proj = torch.nn.Linear(model_dim, target_dim)
        self.norm = torch.nn.RMSNorm(target_dim, eps=1e-6)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)

        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        context = source_hidden_states
        x = self.in_proj(self.embed(target_input_ids).to(dtype=context.dtype))
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(x, context, target_attention_mask=target_attention_mask, source_attention_mask=source_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        return self.norm(self.out_proj(x))


class AnimaModel(BaseModel):
    tokenizer: Qwen2Tokenizer | None
    tokenizer_t5: T5TokenizerFast | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Qwen3Model | None
    llm_adapter: LLMAdapter | None
    vae: AutoencoderKLWan | None
    transformer: CosmosTransformer3DModel | None

    text_encoder_autocast_context: torch.autocast | nullcontext
    text_encoder_train_dtype: DataType

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    text_encoder_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(self, model_type: ModelType):
        super().__init__(model_type=model_type)

        self.tokenizer = None
        self.tokenizer_t5 = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.llm_adapter = None
        self.vae = None
        self.transformer = None

        self.text_encoder_autocast_context = nullcontext()
        self.text_encoder_train_dtype = DataType.FLOAT_32

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.transformer_lora,
        ] if a is not None]

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            if self.text_encoder_offload_conductor is not None and \
                    self.text_encoder_offload_conductor.layer_offload_activated():
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)

        if self.llm_adapter is not None:
            self.llm_adapter.to(device=device)

    def transformer_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None and \
                self.transformer_offload_conductor.layer_offload_activated():
            self.transformer_offload_conductor.to(device)
        else:
            self.transformer.to(device=device)

        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)

    def eval(self):
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        if self.llm_adapter is not None:
            self.llm_adapter.eval()
        self.transformer.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        from diffusers import Cosmos2TextToImagePipeline
        return Cosmos2TextToImagePipeline(
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        )

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] = None,
            tokens: Tensor = None,
            tokens_mask: Tensor = None,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
    ) -> Tensor:
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            # Replace empty/blank strings to avoid 0-length tokenization crashes
            text = [t if t and t.strip() else " " for t in text]

            tokenizer_output = self.tokenizer(
                text,
                max_length=PROMPT_MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)
            tokens_mask = tokenizer_output.attention_mask.to(self.text_encoder.device)

        if text_encoder_output is None and self.text_encoder is not None:
            with self.text_encoder_autocast_context:
                text_encoder_output = self.text_encoder(
                    tokens,
                    attention_mask=tokens_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                text_encoder_output = text_encoder_output.last_hidden_state

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError

        # Run LLM adapter with T5 token IDs (no attention masks — matches reference pipeline)
        if self.llm_adapter is not None and self.tokenizer_t5 is not None:
            # Ensure adapter is on the same device as the text encoder output
            adapter_device = next(self.llm_adapter.parameters()).device
            text_encoder_device = text_encoder_output.device
            if adapter_device != text_encoder_device:
                self.llm_adapter.to(text_encoder_device)

            if text is not None:
                t5_tokens = self.tokenizer_t5(
                    text,
                    max_length=PROMPT_MAX_LENGTH,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                t5_input_ids = t5_tokens.input_ids.to(text_encoder_device)
            elif tokens is not None and self.tokenizer is not None:
                # Decode Qwen tokens back to text, then tokenize with T5
                # (Qwen vocab is ~150K, T5 embed only has 32K slots — can't reuse Qwen IDs)
                text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
                t5_tokens = self.tokenizer_t5(
                    text,
                    max_length=PROMPT_MAX_LENGTH,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                t5_input_ids = t5_tokens.input_ids.to(text_encoder_device)
            else:
                # No text or tokens available — skip LLM adapter
                t5_input_ids = None

            if t5_input_ids is not None:
                with self.text_encoder_autocast_context:
                    text_encoder_output = self.llm_adapter(
                        text_encoder_output,
                        t5_input_ids,
                    )

            # Pad to 512 sequence length if shorter (matches reference pipeline)
            if text_encoder_output.shape[1] < PROMPT_MAX_LENGTH:
                text_encoder_output = torch.nn.functional.pad(
                    text_encoder_output, (0, 0, 0, PROMPT_MAX_LENGTH - text_encoder_output.shape[1])
                )

        return text_encoder_output

    def scale_latents(self, latents: Tensor) -> Tensor:
        # Support both 4D (B, C, H, W) image latents and 5D (B, C, T, H, W) video latents
        num_dims = latents.ndim
        shape = [1, -1] + [1] * (num_dims - 2)
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(*shape).to(latents.device, latents.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std).view(*shape).to(latents.device, latents.dtype)
        return (latents - latents_mean) / latents_std

    def unscale_latents(self, latents: Tensor) -> Tensor:
        # Support both 4D (B, C, H, W) image latents and 5D (B, C, T, H, W) video latents
        num_dims = latents.ndim
        shape = [1, -1] + [1] * (num_dims - 2)
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(*shape).to(latents.device, latents.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std).view(*shape).to(latents.device, latents.dtype)
        return latents * latents_std + latents_mean

    def calculate_timestep_shift(self, latent_width: int, latent_height: int):
        # Anima uses fixed shift=3.0 from its FlowMatchEulerDiscreteScheduler
        return getattr(self.noise_scheduler.config, 'shift', 3.0)
