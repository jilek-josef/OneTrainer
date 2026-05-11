import os
import traceback

from modules.model.AnimaModel import AnimaModel, LLMAdapter
from modules.model.BaseModel import BaseModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch
from torch import nn

from diffusers import (
    AutoencoderKLWan,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import AutoTokenizer
from diffusers.loaders.single_file_utils import (
    convert_cosmos_transformer_checkpoint_to_diffusers,
    convert_wan_vae_to_diffusers,
)
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3Model, T5TokenizerFast

import accelerate


class AnimaModelLoader(HFModelLoaderMixin):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _wrap_vae_for_images(vae: AutoencoderKLWan):
        """
        Wan VAE expects 5D (B,C,T,H,W) video input but we train on 4D (B,C,H,W) images.
        Wrap encode so 4D images are automatically expanded to T=1 and the returned
        latent distribution yields 4D latents.
        """
        _orig_encode = vae.encode

        class _SqueezedDist:
            __slots__ = ("_dist",)
            def __init__(self, dist):
                self._dist = dist
            def sample(self, generator=None):
                return self._dist.sample(generator).squeeze(2)
            def mode(self):
                return self._dist.mode().squeeze(2)
            def __getattr__(self, name):
                return getattr(self._dist, name)

        def _encode(x, return_dict=True):
            squeeze = x.dim() == 4
            if squeeze:
                x = x.unsqueeze(2)
            out = _orig_encode(x, return_dict=return_dict)
            if squeeze:
                if return_dict:
                    out.latent_dist = _SqueezedDist(out.latent_dist)
                else:
                    out = (_SqueezedDist(out[0]),)
            return out

        vae.encode = _encode

    def __load_qwen_text_encoder(
            self,
            model: AnimaModel,
            weight_dtypes: ModelWeightDtypes,
            text_encoder_path: str,
    ):
        is_local_file = os.path.isfile(text_encoder_path)
        is_local_dir = os.path.isdir(text_encoder_path)

        if is_local_dir and os.path.isfile(os.path.join(text_encoder_path, "config.json")):
            # Load from a proper transformers directory
            return self._load_transformers_sub_module(
                Qwen3Model,
                weight_dtypes.text_encoder,
                weight_dtypes.fallback_train_dtype,
                text_encoder_path,
                "",
            )

        # Load config from HuggingFace and weights from local file
        config = Qwen3Model.config_class.from_pretrained("Qwen/Qwen3-0.6B")

        with accelerate.init_empty_weights():
            text_encoder = Qwen3Model(config)

        if is_local_file:
            state_dict = load_file(text_encoder_path)
        elif is_local_dir:
            # Try to find safetensors in directory
            safetensors_file = os.path.join(text_encoder_path, "model.safetensors")
            if not os.path.isfile(safetensors_file):
                safetensors_file = os.path.join(text_encoder_path, "qwen_3_06b_base.safetensors")
            state_dict = load_file(safetensors_file)
        else:
            raise ValueError(f"text_encoder path not found: {text_encoder_path}")

        # Remap keys: Qwen3ForCausalLM checkpoint has "model.*" prefix,
        # but Qwen3Model expects keys without the prefix
        remapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[len("model."):]
            else:
                new_key = key
            remapped_state_dict[new_key] = value

        text_encoder.load_state_dict(remapped_state_dict, strict=False, assign=True)
        del state_dict
        del remapped_state_dict

        return self._convert_transformers_sub_module_to_dtype(
            text_encoder, weight_dtypes.text_encoder, weight_dtypes.fallback_train_dtype
        )

    def __load_llm_adapter(
            self,
            model: AnimaModel,
            transformer_state_dict: dict,
    ):
        adapter = LLMAdapter(
            source_dim=1024,
            target_dim=1024,
            model_dim=1024,
            num_layers=6,
            num_heads=16,
            use_self_attn=True,
        )

        adapter_state_dict = {}
        prefix = "net.llm_adapter."
        for key, value in transformer_state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                # Map ComfyUI keys to our implementation keys
                new_key = new_key.replace("cross_attn.k_norm", "cross_attn.k_norm")
                new_key = new_key.replace("cross_attn.k_proj", "cross_attn.k_proj")
                new_key = new_key.replace("cross_attn.o_proj", "cross_attn.o_proj")
                new_key = new_key.replace("cross_attn.q_norm", "cross_attn.q_norm")
                new_key = new_key.replace("cross_attn.q_proj", "cross_attn.q_proj")
                new_key = new_key.replace("cross_attn.v_proj", "cross_attn.v_proj")
                new_key = new_key.replace("self_attn.k_norm", "self_attn.k_norm")
                new_key = new_key.replace("self_attn.k_proj", "self_attn.k_proj")
                new_key = new_key.replace("self_attn.o_proj", "self_attn.o_proj")
                new_key = new_key.replace("self_attn.q_norm", "self_attn.q_norm")
                new_key = new_key.replace("self_attn.q_proj", "self_attn.q_proj")
                new_key = new_key.replace("self_attn.v_proj", "self_attn.v_proj")
                new_key = new_key.replace("norm_cross_attn", "norm_cross_attn")
                new_key = new_key.replace("norm_mlp", "norm_mlp")
                new_key = new_key.replace("norm_self_attn", "norm_self_attn")
                new_key = new_key.replace("mlp.0", "mlp.0")
                new_key = new_key.replace("mlp.2", "mlp.2")
                adapter_state_dict[new_key] = value

        adapter.load_state_dict(adapter_state_dict, strict=False, assign=True)
        return adapter

    def __load_transformer_from_single_file(
            self,
            model: AnimaModel,
            weight_dtypes: ModelWeightDtypes,
            transformer_path: str,
            quantization: QuantizationConfig,
    ):
        state_dict = load_file(transformer_path)

        # Extract and load adapter weights
        adapter = self.__load_llm_adapter(model, state_dict)
        model.llm_adapter = adapter

        # Convert transformer keys to diffusers format
        converted_state_dict = convert_cosmos_transformer_checkpoint_to_diffusers(state_dict)

        # Remove adapter keys from converted state dict (they don't belong to transformer)
        converted_state_dict = {k: v for k, v in converted_state_dict.items() if not k.startswith("llm_adapter.")}

        transformer = CosmosTransformer3DModel(
            in_channels=16,
            out_channels=16,
            num_attention_heads=16,
            attention_head_dim=128,
            num_layers=28,
            mlp_ratio=4.0,
            text_embed_dim=1024,
            adaln_lora_dim=256,
            max_size=(128, 240, 240),
            patch_size=(1, 2, 2),
            rope_scale=(1.0, 4.0, 4.0),
            concat_padding_mask=True,
            extra_pos_embed_type=None,
        )

        missing, unexpected = transformer.load_state_dict(converted_state_dict, strict=False)
        if missing:
            print(f"Warning: Missing transformer keys: {missing}")
        if unexpected:
            print(f"Warning: Unexpected transformer keys: {unexpected}")

        del converted_state_dict
        del state_dict

        return self._convert_diffusers_sub_module_to_dtype(
            transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization
        )

    def __load_vae_from_single_file(
            self,
            model: AnimaModel,
            weight_dtypes: ModelWeightDtypes,
            vae_path: str,
    ):
        state_dict = load_file(vae_path)
        converted_state_dict = convert_wan_vae_to_diffusers(state_dict)

        vae = AutoencoderKLWan()
        missing, unexpected = vae.load_state_dict(converted_state_dict, strict=False)
        if missing:
            print(f"Warning: Missing VAE keys: {missing}")
        if unexpected:
            print(f"Warning: Unexpected VAE keys: {unexpected}")

        del converted_state_dict
        del state_dict

        self._wrap_vae_for_images(vae)

        return self._convert_diffusers_sub_module_to_dtype(
            vae, weight_dtypes.vae, weight_dtypes.train_dtype
        )

    def __load_internal(
            self,
            model: AnimaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            text_encoder_model_name: str,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name,
                text_encoder_model_name, quantization,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: AnimaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            text_encoder_model_name: str,
            quantization: QuantizationConfig,
    ):
        diffusers_sub = []
        transformers_sub = ["text_encoder"]
        if not transformer_model_name:
            diffusers_sub.append("transformer")
        if not vae_model_name:
            diffusers_sub.append("vae")

        self._prepare_sub_modules(
            base_model_name,
            diffusers_modules=diffusers_sub,
            transformers_modules=transformers_sub,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
            trust_remote_code=True,
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        # Load T5 tokenizer for adapter (prefer repo's own t5_tokenizer)
        tokenizer_t5 = None
        try:
            tokenizer_t5 = T5TokenizerFast.from_pretrained(base_model_name, subfolder="t5_tokenizer")
        except Exception:
            for t5_repo in [
                ("stabilityai/stable-diffusion-3.5-large", "tokenizer_3"),
                ("google-t5/t5-xxl", None),
                ("google-t5/t5-large", None),
                ("google-t5/t5-base", None),
                ("google-t5/t5-small", None),
            ]:
                try:
                    repo, subfolder = t5_repo
                    kwargs = {"subfolder": subfolder} if subfolder else {}
                    tokenizer_t5 = T5TokenizerFast.from_pretrained(repo, **kwargs)
                    break
                except Exception:
                    continue

        # The diffusers-format text encoder checkpoint stores weights with a
        # "model." prefix (from Qwen3ForCausalLM), but we load into Qwen3Model
        # which doesn't have that nested module. Set up key remapping.
        Qwen3Model._checkpoint_conversion_mapping["^model\\."] = ""

        if text_encoder_model_name:
            text_encoder = self.__load_qwen_text_encoder(model, weight_dtypes, text_encoder_model_name)
        else:
            text_encoder = self._load_transformers_sub_module(
                Qwen3Model,
                weight_dtypes.text_encoder,
                weight_dtypes.fallback_train_dtype,
                base_model_name,
                "text_encoder",
            )

        # Load LLM adapter from diffusers repo
        llm_adapter = None
        try:
            adapter_path = hf_hub_download(
                repo_id=base_model_name,
                subfolder="llm_adapter",
                filename="diffusion_pytorch_model.safetensors",
            )
            state_dict = load_file(adapter_path)

            llm_adapter = LLMAdapter(
                source_dim=1024,
                target_dim=1024,
                model_dim=1024,
                num_layers=6,
                num_heads=16,
                use_self_attn=True,
            )
            llm_adapter.load_state_dict(state_dict, strict=False, assign=True)
            del state_dict
        except Exception:
            pass  # Adapter is optional when loading individual components

        if vae_model_name:
            vae = self.__load_vae_from_single_file(model, weight_dtypes, vae_model_name)
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLWan,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        self._wrap_vae_for_images(vae)
        vae.enable_tiling()

        if transformer_model_name:
            transformer = self.__load_transformer_from_single_file(
                model, weight_dtypes, transformer_model_name, quantization
            )
        else:
            transformer = self._load_diffusers_sub_module(
                CosmosTransformer3DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.tokenizer_t5 = tokenizer_t5
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer
        model.llm_adapter = llm_adapter

    def __load_safetensors(
            self,
            model: AnimaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            text_encoder_model_name: str,
            quantization: QuantizationConfig,
    ):
        raise NotImplementedError("Loading of single file Anima models not supported. Use the diffusers model instead. Optionally, transformer/vae/text_encoder safetensor files can be loaded by overriding the components.")

    def load(
            self,
            model: AnimaModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.vae_model, model_names.text_encoder_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.vae_model, model_names.text_encoder_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.vae_model, model_names.text_encoder_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)


class AnimaLoRALoader(LoRALoaderMixin):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return None

    def load(
            self,
            model: AnimaModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


AnimaLoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.ANIMA: "resources/sd_model_spec/anima-lora.json",
    },
    model_class=AnimaModel,
    model_loader_class=AnimaModelLoader,
    lora_loader_class=AnimaLoRALoader,
    embedding_loader_class=None,
)

AnimaFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.ANIMA: "resources/sd_model_spec/anima.json",
    },
    model_class=AnimaModel,
    model_loader_class=AnimaModelLoader,
    embedding_loader_class=None,
)
