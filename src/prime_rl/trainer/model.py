import logging
import time
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from beartype import beartype as typechecker
from huggingface_hub import snapshot_download
from jaxtyping import Float, Int, jaxtyped
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.import_utils import is_flash_attn_3_available

from prime_rl.trainer.config import ActivationCheckpointConfig, CompileConfig, ModelConfig
from prime_rl.trainer.lora import apply_lora_to_model
from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.weights import (
    convert_hf_to_tt_moe,
    convert_tt_to_hf_moe,
    has_hf_moe_layers,
    has_tt_moe_layers,
    load_state_dict,
    save_state_dict,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.tensor_hashing import get_module_signature

# Add filter to the standard logging module for transformers.modeling_utils to supress the
# flash attention dtype warnings since FSDP is used to handle mixed precision.
transformers_modeling_utils_logger = logging.getLogger("transformers.modeling_utils")
transformers_modeling_utils_logger.addFilter(
    lambda record: "Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes" not in record.getMessage()
)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def is_tt_moe_model(model: nn.Module) -> bool:
    return hasattr(model.config, "num_experts") or hasattr(model.config, "n_routed_experts")


def get_load_balance_stats(model: nn.Module, reset_stats: bool = True) -> dict[str, Tensor | None]:
    per_layer_max_vio = []
    for transformer_block in model.model.layers:
        # This is necessary for models that have mixed dense layers
        if not hasattr(transformer_block.mlp, "tokens_per_expert"):
            continue
        tokens_per_expert = transformer_block.mlp.tokens_per_expert
        balanced_load = tokens_per_expert.mean()
        max_vio = (tokens_per_expert.max() - balanced_load) / balanced_load
        per_layer_max_vio.append(max_vio.item())
        if reset_stats:
            tokens_per_expert.zero_()
    if len(per_layer_max_vio) == 0:
        return {"max_vio": None}
    return {"max_vio": torch.tensor(per_layer_max_vio, device=torch.device("cuda"))}


def get_model(
    config: ModelConfig, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.bfloat16
) -> nn.Module:
    logger = get_logger()
    logger.info(
        f"Loading model config (name={config.name}, attn={config.attn}, trust_remote_code={config.trust_remote_code})"
    )
    model_config = cast(
        PretrainedConfig,
        AutoConfig.from_pretrained(
            config.name, attn_implementation=config.attn, trust_remote_code=config.trust_remote_code
        ),
    )
    model_config.use_cache = False
    model_config.use_grouped_mm = config.moe_use_grouped_mm
    logger.debug(f"Loaded model config ({model_config.to_dict()})")

    if config.debug.num_layers is not None:
        num_hidden_layers = min(config.debug.num_layers, model_config.num_hidden_layers)
        logger.warning(
            f"Setting the number of layers to {config.debug.num_layers} in the model config. This means {model_config.num_hidden_layers - num_hidden_layers} layers will not be loaded."
        )
        model_config.num_hidden_layers = num_hidden_layers

    with device:
        match config.impl:
            case "hf":
                # Detect if this is a vision-language model
                model_type = model_config.model_type if hasattr(model_config, "model_type") else None
                config_class_name = type(model_config).__name__.lower()
                is_vlm = (
                    (model_type and ("vl" in model_type.lower() or "vision" in model_type.lower()))
                    or "qwen2_vl" in config_class_name
                    or "vision" in config_class_name
                )
                
                if is_vlm:
                    model_cls = AutoModelForImageTextToText
                    logger.info(f"Detected VLM model (type={model_type}, config={type(model_config).__name__}), using AutoModelForImageTextToText")
                else:
                    model_cls = AutoModelForCausalLM
            case "liger_kernel":
                model_cls = AutoLigerKernelForCausalLM
            case "custom":
                model_cls = AutoModelForCausalLMPrimeRL

        load_model_start_time = time.perf_counter()
        if device == torch.device("meta"):
            logger.info(f"Loading model {config.name} using {model_cls.__name__} to meta device")
            model = model_cls.from_config(model_config, trust_remote_code=config.trust_remote_code, dtype=dtype)
        else:
            logger.info(f"Loading model {config.name} using {model_cls.__name__} to CPU")
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path=config.name,
                config=model_config,
                trust_remote_code=config.trust_remote_code,
                dtype=dtype,
            )
        logger.debug(f"Loaded model {config.name} in {time.perf_counter() - load_model_start_time:.2f} seconds")

    # Check LM head dtype (VLMs might use different attribute names)
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        assert model.lm_head.weight.dtype == dtype, (
            f"LM head dtype wasnt loaded correctly {model.lm_head.weight.dtype} != {dtype}"
        )
    elif hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        # Some VLMs have the language model nested
        if hasattr(model.language_model.lm_head, "weight"):
            assert model.language_model.lm_head.weight.dtype == dtype, (
                f"LM head dtype wasnt loaded correctly {model.language_model.lm_head.weight.dtype} != {dtype}"
            )
    else:
        logger.warning(f"Could not find lm_head attribute on model {config.name}, skipping dtype check")
    
    # Freeze vision encoder if requested (for VLMs)
    if config.freeze_vision_encoder:
        frozen_params = 0
        vision_modules = []
        
        # Common vision encoder paths across different VLM architectures
        # Format: (path, attribute_chain) - we try both direct and nested paths
        vision_paths = [
            # Direct attributes (e.g., LLaVA-style)
            ("visual", ["visual"]),
            ("vision_model", ["vision_model"]),
            ("vision_tower", ["vision_tower"]),
            ("image_encoder", ["image_encoder"]),
            ("vit", ["vit"]),
            # Nested under model (e.g., Qwen-VL style: model.model.visual)
            ("model.visual", ["model", "visual"]),
            ("model.vision_model", ["model", "vision_model"]),
        ]
        
        for path_name, attr_chain in vision_paths:
            obj = model
            found = True
            for attr in attr_chain:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    found = False
                    break
            if found and obj is not model:
                vision_modules.append((path_name, obj))
                break  # Only freeze one vision encoder
        
        if vision_modules:
            for module_name, module in vision_modules:
                for param in module.parameters():
                    if param.requires_grad:
                        param.requires_grad = False
                        frozen_params += param.numel()
            logger.info(f"Froze vision encoder '{vision_modules[0][0]}': {frozen_params:,} parameters frozen")
        else:
            logger.warning(f"freeze_vision_encoder=True but no vision encoder found. Searched paths: {[p[0] for p in vision_paths]}")
    
    return model


def setup_tokenizer(config: ModelConfig) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=config.trust_remote_code)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_fsdp(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims):
    """
    Setup FSDP for model. Supports both standard transformers (model.model.layers)
    and VLMs (model.language_model.layers or model.model.language_model.layers).
    """
    logger = get_logger()
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=DTYPE_MAP[config.reduce_dtype])
    # Always use 2D mesh format for consistency (dp_replicate dimension always present)
    hsdp_mesh = parallel_dims.world_mesh["dp_replicate", "dp_shard_cp"]
    offload_policy: OffloadPolicy = CPUOffloadPolicy(pin_memory=True) if config.fsdp_cpu_offload else OffloadPolicy()

    # Detect model structure: standard transformer vs VLM
    layers = None
    embed_tokens = None
    norm = None
    lm_head = None
    is_vlm = False
    
    # Check for standard transformer structure (model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        embed_tokens = getattr(model.model, "embed_tokens", None)
        norm = getattr(model.model, "norm", None)
        lm_head = getattr(model, "lm_head", None)
        logger.info("Detected standard transformer structure (model.model.layers)")
    # Check for VLM structure: model.language_model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
        embed_tokens = getattr(model.language_model, "embed_tokens", None)
        norm = getattr(model.language_model, "norm", None)
        lm_head = getattr(model, "lm_head", None) or getattr(model.language_model, "lm_head", None)
        is_vlm = True
        logger.info("Detected VLM structure (model.language_model.layers)")
    # Check for VLM structure: model.model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        layers = model.model.language_model.layers
        embed_tokens = getattr(model.model.language_model, "embed_tokens", None)
        norm = getattr(model.model.language_model, "norm", None)
        lm_head = getattr(model, "lm_head", None) or getattr(model.model.language_model, "lm_head", None)
        is_vlm = True
        logger.info("Detected VLM structure (model.model.language_model.layers)")
    else:
        # Unknown structure - skip FSDP
        logger.warning(f"Skipping FSDP setup for model with unknown structure (type: {type(model).__name__})")
        device = torch.device(f"cuda:{get_world().local_rank}")
        if config.attn in ("flash_attention_2", "flash_attention_3"):
            logger.info(f"Converting model to bfloat16 and moving to {device} for FlashAttention compatibility")
            model.to(dtype=torch.bfloat16, device=device)
        else:
            logger.info(f"Moving model to {device}")
            model.to(device)
        return

    # Wrap transformer layers
    for transformer_block in layers:
        fully_shard(
            transformer_block,
            mesh=hsdp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=config.reshard_after_forward,
        )

    # Wrap embeddings and norm if available and not tied
    # For standard transformers, use original logic exactly
    if not is_vlm:
        # Standard transformer path - maintain exact backward compatibility
        if hasattr(model, "config") and not model.config.tie_word_embeddings:
            # This optimization breaks weight tying
            fully_shard(
                model.model.embed_tokens,
                mesh=hsdp_mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                reshard_after_forward=config.reshard_after_forward,
            )
            fully_shard(
                [model.lm_head, model.model.norm],
                mesh=hsdp_mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                reshard_after_forward=False,
            )
        else:
            logger.warning("Model is tied word embeddings, so not doing the last layer not resharding optimization")
    else:
        # VLM path - wrap language_model components
        # Check for tie_word_embeddings if config is available
        tie_embeddings = hasattr(model, "config") and getattr(model.config, "tie_word_embeddings", False)
        if not tie_embeddings:
            if embed_tokens is not None:
                fully_shard(
                    embed_tokens,
                    mesh=hsdp_mesh,
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                    reshard_after_forward=config.reshard_after_forward,
                )
            if norm is not None and lm_head is not None:
                fully_shard(
                    [lm_head, norm],
                    mesh=hsdp_mesh,
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                    reshard_after_forward=False,
                )
        else:
            logger.warning("VLM model has tied word embeddings, so not doing the last layer not resharding optimization")

    # Wrap the entire model
    fully_shard(
        model,
        mesh=hsdp_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=config.reshard_after_forward,
    )


def load_dcp_from_hf(model: nn.Module, config: ModelConfig):
    model.to_empty(device="cuda")
    torch.distributed.barrier()

    logger = get_logger()
    if config.debug.random_init:
        logger.warning("Randomly initializing model. Skipping loading weights from HF.")
        return

    if not Path(config.name).exists():
        snapshot_path = Path(snapshot_download(repo_id=config.name, repo_type="model"))
    else:
        logger.info(
            f"Loading model weights from path {config.name}, skipping snapshot download. If this is not expected, please remove the directory {config.name} and run again"
        )
        snapshot_path = Path(config.name)

    # Load the snapshot state
    snapshot_state_dict = load_state_dict(snapshot_path)
    model_state_dict = model.state_dict()

    # Dynamically convert between different weight formats if needed
    if has_hf_moe_layers(snapshot_state_dict) and has_tt_moe_layers(model_state_dict):
        logger.warning(
            "Found HF weight format in snapshot state dict and TT weight format in model state dict. Trying to auto-convert..."
        )
        snapshot_path = snapshot_path / "tt"
        if snapshot_path.exists():
            logger.debug(f"Conversion found at {snapshot_path}.")
        else:
            if get_world().is_master:
                logger.debug(
                    f"Converting snapshot state dict to TT format and saving to {snapshot_path} on master rank. This is a one-time operation."
                )
                convert_hf_to_tt_moe(snapshot_state_dict)
                save_state_dict(snapshot_state_dict, snapshot_path)

    elif has_tt_moe_layers(snapshot_state_dict) and has_hf_moe_layers(model_state_dict):
        logger.warning(
            "Found TT weight format in snapshot state dict and HF weight format in model state dict. Trying to auto-convert..."
        )
        snapshot_path = snapshot_path / "hf"
        if snapshot_path.exists():
            logger.debug(f"Conversion found at {snapshot_path}.")
        else:
            if get_world().is_master:
                logger.debug(
                    f"Converting snapshot state dict to HF format and saving to {snapshot_path} on master rank. This is a one-time operation."
                )
                convert_tt_to_hf_moe(snapshot_state_dict)
                save_state_dict(snapshot_state_dict, snapshot_path)

    # All ranks wait for master rank to finish conversion
    torch.distributed.barrier()

    logger.info(f"Loading weights using HF DCP from {snapshot_path}")
    load_dcp_start_time = time.perf_counter()
    dcp_load(
        model.state_dict(),
        storage_reader=HuggingFaceStorageReader(path=snapshot_path.as_posix()),
        # Note: This allow is needed by weight tying but could cause silent issues
        # planner=DefaultLoadPlanner(allow_partial_load=True),
    )
    fix_model_post_empty(model)
    logger.debug(f"Loaded weights using HF DCP in {time.perf_counter() - load_dcp_start_time:.2f} seconds")


def can_load_dcp_from_hf(model: nn.Module):
    """Whether the model will be loaded correctly by load_dcp_from_hf.

    The main issue is with anything that is not in the checkpoint.
    This is usually any non-persistent buffers.
    """
    buffer_names = [name for name, _ in model.named_buffers()]

    # TT MoE buffers
    buffer_names = [
        name
        for name in buffer_names
        if not (name.startswith("model.layers.") and name.endswith("mlp.tokens_per_expert"))
    ]
    buffer_names = [
        name for name in buffer_names if not (name.startswith("model.layers.") and name.endswith("mlp.expert_bias"))
    ]
    # HF standard transformer model
    if len(buffer_names) == 1 and buffer_names[0] == "model.rotary_emb.inv_freq":
        return True

    get_logger().warning(f"Model cannot be loaded using meta device because of buffers: {buffer_names}")
    return False


def fix_model_post_empty(model: nn.Module):
    buffer_names = [name for name, _ in model.named_buffers()]
    # HF standard transformer model
    if "model.rotary_emb.inv_freq" in buffer_names:
        rotary_emb = model.model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, rotary_emb.inv_freq.device)
        rotary_emb.inv_freq.copy_(inv_freq)

    # TODO: Init TT MoE buffers
    # I think .to_empty() on gpu by default fills 0 so we are ok but this might not be guaranteed behavior


def reshard_module(model: nn.Module):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def apply_ac(model: nn.Module, ac_config: ActivationCheckpointConfig):
    """
    Apply activation checkpointing. Supports both standard transformers and VLMs.
    """
    logger = get_logger()
    layers = None
    
    # Detect model structure
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        layers_attr = "model.model.layers"
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
        layers_attr = "language_model.layers"
    elif hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        layers = model.model.language_model.layers
        layers_attr = "model.model.language_model.layers"
    else:
        raise ValueError(f"Cannot apply activation checkpointing: model structure not recognized (type: {type(model).__name__})")
    
    for layer_id, (layer_name, transformer_block) in enumerate(layers.named_children()):
        if layer_id % ac_config.freq == 0:
            transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        layers.register_module(layer_name, transformer_block)
    logger.info(f"Applied activation checkpointing to {layers_attr} (freq={ac_config.freq})")


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    """
    Apply torch.compile to model layers. Supports both standard transformers and VLMs.
    """
    logger = get_logger()
    torch._dynamo.config.capture_scalar_outputs = True
    layers = None
    layers_attr = None
    
    # Detect model structure
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        layers_attr = "model.model.layers"
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
        layers_attr = "language_model.layers"
    elif hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        layers = model.model.language_model.layers
        layers_attr = "model.model.language_model.layers"
    else:
        raise ValueError(f"Cannot apply compilation: model structure not recognized (type: {type(model).__name__})")
    
    for layer_id in range(len(layers)):
        # Doing it in-place avoids mangled fqn which can break checkpoint loading
        layers[layer_id].compile(fullgraph=compile_config.fullgraph)
    logger.info(f"Compiled {len(layers)} layers from {layers_attr} (fullgraph={compile_config.fullgraph})")


def setup_model(config: ModelConfig, parallel_dims: ParallelDims) -> nn.Module:
    if config.attn == "flash_attention_3" and not is_flash_attn_3_available():
        raise ValueError(
            "Flash attention 3 is only supported if the flash_attn_3 package is installed. Install with `uv pip install 'flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@main#subdirectory=hopper' --no-build-isolation`"
        )

    logger = get_logger()
    # Get model from specified device
    model = get_model(
        config,
        device=torch.device("meta" if config.load_using_meta else "cpu"),
        dtype=DTYPE_MAP[config.optimization_dtype],
    )

    # Reload the model to CPU if we cannot load from
    if config.load_using_meta and not can_load_dcp_from_hf(model):
        logger.warning("Cannot load model from meta device. Loading model to CPU instead.")
        model = get_model(config, device=torch.device("cpu"), dtype=DTYPE_MAP[config.optimization_dtype])

    # Apply LoRA before FSDP setup
    if config.experimental.lora is not None:
        apply_lora_to_model(model, config.experimental.lora)

    # the right order is AC -> Compile -> FSDP
    if config.ac is not None:
        apply_ac(model, config.ac)
    if config.compile is not None:
        apply_compile(model, config.compile)

    setup_fsdp(model, config, parallel_dims)

    if config.load_using_meta and can_load_dcp_from_hf(model):
        load_dcp_from_hf(model, config)

    logger.debug(f"Model signature: {get_module_signature(model, compress=True)}")
    return model


@jaxtyped(typechecker=typechecker)
def forward(
    model: nn.Module,
    input_ids: Int[Tensor, "batch seq"],
    position_ids: Int[Tensor, "batch seq"],
    pixel_values: Tensor | None = None,
    image_grid_thw: Tensor | None = None,
    **extra_kwargs,
) -> Float[Tensor, "batch seq vocab"]:
    """Forward pass for both text-only and multimodal models.
    
    Args:
        model: The model to run forward pass on
        input_ids: Input token IDs
        position_ids: Position IDs
        pixel_values: Optional pixel values for multimodal models (e.g., Qwen-VL)
        image_grid_thw: Optional image grid metadata for Qwen2VL models
        **extra_kwargs: Additional model-specific kwargs
    
    Returns:
        Logits tensor
    """
    model_kwargs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
    }
    
    # Add multimodal inputs if present
    if pixel_values is not None:
        model_kwargs["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        model_kwargs["image_grid_thw"] = image_grid_thw
    
    # Add any extra model-specific kwargs
    model_kwargs.update(extra_kwargs)
    
    return model(**model_kwargs).logits
