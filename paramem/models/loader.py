"""Model-agnostic loader with QLoRA and multi-adapter support.

This module isolates all model-specific logic behind a clean interface.
Swapping the base model requires only changing the model_id in config.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from paramem.utils.config import AdapterConfig, DistillationConfig, ModelConfig

logger = logging.getLogger(__name__)


def _get_quantization_config(model_config: ModelConfig) -> Optional[BitsAndBytesConfig]:
    """Build quantization config from model settings."""
    if model_config.quantization == "none":
        return None

    compute_dtype = getattr(torch, model_config.compute_dtype)

    if model_config.quantization == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    if model_config.quantization == "fp4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    raise ValueError(f"Unknown quantization type: {model_config.quantization}")


def load_base_model(
    model_config: ModelConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a quantized base model and tokenizer.

    Returns the model in evaluation mode with frozen parameters.
    """
    logger.info("Loading base model: %s", model_config.model_id)

    quantization_config = _get_quantization_config(model_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=getattr(torch, model_config.compute_dtype),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_id,
        trust_remote_code=model_config.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logger.info(
        "Model loaded: %s (%.1fM params, quantization=%s)",
        model_config.model_id,
        sum(p.numel() for p in model.parameters()) / 1e6,
        model_config.quantization,
    )

    return model, tokenizer


def load_distillation_model(
    config: DistillationConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a distillation model (instruct-class) for QA generation.

    Supports CPU offload and memory caps for larger models like Gemma 2 9B.
    """
    logger.info("Loading distillation model: %s", config.model_id)

    compute_dtype = getattr(torch, config.compute_dtype)

    bnb_kwargs = {}
    if config.cpu_offload:
        bnb_kwargs["llm_int8_enable_fp32_cpu_offload"] = True

    if config.quantization == "nf4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            **bnb_kwargs,
        )
    else:
        quantization_config = None

    load_kwargs = {}
    if config.cpu_offload:
        load_kwargs["max_memory"] = {
            0: config.max_memory_gpu,
            "cpu": config.max_memory_cpu,
        }

    load_kwargs_full = {
        "device_map": "auto",
        "trust_remote_code": config.trust_remote_code,
        **load_kwargs,
    }
    if quantization_config is not None:
        # Compute dtype is already in the BnB config — don't pass torch_dtype
        # (redundant, and triggers Transformers 5.x deprecation + CUDA race)
        load_kwargs_full["quantization_config"] = quantization_config
    else:
        load_kwargs_full["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        **load_kwargs_full,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logger.info(
        "Distillation model loaded: %s (%.1fM params, quantization=%s)",
        config.model_id,
        sum(p.numel() for p in model.parameters()) / 1e6,
        config.quantization,
    )

    return model, tokenizer


def unload_model(model, tokenizer=None) -> None:
    """Free a model from GPU/CPU memory."""
    import gc

    del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Model unloaded, CUDA cache cleared")


def create_adapter(
    model: PreTrainedModel,
    adapter_config: AdapterConfig,
    adapter_name: str = "default",
) -> PeftModel:
    """Create a new LoRA adapter on the base model."""
    lora_config = LoraConfig(
        r=adapter_config.rank,
        lora_alpha=adapter_config.alpha,
        target_modules=adapter_config.target_modules,
        lora_dropout=adapter_config.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, lora_config, adapter_name=adapter_name)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "Adapter '%s' created: rank=%d, trainable=%.2fM / %.1fM total (%.2f%%)",
        adapter_name,
        adapter_config.rank,
        trainable_params / 1e6,
        total_params / 1e6,
        100 * trainable_params / total_params,
    )

    return peft_model


def load_adapter(
    model: PreTrainedModel,
    adapter_path: str | Path,
    adapter_name: str,
) -> PeftModel:
    """Load a saved LoRA adapter onto the base model."""
    adapter_path = str(adapter_path)

    if isinstance(model, PeftModel):
        model.load_adapter(adapter_path, adapter_name=adapter_name)
        return model

    return PeftModel.from_pretrained(
        model,
        adapter_path,
        adapter_name=adapter_name,
    )


def switch_adapter(model: PeftModel, adapter_name: str) -> None:
    """Switch the active adapter on a multi-adapter model."""
    model.set_adapter(adapter_name)
    logger.debug("Switched to adapter: %s", adapter_name)


def save_adapter(model: PeftModel, path: str | Path, adapter_name: str) -> None:
    """Save a specific adapter to disk."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path), selected_adapters=[adapter_name])
    logger.info("Adapter '%s' saved to %s", adapter_name, path)


def get_adapter_info(model: PeftModel) -> dict:
    """Return summary info about all loaded adapters."""
    info = {}
    for name in model.peft_config:
        config = model.peft_config[name]
        info[name] = {
            "rank": config.r,
            "alpha": config.lora_alpha,
            "target_modules": list(config.target_modules),
            "dropout": config.lora_dropout,
        }
    return info
