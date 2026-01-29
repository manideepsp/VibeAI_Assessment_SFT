from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils.other import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from config import AppConfig


@dataclass(frozen=True)
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


def infer_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype_from_string(name: str) -> torch.dtype:
    n = (name or "").strip().lower()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp16", "float16"}:
        return torch.float16
    if n in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {name}")


def build_bnb_config(cfg: AppConfig) -> BitsAndBytesConfig:
    if not cfg.quant.load_in_4bit:
        raise ValueError("build_bnb_config requires quant.load_in_4bit=True")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit quantization (bitsandbytes) typically requires CUDA. "
            "Either run on a CUDA machine or set quant.load_in_4bit=false for CPU-only smoke tests."
        )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quant.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.quant.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_dtype_from_string(cfg.quant.bnb_4bit_compute_dtype),
    )


def load_tokenizer(cfg: AppConfig) -> PreTrainedTokenizerBase:
    tok_id = cfg.model.tokenizer_id or cfg.model.base_model_id
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    except Exception:
        # Fallback for constrained environments where the fast tokenizer files
        # (e.g., tokenizer.json) may fail to download or load.
        tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=False)

    # Many instruct models expect a pad token; if absent, fall back to eos.
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optionally cap the effective sequence length for memory reasons.
    if cfg.model.max_seq_len is not None and cfg.model.max_seq_len > 0:
        try:
            tokenizer.model_max_length = int(cfg.model.max_seq_len)
        except Exception:
            pass

    return tokenizer


def default_lora_target_modules(model: PreTrainedModel) -> list[str]:
    """Conservative target module names for common decoder-only architectures."""

    candidates = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    present: set[str] = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates:
            present.add(leaf)

    return sorted(present) if present else candidates


def build_lora_config(cfg: AppConfig, model: PreTrainedModel) -> LoraConfig:
    target_modules = cfg.lora.target_modules or default_lora_target_modules(model)

    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def load_base_model(cfg: AppConfig, device: Optional[torch.device] = None) -> PreTrainedModel:
    device = device or infer_device()

    quant_config = build_bnb_config(cfg) if cfg.quant.load_in_4bit else None

    # Keep device handling explicit for Windows/local dev.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_id,
        quantization_config=quant_config,
        torch_dtype=None,
        trust_remote_code=False,
    )

    # Ensure hidden states are available for auxiliary heads.
    model.config.output_hidden_states = True
    model.config.use_cache = False

    if cfg.train.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.to(device)
    return model


def attach_qlora_adapters(cfg: AppConfig, model: PreTrainedModel) -> PeftModel:
    """Attach LoRA adapters.

    If quant.load_in_4bit=True, prepares the k-bit model for QLoRA.
    If quant.load_in_4bit=False, attaches standard LoRA (useful for CPU smoke runs).
    """

    if cfg.quant.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.train.gradient_checkpointing)

    lora_cfg = build_lora_config(cfg, model)
    return get_peft_model(model, lora_cfg)


def load_model_and_tokenizer(cfg: AppConfig, device: Optional[torch.device] = None) -> LoadedModel:
    device = device or infer_device()

    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg, device=device)

    # Attach adapters last so they land on the same device.
    model = attach_qlora_adapters(cfg, model)

    # Align padding token id with tokenizer.
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return LoadedModel(model=model, tokenizer=tokenizer)


def save_adapter(model: PeftModel, output_dir: str | Path) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))


def load_adapter(base_model: PreTrainedModel, adapter_dir: str | Path) -> PeftModel:
    return PeftModel.from_pretrained(base_model, str(adapter_dir))


def merge_adapters_and_unload(model: PeftModel) -> PreTrainedModel:
    """Merge LoRA weights into the base model (useful for inference export)."""

    return model.merge_and_unload()
