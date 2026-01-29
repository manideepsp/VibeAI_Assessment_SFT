from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Optional

from label_policy import LabelPolicy


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Shallow+recursive dict merge (override wins).

    This is used for JSON config overrides.
    """

    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _filter_kwargs(cls: type[Any], data: dict[str, Any]) -> dict[str, Any]:
    """Filter a dict down to keys that are valid dataclass fields."""

    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


def _app_config_from_dict(data: dict[str, Any]) -> "AppConfig":
    """Rehydrate nested dataclasses explicitly.

    We do not rely on runtime type introspection of annotations (which can be forward refs),
    so this stays stable across Python versions.
    """

    return AppConfig(
        paths=PathsConfig(**_filter_kwargs(PathsConfig, data.get("paths", {}))),
        datasets=DatasetConfig(**_filter_kwargs(DatasetConfig, data.get("datasets", {}))),
        model=ModelConfig(**_filter_kwargs(ModelConfig, data.get("model", {}))),
        quant=QuantizationConfig(**_filter_kwargs(QuantizationConfig, data.get("quant", {}))),
        lora=LoRAConfig(**_filter_kwargs(LoRAConfig, data.get("lora", {}))),
        train=TrainConfig(**_filter_kwargs(TrainConfig, data.get("train", {}))),
        losses=LossWeights(**_filter_kwargs(LossWeights, data.get("losses", {}))),
        safety=SafetyTeacherConfig(**_filter_kwargs(SafetyTeacherConfig, data.get("safety", {}))),
        dpo=DPOConfig(**_filter_kwargs(DPOConfig, data.get("dpo", {}))),
        generation=GenerationConfig(**_filter_kwargs(GenerationConfig, data.get("generation", {}))),
        eval=EvalConfig(**_filter_kwargs(EvalConfig, data.get("eval", {}))),
        label_policy=LabelPolicy(**_filter_kwargs(LabelPolicy, data.get("label_policy", {}))),
    )


@dataclass(frozen=True)
class PathsConfig:
    project_root: str = field(default_factory=lambda: str(_repo_root()))
    artifacts_dir: str = field(default_factory=lambda: str(_repo_root() / "artifacts"))
    hf_cache_dir: str = field(default_factory=lambda: str(_repo_root() / ".hf_cache"))

    def ensure_dirs(self) -> None:
        Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)
        Path(self.hf_cache_dir).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetConfig:
    empathetic_dialogues_id: str = "Adapting/empathetic_dialogues_v2"
    esconv_id: str = "thu-coai/esconv"
    go_emotions_id: str = "google-research-datasets/go_emotions"

    mixture_alpha: float = 0.5
    max_train_examples: Optional[int] = None
    max_eval_examples: Optional[int] = 256


@dataclass(frozen=True)
class ModelConfig:
    # Small open-weight instruction model (≈3B class).
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer_id: Optional[str] = None

    # Let base model defaults apply unless overridden.
    max_seq_len: Optional[int] = None


@dataclass(frozen=True)
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    # Keep dtype as string here; model.py will map it to torch.dtype.
    bnb_4bit_compute_dtype: str = "bfloat16"  # or "float16" on older GPUs


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05

    # If empty, model.py will auto-detect a reasonable set of target modules.
    target_modules: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 7

    # Informational only for mixture/streaming training. Do not use as a stopping condition.
    num_train_epochs: float = 1.0
    # Step-based control is the primary training budget for mixture/streaming training.
    max_train_steps: int = 1000
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16

    # If set, training.py will write checkpoints here; otherwise uses paths.artifacts_dir / "checkpoints".
    output_dir: Optional[str] = None

    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True

    logging_steps: int = 10
    eval_steps: int = 200
    # Optional aliases; if set, they override logging_steps/eval_steps.
    log_every_steps: Optional[int] = None
    eval_every_steps: Optional[int] = None
    save_steps: int = 200
    save_total_limit: int = 2


@dataclass(frozen=True)
class LossWeights:
    lambda_lm: float = 1.0
    lambda_emo: float = 0.2
    lambda_strat: float = 0.2
    lambda_val: float = 0.0
    lambda_safe: float = 0.0


@dataclass(frozen=True)
class SafetyTeacherConfig:
    enabled: bool = False
    teacher_model_id: str = ""
    temperature_tau: float = 2.0

    # Compute budget: apply teacher KL on only a subset of batches.
    teacher_subset_ratio: float = 0.05


@dataclass(frozen=True)
class DPOConfig:
    enabled: bool = False
    beta: float = 0.1
    lambda_violation: float = 0.0


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 192
    temperature: float = 0.7
    top_p: float = 0.9

    # Optional style tokens (model-specific). Keep empty by default.
    style_tokens: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalConfig:
    # If EQ-Bench isn’t available or compute is limited, evaluation.py will fall back to a stub runner.
    eqbench_enabled: bool = True
    # Dataset ID and split for EQ-Bench prompt export.
    # Default may change over time on the Hub, so keep it configurable.
    # Note: the official EQ-Bench repo IDs may be gated/inaccessible; this default is
    # a public, accessible mirror used for prompt export/comparisons.
    eqbench_dataset_id: str = "pbevan11/EQ-Bench"
    eqbench_split: str = "validation"
    qualitative_prompts_path: Optional[str] = None
    redteam_prompts_path: Optional[str] = None


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quant: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    losses: LossWeights = field(default_factory=LossWeights)
    safety: SafetyTeacherConfig = field(default_factory=SafetyTeacherConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    label_policy: LabelPolicy = field(default_factory=LabelPolicy)

    def ensure_dirs(self) -> None:
        self.paths.ensure_dirs()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_config(config_path: Optional[str | Path] = None, overrides: Optional[dict[str, Any]] = None) -> AppConfig:
    """Load config with optional JSON override file + dict overrides.

    - If `config_path` is provided, it must be JSON.
    - `overrides` is a nested dict merged over defaults (override wins).
    """

    cfg = AppConfig()

    merged = cfg.to_dict()
    if config_path is not None:
        p = Path(config_path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Config JSON must be an object")
        merged = _merge_dict(merged, data)

    if overrides:
        merged = _merge_dict(merged, overrides)

    cfg2 = _app_config_from_dict(merged)
    cfg2.ensure_dirs()
    return cfg2


def with_overrides(cfg: AppConfig, **kwargs: Any) -> AppConfig:
    """Convenience for small programmatic overrides."""

    return replace(cfg, **kwargs)
