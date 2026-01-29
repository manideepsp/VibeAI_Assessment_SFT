from __future__ import annotations

import time
import json
import datetime
import subprocess
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from config import AppConfig, load_config
from dataset import CanonicalDatasets, IGNORE_INDEX, load_canonical_datasets
from heads import AuxHeads, masked_cross_entropy
from mixture import TemperatureMixtureSampler
from model import LoadedModel, infer_device, load_model_and_tokenizer


@dataclass(frozen=True)
class TrainStats:
    step: int
    lm_loss: float
    emo_loss: float
    strat_loss: float
    total_loss: float
    tokens: int
    seconds: float


class _MixtureIterableDataset(IterableDataset):
    def __init__(self, sampler: TemperatureMixtureSampler):
        super().__init__()
        self._sampler = sampler

    def __iter__(self):
        yield from iter(self._sampler)


def _pad_1d(seqs: list[list[int]], pad_value: int) -> torch.Tensor:
    max_len = max((len(x) for x in seqs), default=0)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        if not s:
            continue
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def collate_batch(batch: list[dict[str, Any]], tokenizer) -> dict[str, torch.Tensor]:
    input_ids = [ex["input_ids"] for ex in batch]
    attention_mask = [ex.get("attention_mask", [1] * len(ex["input_ids"])) for ex in batch]
    labels = [ex["labels"] for ex in batch]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids_t = _pad_1d(input_ids, pad_id)
    attention_mask_t = _pad_1d(attention_mask, 0)
    labels_t = _pad_1d(labels, IGNORE_INDEX)

    prompt_length_t = torch.tensor([int(ex["prompt_length"]) for ex in batch], dtype=torch.long)
    emotion_label_t = torch.tensor([int(ex.get("emotion_label", IGNORE_INDEX)) for ex in batch], dtype=torch.long)
    strategy_label_t = torch.tensor([int(ex.get("strategy_label", IGNORE_INDEX)) for ex in batch], dtype=torch.long)

    return {
        "input_ids": input_ids_t,
        "attention_mask": attention_mask_t,
        "labels": labels_t,
        "prompt_length": prompt_length_t,
        "emotion_label": emotion_label_t,
        "strategy_label": strategy_label_t,
    }


def _split_by_source(ds) -> dict[str, Any]:
    sources = sorted(set(ds["source"]))
    by_source: dict[str, Any] = {}
    col = ds["source"]
    for s in sources:
        idxs = [i for i, v in enumerate(col) if v == s]
        by_source[s] = ds.select(idxs)
    return by_source


def _infer_num_emotions(canonical: CanonicalDatasets) -> int:
    # GoEmotions is a 28-way taxonomy in the canonical HF dataset.
    # We infer it from the observed max label in the (filtered) canonical train split.
    labels = canonical.train["emotion_label"]
    m = -1
    for v in labels:
        if v == IGNORE_INDEX:
            continue
        if int(v) > m:
            m = int(v)
    return (m + 1) if m >= 0 else 0


def _infer_num_strategies(canonical: CanonicalDatasets) -> int:
    return len(canonical.label_maps.strategy2id)


def _trainable_params_summary(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _default_checkpoint_root(cfg: AppConfig) -> Path:
    if cfg.train.output_dir:
        return Path(cfg.train.output_dir)
    return Path(cfg.paths.artifacts_dir) / "checkpoints"


def _save_checkpoint(peft_model: torch.nn.Module, heads: AuxHeads, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # PEFT adapter
    if hasattr(peft_model, "save_pretrained"):
        peft_model.save_pretrained(str(out_dir / "adapter"))
    else:
        raise RuntimeError("Model does not support save_pretrained (expected PEFT model)")

    torch.save(heads.state_dict(), out_dir / "aux_heads.pt")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        s = out.decode("utf-8", errors="ignore").strip()
        return s or None
    except Exception:
        return None


def _write_run_manifest(cfg: AppConfig, *, log_every: int, eval_every: int, warmup_steps: int, device: torch.device) -> str:
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit_hash": _git_commit_hash(),
        "device": str(device),
        "base_model_id": cfg.model.base_model_id,
        "peft": "qlora" if cfg.quant.load_in_4bit else "lora",
        "max_train_steps": int(cfg.train.max_train_steps),
        "log_every_steps": int(log_every),
        "eval_every_steps": int(eval_every),
        "save_steps": int(cfg.train.save_steps),
        "mixture_alpha": float(cfg.datasets.mixture_alpha),
        "learning_rate": float(cfg.train.learning_rate),
        "weight_decay": float(cfg.train.weight_decay),
        "warmup_ratio": float(cfg.train.warmup_ratio),
        "warmup_steps": int(warmup_steps),
        "seed": int(cfg.train.seed),
        "loss_weights": {
            "lambda_lm": float(cfg.losses.lambda_lm),
            "lambda_emo": float(cfg.losses.lambda_emo),
            "lambda_strat": float(cfg.losses.lambda_strat),
        },
        "config": cfg.to_dict(),
    }

    out_path = Path(cfg.paths.artifacts_dir) / "run_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return run_id


def build_dataloaders(cfg: AppConfig, tokenizer, canonical: CanonicalDatasets) -> tuple[DataLoader, DataLoader]:
    by_source = _split_by_source(canonical.train)

    sampler = TemperatureMixtureSampler(
        by_source,
        alpha=cfg.datasets.mixture_alpha,
        seed=cfg.train.seed,
        max_examples=cfg.datasets.max_train_examples,
    )

    train_ds = _MixtureIterableDataset(sampler)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.per_device_train_batch_size,
        collate_fn=lambda b: collate_batch(b, tokenizer),
    )

    val_loader = DataLoader(
        canonical.validation,
        batch_size=cfg.train.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, tokenizer),
    )

    return train_loader, val_loader


def evaluate(
    model: torch.nn.Module,
    heads: AuxHeads,
    loader: DataLoader,
    device: torch.device,
    *,
    eval_max_len: int = 512,
) -> dict[str, float]:
    # Eval should be pure inference: do NOT pass labels to the base model.
    # This avoids allocating extra buffers for loss computation and is much
    # more memory-friendly on small GPUs.
    model.eval()
    heads.eval()

    total_emo = 0.0
    total_strat = 0.0
    steps = 0

    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Cap eval sequence length (often lower than train) to reduce peak VRAM.
            if isinstance(eval_max_len, int) and eval_max_len > 0:
                cur_len = int(batch["input_ids"].shape[1])
                if cur_len > eval_max_len:
                    batch["input_ids"] = batch["input_ids"][:, :eval_max_len]
                    batch["attention_mask"] = batch["attention_mask"][:, :eval_max_len]
                    batch["prompt_length"] = torch.clamp(batch["prompt_length"], min=1, max=eval_max_len)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )

            hs = outputs.hidden_states
            if hs is None:
                raise RuntimeError("Model did not return hidden_states; ensure output_hidden_states=True")
            last_h = hs[-1]

            aux = heads(last_h, batch["prompt_length"])

            emo_loss = torch.tensor(0.0, device=device)
            strat_loss = torch.tensor(0.0, device=device)
            if aux.emotion is not None:
                emo_loss = masked_cross_entropy(aux.emotion, batch["emotion_label"], ignore_index=IGNORE_INDEX)
            if aux.strategy is not None:
                strat_loss = masked_cross_entropy(aux.strategy, batch["strategy_label"], ignore_index=IGNORE_INDEX)

            total_emo += float(emo_loss.detach().cpu())
            total_strat += float(strat_loss.detach().cpu())
            steps += 1

    if steps == 0:
        return {"lm_loss": float("nan"), "emo_loss": 0.0, "strat_loss": 0.0}

    return {
        # LM loss is intentionally disabled during eval.
        "lm_loss": float("nan"),
        "emo_loss": total_emo / steps,
        "strat_loss": total_strat / steps,
    }


def train(cfg: AppConfig) -> None:
    device = infer_device()

    loaded: LoadedModel = load_model_and_tokenizer(cfg, device=device)
    tokenizer = loaded.tokenizer
    model = loaded.model

    canonical = load_canonical_datasets(cfg, tokenizer)
    train_loader, val_loader = build_dataloaders(cfg, tokenizer, canonical)

    num_emotions = _infer_num_emotions(canonical)
    num_strategies = _infer_num_strategies(canonical)

    hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "n_embd", 0))
    if hidden_size <= 0:
        raise RuntimeError("Could not infer hidden size from model.config")

    # Ensure auxiliary heads use the same dtype and device as the base model
    # to avoid matmul dtype mismatches (e.g., bf16 activations vs fp32 weights).
    try:
        base_dtype = next(model.parameters()).dtype
    except StopIteration:
        base_dtype = torch.float32

    heads = AuxHeads(
        hidden_size=hidden_size,
        num_emotions=num_emotions,
        num_strategies=num_strategies,
        dropout=0.1,
    ).to(device=device, dtype=base_dtype)

    trainable, total = _trainable_params_summary(model)
    print(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100.0*trainable/max(1,total):.4f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in list(model.parameters()) + list(heads.parameters()) if p.requires_grad],
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    total_steps = int(cfg.train.max_train_steps)
    if total_steps <= 0:
        raise ValueError("train.max_train_steps must be > 0")

    warmup_steps = int(cfg.train.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    heads.train()

    grad_accum = max(1, int(cfg.train.gradient_accumulation_steps))
    log_every = int(cfg.train.log_every_steps or cfg.train.logging_steps)
    eval_every = int(cfg.train.eval_every_steps or cfg.train.eval_steps)
    if log_every <= 0:
        raise ValueError("train.logging_steps/log_every_steps must be > 0")
    if eval_every <= 0:
        raise ValueError("train.eval_steps/eval_every_steps must be > 0")

    step = 0
    accum = 0
    running_tokens = 0
    t0 = time.time()

    iterator = iter(train_loader)

    ckpt_root = _default_checkpoint_root(cfg)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    log_dir = Path(cfg.paths.artifacts_dir) / "logs"
    train_log = log_dir / "train_metrics.jsonl"
    eval_log = log_dir / "eval_metrics.jsonl"

    run_id = _write_run_manifest(cfg, log_every=log_every, eval_every=eval_every, warmup_steps=warmup_steps, device=device)
    _append_jsonl(train_log, {"event": "run_start", "run_id": run_id})

    pbar = tqdm(total=total_steps, desc="train_steps", dynamic_ncols=True)

    while step < total_steps:
        batch = next(iterator)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_hidden_states=True,
                use_cache=False,
            )

            lm_loss = outputs.loss
            hs = outputs.hidden_states
            if hs is None:
                raise RuntimeError("Model did not return hidden_states; ensure output_hidden_states=True")
            last_h = hs[-1]
            if last_h.dim() != 3:
                raise RuntimeError("Expected hidden_states[-1] to be [B,T,H]")

            aux = heads(last_h, batch["prompt_length"])

            emo_loss = torch.tensor(0.0, device=device)
            strat_loss = torch.tensor(0.0, device=device)
            if aux.emotion is not None:
                emo_loss = masked_cross_entropy(aux.emotion, batch["emotion_label"], ignore_index=IGNORE_INDEX)
            if aux.strategy is not None:
                strat_loss = masked_cross_entropy(aux.strategy, batch["strategy_label"], ignore_index=IGNORE_INDEX)

            total_loss = (
                cfg.losses.lambda_lm * lm_loss
                + cfg.losses.lambda_emo * emo_loss
                + cfg.losses.lambda_strat * strat_loss
            )

        if use_amp:
            scaler.scale(total_loss / grad_accum).backward()
        else:
            (total_loss / grad_accum).backward()
        accum += 1

        running_tokens += int(batch["attention_mask"].sum().detach().cpu())

        if accum >= grad_accum:
            if use_amp:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                [p for p in list(model.parameters()) + list(heads.parameters()) if p.requires_grad],
                max_norm=cfg.train.max_grad_norm,
            )

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0

            step += 1
            pbar.update(1)

            if step % log_every == 0:
                dt = time.time() - t0
                stats = TrainStats(
                    step=step,
                    lm_loss=float(lm_loss.detach().cpu()),
                    emo_loss=float(emo_loss.detach().cpu()),
                    strat_loss=float(strat_loss.detach().cpu()),
                    total_loss=float(total_loss.detach().cpu()),
                    tokens=running_tokens,
                    seconds=dt,
                )
                toks_per_s = stats.tokens / max(1e-6, stats.seconds)
                print(
                    f"step={stats.step} total={stats.total_loss:.4f} lm={stats.lm_loss:.4f} "
                    f"emo={stats.emo_loss:.4f} strat={stats.strat_loss:.4f} "
                    f"tokens={stats.tokens} tok/s={toks_per_s:.1f}"
                )

                lr = float(optimizer.param_groups[0].get("lr", 0.0))
                gpu_mem_mb: Optional[float] = None
                if device.type == "cuda":
                    try:
                        gpu_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
                    except Exception:
                        gpu_mem_mb = None
                _append_jsonl(
                    train_log,
                    {
                        "step": stats.step,
                        "loss": stats.total_loss,
                        "lm_loss": stats.lm_loss,
                        "emotion_loss": stats.emo_loss,
                        "strategy_loss": stats.strat_loss,
                        "tokens": stats.tokens,
                        "seconds": stats.seconds,
                        "tokens_per_s": toks_per_s,
                        "lr": lr,
                        "gpu_mem_mb": gpu_mem_mb,
                    },
                )

            # Save before eval so we still have a checkpoint even if eval OOMs.
            if cfg.train.save_steps > 0 and step % int(cfg.train.save_steps) == 0:
                ckpt_dir = ckpt_root / f"step-{step}"
                _save_checkpoint(model, heads, ckpt_dir)
                print(f"saved_checkpoint={ckpt_dir}")
                _append_jsonl(
                    train_log,
                    {
                        "step": step,
                        "event": "checkpoint",
                        "path": str(ckpt_dir),
                    },
                )

            if step % eval_every == 0:
                # Fix #3: clear memory before eval to reduce fragmentation/peak.
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                metrics = evaluate(model, heads, val_loader, device, eval_max_len=512)
                print(
                    f"eval@{step}: lm=nan emo={metrics['emo_loss']:.4f} strat={metrics['strat_loss']:.4f}"
                )

                _append_jsonl(
                    eval_log,
                    {
                        "step": step,
                        "lm_loss": float(metrics["lm_loss"]),
                        "emotion_loss": float(metrics["emo_loss"]),
                        "strategy_loss": float(metrics["strat_loss"]),
                    },
                )

    pbar.close()

    final_dir = ckpt_root / "final"
    _save_checkpoint(model, heads, final_dir)
    print(f"saved_checkpoint={final_dir}")
    _append_jsonl(
        train_log,
        {
            "step": step,
            "event": "final_checkpoint",
            "path": str(final_dir),
        },
    )
    print("training_done")


def main() -> None:
    cfg = load_config()
    train(cfg)


if __name__ == "__main__":
    main()
