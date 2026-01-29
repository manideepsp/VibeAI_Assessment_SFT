from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import AppConfig, load_config
from model import build_bnb_config, infer_device


@dataclass(frozen=True)
class PromptItem:
    prompt_id: str
    prompt: str
    meta: dict[str, Any]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_prompt_file(path: Path) -> list[PromptItem]:
    """Load prompts from jsonl/json/txt.

    - jsonl: each line must contain {"prompt": str, "id"?: str}
    - json: list of objects with "prompt" and optional "id"
    - txt: one prompt per line
    """

    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() == ".jsonl":
        out: list[PromptItem] = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = str(obj.get("prompt", ""))
            pid = str(obj.get("id", i))
            out.append(PromptItem(prompt_id=pid, prompt=prompt, meta={k: v for k, v in obj.items() if k not in {"prompt", "id"}}))
        return out

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON prompt file must be a list")
        out = []
        for i, obj in enumerate(data):
            if not isinstance(obj, dict):
                raise ValueError("JSON prompt list must contain objects")
            prompt = str(obj.get("prompt", ""))
            pid = str(obj.get("id", i))
            out.append(PromptItem(prompt_id=pid, prompt=prompt, meta={k: v for k, v in obj.items() if k not in {"prompt", "id"}}))
        return out

    # txt
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [PromptItem(prompt_id=str(i), prompt=ln, meta={}) for i, ln in enumerate(lines)]


def _pick_prompt_field(example: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Best-effort extraction for EQ-Bench-style datasets."""

    for k in ("prompt", "question", "instruction", "input", "text"):
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            meta = {kk: vv for kk, vv in example.items() if kk != k}
            return v, meta

    # fallback: first non-empty string field
    for k, v in example.items():
        if isinstance(v, str) and v.strip():
            meta = {kk: vv for kk, vv in example.items() if kk != k}
            return v, meta

    raise ValueError("Could not find a string prompt field in example")


def load_eqbench_prompts(
    dataset_id: str = "eq-bench/eq-bench-3",
    split: str = "test",
    limit: Optional[int] = None,
) -> list[PromptItem]:
    # Prefer streaming when a limit is provided to avoid downloading the full dataset.
    if limit is not None:
        ds = load_dataset(dataset_id, split=split, streaming=True)
        items: list[PromptItem] = []
        for i, ex in enumerate(ds):
            if i >= int(limit):
                break
            prompt, meta = _pick_prompt_field(ex)
            pid = str(ex.get("id", ex.get("prompt_id", i)))
            items.append(PromptItem(prompt_id=pid, prompt=prompt, meta=meta))
        return items

    ds = load_dataset(dataset_id, split=split)
    if not isinstance(ds, Dataset):
        raise ValueError("Expected a single Dataset split")

    items: list[PromptItem] = []
    for i in range(len(ds)):
        ex = ds[i]
        prompt, meta = _pick_prompt_field(ex)
        pid = str(ex.get("id", ex.get("prompt_id", i)))
        items.append(PromptItem(prompt_id=pid, prompt=prompt, meta=meta))

    return items


def _format_chat_input(tokenizer, prompt: str) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        enc = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        # Some tokenizers return a BatchEncoding, others return a Tensor.
        if isinstance(enc, torch.Tensor):
            return enc
        if isinstance(enc, Mapping) and "input_ids" in enc:
            return enc["input_ids"]
        raise TypeError(f"Unexpected chat template encoding type: {type(enc)}")

    enc = tokenizer(prompt, return_tensors="pt")
    if isinstance(enc, Mapping) and "input_ids" in enc:
        return enc["input_ids"]
    raise TypeError(f"Unexpected tokenizer encoding type: {type(enc)}")


def generate_one(model, tokenizer, prompt: str, cfg: AppConfig, device: torch.device) -> str:
    input_ids = _format_chat_input(tokenizer, prompt).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    do_sample = cfg.generation.temperature is not None and float(cfg.generation.temperature) > 0.0

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(cfg.generation.max_new_tokens),
            do_sample=do_sample,
            temperature=float(cfg.generation.temperature),
            top_p=float(cfg.generation.top_p),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = out[0, input_ids.shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def load_inference_model(cfg: AppConfig, device: torch.device, adapter_dir: Optional[str] = None):
    tok_id = cfg.model.tokenizer_id or cfg.model.base_model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = build_bnb_config(cfg) if cfg.quant.load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_id,
        quantization_config=quantization_config,
        torch_dtype=None,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model.to(device)

    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.to(device)

    model.eval()
    return model, tokenizer


def export_responses(
    cfg: AppConfig,
    prompts: list[PromptItem],
    out_path: Path,
    adapter_dir: Optional[str],
    model_tag: str,
) -> None:
    device = infer_device()

    # Reproducibility for sampling-based decoding.
    torch.manual_seed(int(cfg.train.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.train.seed))

    model, tokenizer = load_inference_model(cfg, device=device, adapter_dir=adapter_dir)

    decoding = {
        "max_new_tokens": int(cfg.generation.max_new_tokens),
        "temperature": float(cfg.generation.temperature),
        "top_p": float(cfg.generation.top_p),
        "seed": int(cfg.train.seed),
    }

    model_info: dict[str, Any] = {
        "type": model_tag,
        "base_model_id": cfg.model.base_model_id,
        "adapter_dir": adapter_dir,
    }

    rows = []
    for item in tqdm(prompts, desc=f"export_{model_tag}", dynamic_ncols=True):
        resp = generate_one(model, tokenizer, item.prompt, cfg, device)
        rows.append(
            {
                "prompt_id": item.prompt_id,
                "prompt": item.prompt,
                "model": model_info,
                "response": resp,
                "meta": item.meta,
                "decoding": decoding,
            }
        )

    _write_jsonl(out_path, rows)
    print(f"wrote={out_path} rows={len(rows)}")


def export_responses_with_loaded_model(
    cfg: AppConfig,
    prompts: list[PromptItem],
    out_path: Path,
    *,
    model,
    tokenizer,
    model_tag: str,
    device: torch.device,
    adapter_dir: Optional[str],
) -> None:
    decoding = {
        "max_new_tokens": int(cfg.generation.max_new_tokens),
        "temperature": float(cfg.generation.temperature),
        "top_p": float(cfg.generation.top_p),
        "seed": int(cfg.train.seed),
    }

    model_info: dict[str, Any] = {
        "type": model_tag,
        "base_model_id": cfg.model.base_model_id,
        "adapter_dir": adapter_dir,
    }

    rows = []
    for item in tqdm(prompts, desc=f"export_{model_tag}", dynamic_ncols=True):
        resp = generate_one(model, tokenizer, item.prompt, cfg, device)
        rows.append(
            {
                "prompt_id": item.prompt_id,
                "prompt": item.prompt,
                "model": model_info,
                "response": resp,
                "meta": item.meta,
                "decoding": decoding,
            }
        )

    _write_jsonl(out_path, rows)
    print(f"wrote={out_path} rows={len(rows)}")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to JSON config override")
    ap.add_argument("--adapter_dir", type=str, default=None, help="Path to PEFT adapter dir for SFT model")
    ap.add_argument("--out_dir", type=str, default=None, help="Where to write eval artifacts")

    ap.add_argument(
        "--skip_base",
        action="store_true",
        help="Skip exporting the base model (useful when running base/SFT in separate processes).",
    )
    ap.add_argument(
        "--skip_sft",
        action="store_true",
        help="Skip exporting the SFT adapter even if --adapter_dir is provided.",
    )

    ap.add_argument("--eqbench", action="store_true", help="Run EQ-Bench 3 prompt export")
    ap.add_argument("--eqbench_dataset_id", type=str, default=None)
    ap.add_argument("--eqbench_split", type=str, default=None)
    ap.add_argument("--eqbench_limit", type=int, default=None)

    ap.add_argument("--qual", action="store_true", help="Run qualitative prompt export")
    ap.add_argument("--redteam", action="store_true", help="Run red-team prompt export")

    args = ap.parse_args(argv)

    cfg = load_config(config_path=args.config)

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.paths.artifacts_dir) / "eval"
    _ensure_dir(out_dir)

    ran_any = False
    prompt_sets: list[tuple[str, list[PromptItem]]] = []

    if args.eqbench or cfg.eval.eqbench_enabled:
        ran_any = True
        try:
            dataset_id = args.eqbench_dataset_id or getattr(cfg.eval, "eqbench_dataset_id", "eq-bench/eq-bench-3")
            split = args.eqbench_split or getattr(cfg.eval, "eqbench_split", "test")
            prompts = load_eqbench_prompts(dataset_id=dataset_id, split=split, limit=args.eqbench_limit)
        except Exception as e:
            raise RuntimeError(
                "Failed to load EQ-Bench 3 prompts. Ensure `datasets` can access 'eq-bench/eq-bench-3' "
                "(network/auth) or disable eqbench in config.eval.eqbench_enabled."
            ) from e
        prompt_sets.append(("eqbench", prompts))

    if args.qual and cfg.eval.qualitative_prompts_path:
        ran_any = True
        prompt_sets.append(("qual", _read_prompt_file(Path(cfg.eval.qualitative_prompts_path))))

    if args.redteam and cfg.eval.redteam_prompts_path:
        ran_any = True
        prompt_sets.append(("redteam", _read_prompt_file(Path(cfg.eval.redteam_prompts_path))))

    if not ran_any:
        print("Nothing to run. Use --eqbench/--qual/--redteam or set config.eval.* paths.")
        return

    if args.skip_base and (not args.adapter_dir or args.skip_sft):
        print("Nothing to export: --skip_base with no SFT export enabled.")
        return

    device = infer_device()
    torch.manual_seed(int(cfg.train.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.train.seed))

    # Load the base model once, then reuse it for all prompt sets.
    base_model = None
    tokenizer = None
    if not args.skip_base or (args.adapter_dir and not args.skip_sft):
        base_model, tokenizer = load_inference_model(cfg, device=device, adapter_dir=None)

    # Base exports.
    if not args.skip_base:
        for name, prompts in prompt_sets:
            export_responses_with_loaded_model(
                cfg,
                prompts,
                out_dir / f"{name}_base.jsonl",
                model=base_model,
                tokenizer=tokenizer,
                model_tag="base",
                device=device,
                adapter_dir=None,
            )

    # SFT exports: attach adapter to the existing base model to avoid a second full model load.
    if args.adapter_dir and not args.skip_sft:
        sft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
        sft_model.to(device)
        sft_model.eval()
        for name, prompts in prompt_sets:
            export_responses_with_loaded_model(
                cfg,
                prompts,
                out_dir / f"{name}_sft.jsonl",
                model=sft_model,
                tokenizer=tokenizer,
                model_tag="sft",
                device=device,
                adapter_dir=args.adapter_dir,
            )


if __name__ == "__main__":
    main()
