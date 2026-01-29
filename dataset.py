from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerBase

from config import AppConfig
from label_policy import LabelPolicy, goemotions_emotion_label, goemotions_is_single_label


IGNORE_INDEX = -100


@dataclass(frozen=True)
class LabelMaps:
    """Train-split-only vocabularies for auxiliary heads."""

    # ESConv strategy (string) -> class id
    strategy2id: dict[str, int]

    # Optional mappings (disabled by default by LabelPolicy)
    ed_emotion2id: dict[str, int]


@dataclass(frozen=True)
class CanonicalDatasets:
    train: Dataset
    validation: Dataset
    test: Optional[Dataset]
    label_maps: LabelMaps


def _set_hf_cache(cfg: AppConfig) -> None:
    # Keep HF cache under repo for reproducibility / easy cleanup.
    os.environ.setdefault("HF_HOME", cfg.paths.hf_cache_dir)


def _mask_labels(prompt_len: int, full_ids: list[int]) -> list[int]:
    labels = full_ids.copy()
    for i in range(min(prompt_len, len(labels))):
        labels[i] = IGNORE_INDEX
    return labels


def _tokenize_chat(
    tokenizer: PreTrainedTokenizerBase,
    prompt_messages: list[dict[str, str]],
    full_messages: list[dict[str, str]],
) -> tuple[list[int], list[int], int]:
    """Tokenize chat and return (input_ids, labels, prompt_length).

    prompt_length is measured in tokens of the prompt (including any generation prompt tokens).
    Labels are masked for prompt tokens so LM loss applies only to assistant tokens.

    For auxiliary heads, prompt_length is also the safest pooling anchor (use the last prompt token).
    """

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        # Fallback for non-chat tokenizers/models (CPU smoke runs).
        def _fmt(msgs: list[dict[str, str]], add_generation_prompt: bool) -> str:
            lines: list[str] = []
            for m in msgs:
                role = str(m.get("role", "user")).strip().lower()
                content = str(m.get("content", ""))
                tag = "User" if role == "user" else "Assistant"
                lines.append(f"{tag}: {content}")
            if add_generation_prompt:
                lines.append("Assistant:")
            return "\n".join(lines)

        prompt_text = _fmt(prompt_messages, add_generation_prompt=True)
        full_text = _fmt(full_messages, add_generation_prompt=False)

    # Use tokenizer.model_max_length as an upper bound if set to keep
    # sequences within a GPU-friendly window.
    max_len = getattr(tokenizer, "model_max_length", None)
    kw = {"add_special_tokens": False}
    if isinstance(max_len, int) and max_len > 0 and max_len < 10_000:
        kw.update({"truncation": True, "max_length": max_len})

    prompt_ids = tokenizer(prompt_text, **kw).input_ids
    full_ids = tokenizer(full_text, **kw).input_ids

    labels = _mask_labels(prompt_len=len(prompt_ids), full_ids=full_ids)
    return full_ids, labels, len(prompt_ids)


def _speaker_to_role(speaker: str) -> str:
    s = speaker.strip().lower()
    if s in {"seeker", "user", "client"}:
        return "user"
    if s in {"supporter", "assistant", "system", "sys"}:
        return "assistant"
    return "user"


def _load_esconv_dialog(row_text: str) -> dict[str, Any]:
    obj = json.loads(row_text)
    if not isinstance(obj, dict):
        raise ValueError("ESConv row JSON is not an object")
    return obj


def load_raw_datasets(cfg: AppConfig) -> dict[str, DatasetDict]:
    """Load raw HF datasets.

    NOTE: dataset.py never downloads models.
    """

    _set_hf_cache(cfg)

    return {
        "ed": load_dataset(cfg.datasets.empathetic_dialogues_id),
        "es": load_dataset(cfg.datasets.esconv_id),
        "go": load_dataset(cfg.datasets.go_emotions_id),
    }


def _build_label_maps(cfg: AppConfig, raw: dict[str, DatasetDict]) -> LabelMaps:
    """Build mapping dicts using TRAIN split only (prevents leakage)."""

    policy = cfg.label_policy

    # ESConv strategy vocab.
    strategies: set[str] = set()
    es_train = raw["es"]["train"]
    for i in range(es_train.num_rows):
        row = es_train[i]
        text = row.get("text")
        if not isinstance(text, str):
            continue
        try:
            convo = _load_esconv_dialog(text)
        except Exception:
            continue
        dialog = convo.get("dialog")
        if not isinstance(dialog, list):
            continue
        for turn in dialog:
            if not isinstance(turn, dict):
                continue
            strat = turn.get("strategy")
            if strat is None:
                continue
            strategies.add(str(strat))

    strategy2id = {s: idx for idx, s in enumerate(sorted(strategies))}

    # ED-v2 emotion mapping (only if explicitly enabled).
    ed_emotion2id: dict[str, int] = {}
    if policy.use_ed_v2_emotion_for_head:
        ed_train = raw["ed"]["train"]
        values = [v for v in ed_train.unique("emotion") if isinstance(v, str) and v.strip()]
        ed_emotion2id = {v: idx for idx, v in enumerate(sorted(set(values)))}

    return LabelMaps(strategy2id=strategy2id, ed_emotion2id=ed_emotion2id)


def _maybe_take_first_n(ds: Dataset, n: Optional[int], seed: int) -> Dataset:
    if n is None:
        return ds
    if n <= 0:
        return ds.select([])
    # Shuffle once for a deterministic subsample.
    ds2 = ds.shuffle(seed=seed)
    return ds2.select(range(min(n, ds2.num_rows)))


def _ed_v2_generator(
    cfg: AppConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: Dataset,
    label_maps: LabelMaps,
) -> Iterator[dict[str, Any]]:
    policy: LabelPolicy = cfg.label_policy

    for row in split:
        chat_history = row.get("chat_history")
        sys_response = row.get("sys_response")
        emotion_str = row.get("emotion")

        if not isinstance(chat_history, str) or not isinstance(sys_response, str):
            continue

        # No leakage: use chat_history only.
        messages = [
            {"role": "user", "content": chat_history.strip()},
            {"role": "assistant", "content": sys_response.strip()},
        ]
        prompt_messages = messages[:1]
        input_ids, labels, prompt_length = _tokenize_chat(tokenizer, prompt_messages, messages)

        emotion_label = IGNORE_INDEX
        if policy.use_ed_v2_emotion_for_head and isinstance(emotion_str, str):
            emotion_label = label_maps.ed_emotion2id.get(emotion_str, IGNORE_INDEX)

        yield {
            "source": "ed_v2",
            "messages": messages,
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "prompt_length": prompt_length,
            "emotion_label": int(emotion_label),
            "strategy_label": int(IGNORE_INDEX),
        }


def _esconv_generator(
    cfg: AppConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: Dataset,
    label_maps: LabelMaps,
) -> Iterator[dict[str, Any]]:
    # Strategy supervision is per assistant/supporter turn. We materialize one example per
    # assistant turn that has a non-null strategy label.
    for row in split:
        text = row.get("text")
        if not isinstance(text, str):
            continue
        try:
            convo = _load_esconv_dialog(text)
        except Exception:
            continue

        dialog = convo.get("dialog")
        if not isinstance(dialog, list) or not dialog:
            continue

        for turn_index, turn in enumerate(dialog):
            if not isinstance(turn, dict):
                continue
            speaker = turn.get("speaker")
            if not isinstance(speaker, str):
                continue
            if _speaker_to_role(speaker) != "assistant":
                continue

            strat = turn.get("strategy")
            if strat is None:
                continue

            target_text = turn.get("text")
            if not isinstance(target_text, str):
                continue

            # Build context from all turns before this assistant turn.
            context: list[dict[str, str]] = []
            for t in dialog[:turn_index]:
                if not isinstance(t, dict):
                    continue
                sp = t.get("speaker")
                tx = t.get("text")
                if not isinstance(sp, str) or not isinstance(tx, str):
                    continue
                context.append({"role": _speaker_to_role(sp), "content": tx.strip()})

            # Skip if context is empty; still okay but often low-signal.
            if not context:
                continue

            messages = context + [{"role": "assistant", "content": target_text.strip()}]
            prompt_messages = messages[:-1]

            input_ids, labels, prompt_length = _tokenize_chat(tokenizer, prompt_messages, messages)

            strategy_str = str(strat)
            strategy_label = label_maps.strategy2id.get(strategy_str, IGNORE_INDEX)

            yield {
                "source": "esconv",
                "messages": messages,
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "prompt_length": prompt_length,
                "emotion_label": int(IGNORE_INDEX),
                "strategy_label": int(strategy_label),
            }


def _goemotions_generator(
    cfg: AppConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: Dataset,
) -> Iterator[dict[str, Any]]:
    policy: LabelPolicy = cfg.label_policy

    for row in split:
        text = row.get("text")
        if not isinstance(text, str):
            continue

        # Clean CE head policy: single-label only.
        if policy.use_goemotions_single_label_only and not goemotions_is_single_label(row):
            continue

        emo = goemotions_emotion_label(row)
        if emo is None:
            continue

        # Head-only sample: no assistant target; LM labels fully masked.
        messages = [
            {"role": "user", "content": text.strip()},
            {"role": "assistant", "content": ""},
        ]

        # Treat the whole sequence as prompt so LM loss is zeroed.
        input_ids, labels, prompt_length = _tokenize_chat(tokenizer, messages, messages)

        yield {
            "source": "go_emotions",
            "messages": messages,
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "prompt_length": prompt_length,
            "emotion_label": int(emo),
            "strategy_label": int(IGNORE_INDEX),
        }


def _from_generator(gen_fn: Callable[[], Iterator[dict[str, Any]]]) -> Dataset:
    return Dataset.from_generator(gen_fn)


def load_canonical_datasets(cfg: AppConfig, tokenizer: PreTrainedTokenizerBase) -> CanonicalDatasets:
    """Load + materialize datasets into a single canonical schema.

    Output schema columns:
    - source: str
    - messages: list[{role, content}] (kept for debugging; training will use token columns)
    - input_ids: List[int]
    - attention_mask: List[int]
    - labels: List[int] (masked prompt tokens to IGNORE_INDEX)
    - prompt_length: int (pooling anchor for auxiliary heads)
    - emotion_label: int (or IGNORE_INDEX if absent)
    - strategy_label: int (or IGNORE_INDEX if absent)

    IMPORTANT: this function does not perform temperature-based sampling; it simply concatenates.
    The mixture sampler is implemented separately.
    """

    raw = load_raw_datasets(cfg)
    label_maps = _build_label_maps(cfg, raw)

    seed = cfg.train.seed

    # Subsample (optional) per-dataset per-split to keep iterations quick.
    ed_train = _maybe_take_first_n(raw["ed"]["train"], cfg.datasets.max_train_examples, seed)
    ed_val = _maybe_take_first_n(raw["ed"]["validation"], cfg.datasets.max_eval_examples, seed)

    es_train = _maybe_take_first_n(raw["es"]["train"], cfg.datasets.max_train_examples, seed)
    es_val = _maybe_take_first_n(raw["es"]["validation"], cfg.datasets.max_eval_examples, seed)

    go_train = _maybe_take_first_n(raw["go"]["train"], cfg.datasets.max_train_examples, seed)
    go_val = _maybe_take_first_n(raw["go"]["validation"], cfg.datasets.max_eval_examples, seed)

    train_parts = [
        _from_generator(lambda: _ed_v2_generator(cfg, tokenizer, ed_train, label_maps)),
        _from_generator(lambda: _esconv_generator(cfg, tokenizer, es_train, label_maps)),
        _from_generator(lambda: _goemotions_generator(cfg, tokenizer, go_train)),
    ]

    val_parts = [
        _from_generator(lambda: _ed_v2_generator(cfg, tokenizer, ed_val, label_maps)),
        _from_generator(lambda: _esconv_generator(cfg, tokenizer, es_val, label_maps)),
        _from_generator(lambda: _goemotions_generator(cfg, tokenizer, go_val)),
    ]

    train = concatenate_datasets(train_parts)
    validation = concatenate_datasets(val_parts)

    # Some datasets have test split; keep optional.
    test = None
    if "test" in raw["ed"] and "test" in raw["es"] and "test" in raw["go"]:
        ed_test = _maybe_take_first_n(raw["ed"]["test"], cfg.datasets.max_eval_examples, seed)
        es_test = _maybe_take_first_n(raw["es"]["test"], cfg.datasets.max_eval_examples, seed)
        go_test = _maybe_take_first_n(raw["go"]["test"], cfg.datasets.max_eval_examples, seed)
        test_parts = [
            _from_generator(lambda: _ed_v2_generator(cfg, tokenizer, ed_test, label_maps)),
            _from_generator(lambda: _esconv_generator(cfg, tokenizer, es_test, label_maps)),
            _from_generator(lambda: _goemotions_generator(cfg, tokenizer, go_test)),
        ]
        test = concatenate_datasets(test_parts)

    return CanonicalDatasets(train=train, validation=validation, test=test, label_maps=label_maps)
