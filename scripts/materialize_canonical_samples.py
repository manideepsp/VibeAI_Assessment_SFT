import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Optional

from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass(frozen=True)
class CanonicalExample:
    dataset: str
    messages: list[dict[str, str]]
    input_ids: list[int]
    labels: list[int]
    emotion_label: Optional[int]
    strategy_label: Optional[int]


def _seed_everything(seed: int) -> None:
    random.seed(seed)


def _mask_labels(prompt_len: int, input_ids: list[int]) -> list[int]:
    labels = input_ids.copy()
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    return labels


def _tokenize_chat(tokenizer: Any, prompt_messages: list[dict[str, str]], full_messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
    """Returns (input_ids, labels) where labels are masked for prompt tokens."""
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

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    labels = _mask_labels(prompt_len=len(prompt_ids), input_ids=full_ids)
    return full_ids, labels


def _build_ed_v2_example(ds: Any, tokenizer: Any, seed: int) -> CanonicalExample:
    row = ds[random.randrange(ds.num_rows)]

    # Avoid label leakage: only use conversation history for context and sys_response as target.
    # DO NOT include: emotion, situation, behavior, question or not.
    chat_history = row.get("chat_history")
    sys_response = row.get("sys_response")
    emotion = row.get("emotion")

    if not isinstance(chat_history, str) or not isinstance(sys_response, str):
        raise ValueError("Unexpected schema for Adapting/empathetic_dialogues_v2: expected string chat_history and sys_response")

    # Represent chat_history as a single user message for now (safe and simple).
    # During training we can optionally segment turns, but this script is only to
    # validate canonical formatting and leakage.
    messages = [
        {"role": "user", "content": chat_history.strip()},
        {"role": "assistant", "content": sys_response.strip()},
    ]

    prompt_messages = messages[:1]
    input_ids, labels = _tokenize_chat(tokenizer, prompt_messages, messages)

    # Emotion label mapping (train-only in training code). Here we just demonstrate materialization.
    # For this sample script, we map string labels deterministically by sorting observed labels
    # within this split.
    emotion_values = ds.unique("emotion")
    emotion_values = [e for e in emotion_values if isinstance(e, str)]
    emotion_values_sorted = sorted(set(emotion_values))
    emo2id = {e: i for i, e in enumerate(emotion_values_sorted)}
    emotion_label = emo2id.get(emotion) if isinstance(emotion, str) else None

    return CanonicalExample(
        dataset="Adapting/empathetic_dialogues_v2",
        messages=messages,
        input_ids=input_ids,
        labels=labels,
        emotion_label=emotion_label,
        strategy_label=None,
    )


def _parse_esconv_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except Exception as e:
        raise ValueError(f"Failed to parse ESConv JSON text: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError("ESConv row JSON was not an object")
    return parsed


def _speaker_to_role(speaker: str) -> str:
    s = speaker.strip().lower()
    if s in {"seeker", "user", "client"}:
        return "user"
    if s in {"supporter", "assistant", "system", "sys"}:
        return "assistant"
    return "user"


def _build_esconv_example(ds: Any, tokenizer: Any, seed: int) -> CanonicalExample:
    # Find any conversation containing an assistant/supporter turn with an explicit strategy label.
    # Some rows may not have strategy annotations; we search across rows to reliably produce a sample.
    if "text" not in getattr(ds, "column_names", []):
        raise ValueError("Unexpected schema for thu-coai/esconv: expected string column 'text'")

    def find_turn_with_strategy(row_obj: dict) -> tuple[dict, int, Any] | None:
        if not isinstance(row_obj.get("text"), str):
            return None
        convo_obj = _parse_esconv_json(row_obj["text"])
        dialog_obj = convo_obj.get("dialog")
        if not isinstance(dialog_obj, list) or not dialog_obj:
            return None
        for idx, turn in enumerate(dialog_obj):
            if not isinstance(turn, dict):
                continue
            speaker = turn.get("speaker")
            if not isinstance(speaker, str):
                continue
            if _speaker_to_role(speaker) != "assistant":
                continue
            if turn.get("strategy") is None:
                continue
            return convo_obj, idx, turn.get("strategy")
        return None

    # Try random rows first (fast), then fall back to sequential scan.
    max_tries = min(ds.num_rows, 200)
    found: tuple[dict, int, Any] | None = None
    for _ in range(max_tries):
        row = ds[random.randrange(ds.num_rows)]
        found = find_turn_with_strategy(row)
        if found is not None:
            break
    if found is None:
        for i in range(ds.num_rows):
            found = find_turn_with_strategy(ds[i])
            if found is not None:
                break
    if found is None:
        raise ValueError("Could not find any assistant turn with a strategy label in ESConv split")

    convo, turn_index, strategy_value = found
    dialog = convo.get("dialog")
    assert isinstance(dialog, list)

    # Build a next-turn example:
    # context = all turns before the chosen assistant turn
    # target  = chosen assistant turn text
    context_turns = dialog[:turn_index]
    target_turn = dialog[turn_index]

    messages: list[dict[str, str]] = []
    for t in context_turns:
        if not isinstance(t, dict):
            continue
        speaker = t.get("speaker")
        text = t.get("text")
        if not isinstance(speaker, str) or not isinstance(text, str):
            continue
        messages.append({"role": _speaker_to_role(speaker), "content": text.strip()})

    # Target assistant message.
    target_text = target_turn.get("text")
    if not isinstance(target_text, str):
        raise ValueError("ESConv target turn has no text")
    messages.append({"role": "assistant", "content": target_text.strip()})

    prompt_messages = messages[:-1]
    input_ids, labels = _tokenize_chat(tokenizer, prompt_messages, messages)

    # Strategy mapping (train-only in training code). For this sample script we build a deterministic
    # mapping from all observed strategy values in this split (small enough to scan fully).
    strategies: list[str] = []
    for i in range(ds.num_rows):
        r = ds[i]
        if not isinstance(r.get("text"), str):
            continue
        try:
            c = _parse_esconv_json(r["text"])
        except Exception:
            continue
        d = c.get("dialog")
        if not isinstance(d, list):
            continue
        for tr in d:
            if isinstance(tr, dict) and tr.get("strategy") is not None:
                strategies.append(str(tr.get("strategy")))

    strat_values_sorted = sorted(set(strategies))
    strat2id = {s: i for i, s in enumerate(strat_values_sorted)}
    strategy_label = strat2id.get(str(strategy_value))

    return CanonicalExample(
        dataset="thu-coai/esconv",
        messages=messages,
        input_ids=input_ids,
        labels=labels,
        emotion_label=None,
        strategy_label=strategy_label,
    )


def _build_goemotions_example(ds: Any, tokenizer: Any, seed: int) -> CanonicalExample:
    # Filter to single-label rows to match CE emotion head.
    # (This is only for a sample; training code will do this in preprocessing.)
    for _ in range(50):
        row = ds[random.randrange(ds.num_rows)]
        text = row.get("text")
        labels = row.get("labels")
        if isinstance(text, str) and isinstance(labels, list) and len(labels) == 1:
            emotion_label = int(labels[0])
            messages = [
                {"role": "user", "content": text.strip()},
                # No assistant response in GoEmotions; we keep an empty assistant target so
                # LM labels are fully masked (no LM loss contribution).
                {"role": "assistant", "content": ""},
            ]
            prompt_messages = messages
            input_ids, lm_labels = _tokenize_chat(tokenizer, prompt_messages, messages)
            return CanonicalExample(
                dataset="google-research-datasets/go_emotions",
                messages=messages,
                input_ids=input_ids,
                labels=lm_labels,
                emotion_label=emotion_label,
                strategy_label=None,
            )

    raise ValueError("Could not sample a single-label GoEmotions row after 50 tries")


def _print_example(ex: CanonicalExample) -> None:
    print("\n" + "-" * 100)
    print(f"dataset: {ex.dataset}")
    print(f"canonical keys: {list(ex.__dict__.keys())}")

    # Print messages with lengths only (avoid dumping full text).
    for i, m in enumerate(ex.messages[:6]):
        role = m.get("role")
        content = m.get("content", "")
        print(f"messages[{i}]: role={role}, chars={len(content)}")

    print(f"input_ids_len: {len(ex.input_ids)}")
    print(f"labels_len: {len(ex.labels)}")
    masked = sum(1 for x in ex.labels if x == -100)
    print(f"masked_labels: {masked} ({masked / max(1, len(ex.labels)):.2%})")

    print(f"emotion_label: {ex.emotion_label}")
    print(f"strategy_label: {ex.strategy_label}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize one canonical sample per dataset and print a leak-safe summary.")
    ap.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Tokenizer id to use for chat templating. Default is a 3B-class instruction tokenizer.",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # Keep cache inside repo if desired.
    os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".hf_cache"))

    _seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    # 1) EmpatheticDialogues replacement (with explicit emotion labels)
    ed = load_dataset("Adapting/empathetic_dialogues_v2")
    ex_ed = _build_ed_v2_example(ed["train"], tokenizer, seed=args.seed)
    _print_example(ex_ed)

    # 2) ESConv with strategy labels in dialog turns
    es = load_dataset("thu-coai/esconv")
    ex_es = _build_esconv_example(es["train"], tokenizer, seed=args.seed)
    _print_example(ex_es)

    # 3) GoEmotions single-label subset for CE emotion head
    go = load_dataset("google-research-datasets/go_emotions")
    ex_go = _build_goemotions_example(go["train"], tokenizer, seed=args.seed)
    _print_example(ex_go)


if __name__ == "__main__":
    main()
