import json
import os
from collections import Counter
from typing import Any, Iterable

from datasets import DatasetDict, load_dataset


DATASETS_TO_CHECK = {
    # NOTE: `facebook/empathetic_dialogues` currently relies on a dataset script.
    # Newer `datasets` releases no longer support dataset scripts, so we also test
    # a few data-only mirrors that may be compatible.
    "empathetic_dialogues_facebook": {
        "hf_id": "facebook/empathetic_dialogues",
    },
    "empathetic_dialogues_lighteval": {
        "hf_id": "lighteval/empathetic_dialogues",
    },
    "empathetic_dialogues_for_lm": {
        "hf_id": "pixelsandpointers/empathetic_dialogues_for_lm",
    },
    "empathetic_dialogues_v2": {
        "hf_id": "Adapting/empathetic_dialogues_v2",
    },
    "esconv_thu": {
        "hf_id": "thu-coai/esconv",
    },
    "esconv_original": {
        "hf_id": "Ashokajou51/ESConv_Original",
    },
    "esconv_sorted": {
        "hf_id": "Ashokajou51/ESConv_Sorted",
    },
    "go_emotions": {
        "hf_id": "google-research-datasets/go_emotions",
    },
}


def _keys(sample: Any) -> list[str]:
    if isinstance(sample, dict):
        return list(sample.keys())
    return []


def _maybe_get_first(ds: Any) -> dict:
    try:
        return ds[0]
    except Exception:
        return {}


def _try_parse_json_text(sample_text: str) -> dict:
    text = sample_text.strip()
    if not text:
        return {}
    if not (text.startswith("{") and text.endswith("}")):
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _safe_counter(values: Iterable[Any], max_unique: int = 30) -> dict[str, int]:
    c = Counter()
    for v in values:
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            # For multi-label, count per element.
            for x in v:
                c[str(x)] += 1
        else:
            c[str(v)] += 1
        if len(c) > max_unique:
            break
    return dict(c.most_common(20))


def _infer_label_columns(column_names: list[str]) -> dict[str, list[str]]:
    lower = [c.lower() for c in column_names]
    emotion_like = [c for c, lc in zip(column_names, lower) if any(k in lc for k in ["emotion", "emotional", "sentiment", "feeling"])]
    strategy_like = [c for c, lc in zip(column_names, lower) if any(k in lc for k in ["strategy", "support", "act", "tactic"])]
    label_like = [c for c, lc in zip(column_names, lower) if lc in {"label", "labels"} or lc.endswith("_label") or lc.endswith("_labels")]
    return {
        "emotion_like": emotion_like,
        "strategy_like": strategy_like,
        "label_like": label_like,
    }


def _inspect_dialog_structure(sample: dict) -> None:
    if "dialog" not in sample:
        return
    dialog = sample.get("dialog")
    if not isinstance(dialog, list):
        print(f"dialog type: {type(dialog).__name__}")
        return
    print(f"dialog len: {len(dialog)}")
    for i, turn in enumerate(dialog[:3]):
        if isinstance(turn, dict):
            print(f"dialog[{i}] keys: {list(turn.keys())}")
            # If nested strategy fields exist, surface their keys without printing text.
            for k in ("strategy", "strategies", "annotation", "labels", "act"):
                if k in turn:
                    v = turn.get(k)
                    if isinstance(v, dict):
                        print(f"dialog[{i}].{k} keys: {list(v.keys())}")
        else:
            print(f"dialog[{i}] type: {type(turn).__name__}")


def _inspect_goemotions_labels(split: Any) -> None:
    if "labels" not in getattr(split, "column_names", []):
        return
    n = min(1000, split.num_rows)
    labels = split.select(range(n))["labels"]
    lengths = [len(x) if isinstance(x, (list, tuple)) else (1 if x is not None else 0) for x in labels]
    if not lengths:
        return
    counts = Counter(lengths)
    single = counts.get(1, 0)
    multi = sum(v for k, v in counts.items() if k >= 2)
    zero = counts.get(0, 0)
    print(f"labels length stats (sample {n}): single={single}, multi={multi}, zero={zero}, max_len={max(lengths)}")


def inspect_dataset(hf_id: str) -> None:
    _print_header(f"Loading {hf_id}")
    try:
        ds = load_dataset(hf_id)
    except Exception as e:
        print(f"FAILED to load {hf_id}: {type(e).__name__}: {e}")
        return

    if isinstance(ds, DatasetDict):
        splits = list(ds.keys())
        print(f"splits: {splits}")
        for split_name in splits:
            split = ds[split_name]
            print(f"\n-- split: {split_name} --")
            print(f"num_rows: {split.num_rows}")
            print(f"column_names: {split.column_names}")
            print(f"label-column candidates: {json.dumps(_infer_label_columns(split.column_names), indent=2)}")

            sample = _maybe_get_first(split)
            print(f"sample keys: {_keys(sample)}")

            # If this dataset stores structured JSON as a single text field, try to parse.
            if split.column_names == ["text"] and isinstance(sample.get("text"), str):
                parsed = _try_parse_json_text(sample["text"])
                if parsed:
                    print(f"parsed JSON from text keys: {_keys(parsed)}")
                    _inspect_dialog_structure(parsed)

            _inspect_dialog_structure(sample)
            _inspect_goemotions_labels(split)

            # Try to print a quick peek at plausible label columns without dumping full text.
            for col in _infer_label_columns(split.column_names)["emotion_like"][:3]:
                values = split.select(range(min(200, split.num_rows)))[col]
                print(f"top emotion-like values for '{col}': {_safe_counter(values)}")
            for col in _infer_label_columns(split.column_names)["strategy_like"][:3]:
                values = split.select(range(min(200, split.num_rows)))[col]
                print(f"top strategy-like values for '{col}': {_safe_counter(values)}")
            for col in _infer_label_columns(split.column_names)["label_like"][:3]:
                values = split.select(range(min(200, split.num_rows)))[col]
                print(f"top label-like values for '{col}': {_safe_counter(values)}")
    else:
        # Some datasets return a single Dataset (rare here).
        print(f"type: {type(ds)}")


def main() -> None:
    print("Inspecting candidate datasets from the Hugging Face Hub...")
    print("Tip: set HF_HOME to control cache location.")

    # Optional: keep cache inside repo to make it easier to clean up.
    os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".hf_cache"))

    for name, spec in DATASETS_TO_CHECK.items():
        inspect_dataset(spec["hf_id"])


if __name__ == "__main__":
    main()
