"""Label mapping and filtering policy.

This module contains the *concrete, testable* decisions for:
- which dataset provides which supervision
- how to map raw labels to integer IDs
- how to avoid label leakage by construction

It is intentionally model-agnostic and tokenizer-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class LabelPolicy:
    """Project-level label policy choices.

    Notes:
    - We default to CE-style emotion classification, so GoEmotions is filtered to single-label rows.
    - EmpatheticDialogues-v2 emotion strings are NOT used for the emotion head by default because
      they are a different taxonomy; we keep them for analysis only unless an explicit mapping is enabled.
    """

    use_goemotions_single_label_only: bool = True
    use_ed_v2_emotion_for_head: bool = False
    use_esconv_emotion_type_for_head: bool = False


def goemotions_is_single_label(row: dict[str, Any]) -> bool:
    labels = row.get("labels")
    return isinstance(labels, list) and len(labels) == 1


def goemotions_emotion_label(row: dict[str, Any]) -> Optional[int]:
    """Return an integer emotion label for GoEmotions, else None."""
    labels = row.get("labels")
    if not isinstance(labels, list) or len(labels) != 1:
        return None
    try:
        return int(labels[0])
    except Exception:
        return None


def get_goemotions_label_names(go_emotions_split: Any) -> list[str]:
    """Best-effort extraction of label names from GoEmotions features.

    Works when the dataset exposes a Sequence(ClassLabel) feature.
    Returns an empty list if unavailable.
    """
    try:
        feat = go_emotions_split.features["labels"]
        # Typically: Sequence(feature=ClassLabel(...))
        class_label = getattr(feat, "feature", None)
        names = getattr(class_label, "names", None)
        if isinstance(names, list) and all(isinstance(x, str) for x in names):
            return list(names)
    except Exception:
        pass
    return []


def build_vocab_from_train_strings(train_values: list[str]) -> dict[str, int]:
    """Deterministically map strings to IDs.

    IMPORTANT: call this on TRAIN split only to avoid distribution leakage.
    """
    uniq = sorted({v for v in train_values if isinstance(v, str) and v.strip()})
    return {v: i for i, v in enumerate(uniq)}


def map_with_vocab(value: Any, vocab: dict[str, int]) -> Optional[int]:
    if not isinstance(value, str):
        return None
    return vocab.get(value)
