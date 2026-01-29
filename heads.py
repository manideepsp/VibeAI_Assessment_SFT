from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = -100


def pool_last_prompt_token(hidden_states: torch.Tensor, prompt_length: torch.Tensor) -> torch.Tensor:
    """Pool per-example representation at anchor_idx = prompt_length - 1.

    This enforces: auxiliary heads predict from the *user context* (end of prompt),
    not from generated assistant tokens.

    Args:
        hidden_states: (B, T, H)
        prompt_length: (B,) int tensor (token count of prompt). Values should be in [1, T].

    Returns:
        pooled: (B, H)
    """

    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be (B,T,H); got {tuple(hidden_states.shape)}")

    if prompt_length.dim() != 1:
        raise ValueError(f"prompt_length must be (B,); got {tuple(prompt_length.shape)}")

    bsz, seq_len, _ = hidden_states.shape
    if prompt_length.numel() != bsz:
        raise ValueError(f"prompt_length batch mismatch: {prompt_length.numel()} vs {bsz}")

    # anchor_idx = prompt_length - 1; clamp to valid range.
    anchor_idx = prompt_length.to(device=hidden_states.device, dtype=torch.long) - 1
    anchor_idx = torch.clamp(anchor_idx, min=0, max=seq_len - 1)

    # Gather along sequence dimension.
    gather_idx = anchor_idx.view(bsz, 1, 1).expand(bsz, 1, hidden_states.size(-1))
    pooled = hidden_states.gather(dim=1, index=gather_idx).squeeze(1)
    return pooled


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Cross-entropy with IGNORE_INDEX masking.

    Returns a 0.0 tensor on the correct device when there are no valid labels.
    """

    if labels.dim() != 1:
        labels = labels.view(-1)

    valid = labels != ignore_index
    if not torch.any(valid):
        return torch.tensor(0.0, device=logits.device)

    return F.cross_entropy(logits[valid], labels[valid])


@dataclass(frozen=True)
class AuxLogits:
    emotion: Optional[torch.Tensor]  # (B, C_emo)
    strategy: Optional[torch.Tensor]  # (B, C_strat)


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(pooled))


class EmotionHead(ClassificationHead):
    pass


class StrategyHead(ClassificationHead):
    pass


class AuxHeads(nn.Module):
    """Container for auxiliary heads.

    Keeps heads separate from the base model (Option A), so training can:
    - run base model forward once
    - pool hidden states at prompt_length-1
    - compute emotion/strategy losses only where labels exist
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_emotions: int,
        num_strategies: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.emotion_head: Optional[EmotionHead] = None
        self.strategy_head: Optional[StrategyHead] = None

        if num_emotions > 0:
            self.emotion_head = EmotionHead(hidden_size, num_emotions, dropout=dropout)
        if num_strategies > 0:
            self.strategy_head = StrategyHead(hidden_size, num_strategies, dropout=dropout)

    def forward(self, hidden_states: torch.Tensor, prompt_length: torch.Tensor) -> AuxLogits:
        pooled = pool_last_prompt_token(hidden_states, prompt_length)

        emo_logits = self.emotion_head(pooled) if self.emotion_head is not None else None
        strat_logits = self.strategy_head(pooled) if self.strategy_head is not None else None

        return AuxLogits(emotion=emo_logits, strategy=strat_logits)
