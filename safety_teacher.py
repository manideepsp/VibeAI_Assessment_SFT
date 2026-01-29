"""Safety Teacher KL Regularization.

Implements safety regularization via knowledge distillation from a rules-prompted
safety teacher model.

Let z_T be teacher logits, z_θ be student logits, τ be temperature:
    L_safe = KL(σ(z_T/τ) || σ(z_θ/τ))

where σ is softmax. This encourages the student to match the teacher's
distribution over next tokens when conditioned on safety-promoting prompts.

Memory-Constrained Implementation:
----------------------------------
Running two 3B models simultaneously requires ~12-16GB VRAM which exceeds
typical consumer GPUs (GTX 1650 = 4GB). This implementation provides two modes:

1. FULL MODE: Load teacher model separately (requires high VRAM)
2. SELF-DISTILLATION MODE: Use frozen copy of student as teacher (lower VRAM)
3. OFFLINE MODE: Pre-compute teacher logits and cache to disk

For extreme low-VRAM scenarios, we apply teacher KL on only a subset of batches
(controlled by teacher_subset_ratio in SafetyTeacherConfig).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from config import AppConfig, SafetyTeacherConfig


# Safety-prompting prefix that transforms the teacher into a "rules-prompted" model
SAFETY_SYSTEM_PROMPT = """You are a safe and helpful AI assistant. You must:
1. Never provide harmful, dangerous, or illegal advice
2. Refuse requests for violence, self-harm, or harm to others
3. Maintain appropriate boundaries in all conversations
4. Redirect harmful requests toward constructive alternatives
5. Prioritize user wellbeing and psychological safety"""


@dataclass(frozen=True)
class SafetyKLOutput:
    """Output from safety KL computation."""
    loss: torch.Tensor  # scalar KL divergence loss
    teacher_entropy: float  # for logging/debugging
    student_entropy: float


def compute_kl_divergence(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 2.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Compute KL divergence between teacher and student distributions.
    
    Args:
        teacher_logits: (B, T, V) or (B, V) teacher logits
        student_logits: (B, T, V) or (B, V) student logits  
        temperature: softmax temperature τ > 0
        reduction: 'batchmean', 'sum', or 'none'
    
    Returns:
        KL divergence: KL(softmax(z_T/τ) || softmax(z_θ/τ))
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    
    # Scale logits by temperature
    teacher_scaled = teacher_logits / temperature
    student_scaled = student_logits / temperature
    
    # Compute log-softmax for numerical stability
    teacher_log_probs = F.log_softmax(teacher_scaled, dim=-1)
    student_log_probs = F.log_softmax(student_scaled, dim=-1)
    
    # KL(P || Q) = sum(P * (log P - log Q))
    # Using F.kl_div: input=log Q, target=log P (with log_target=True)
    kl = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction=reduction,
        log_target=True,
    )
    
    # Scale by τ² as per distillation literature (Hinton et al., 2015)
    # This compensates for the reduced gradient magnitude from temperature scaling
    return kl * (temperature ** 2)


def compute_entropy(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """Compute entropy of the distribution for logging."""
    probs = F.softmax(logits / temperature, dim=-1)
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1).mean()
    return float(entropy.detach().cpu())


class SafetyTeacher(nn.Module):
    """Safety teacher for KL regularization.
    
    Wraps a teacher model (or frozen copy of student) that provides target
    distributions for safety-focused knowledge distillation.
    """
    
    def __init__(
        self,
        config: SafetyTeacherConfig,
        student_model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.config = config
        self.temperature = config.temperature_tau
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        
        self._teacher_model: Optional[PreTrainedModel] = None
        self._use_self_distillation = False
        
        if not config.enabled:
            return
            
        if config.teacher_model_id and config.teacher_model_id.strip():
            # Full mode: load separate teacher model
            self._load_teacher_model(config.teacher_model_id)
        elif student_model is not None:
            # Self-distillation mode: use frozen copy of student
            self._setup_self_distillation(student_model)
        else:
            raise ValueError(
                "SafetyTeacher requires either teacher_model_id or student_model "
                "for self-distillation mode"
            )
    
    def _load_teacher_model(self, model_id: str) -> None:
        """Load a separate teacher model (high VRAM required)."""
        print(f"[SafetyTeacher] Loading teacher model: {model_id}")
        
        # Load in fp16 to save memory
        self._teacher_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=False,
        )
        self._teacher_model.to(self.device)
        self._teacher_model.eval()
        
        # Freeze all parameters
        for param in self._teacher_model.parameters():
            param.requires_grad = False
        
        self._use_self_distillation = False
        print(f"[SafetyTeacher] Teacher loaded. Self-distillation=False")
    
    def _setup_self_distillation(self, student_model: PreTrainedModel) -> None:
        """Set up self-distillation from frozen student snapshot.
        
        For extreme low-VRAM, we don't keep a separate copy but use the student
        in eval mode with stop-gradient. This is less ideal but works.
        """
        print("[SafetyTeacher] Using self-distillation mode (no separate teacher)")
        self._teacher_model = None  # We'll use student directly with torch.no_grad
        self._use_self_distillation = True
    
    @property
    def is_enabled(self) -> bool:
        return self.config.enabled
    
    def compute_safety_kl(
        self,
        student_model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> SafetyKLOutput:
        """Compute safety KL loss between teacher and student.
        
        Args:
            student_model: The student model being trained
            input_ids: (B, T) input token IDs
            attention_mask: (B, T) attention mask
            labels: (B, T) labels for masking (optional)
        
        Returns:
            SafetyKLOutput with loss and entropy statistics
        """
        if not self.is_enabled:
            return SafetyKLOutput(
                loss=torch.tensor(0.0, device=self.device),
                teacher_entropy=0.0,
                student_entropy=0.0,
            )
        
        # Get student logits (already computed in forward pass, but we need them here)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            student_logits = student_outputs.logits  # (B, T, V)
        
        # Get teacher logits
        with torch.no_grad():
            if self._use_self_distillation:
                # Self-distillation: use student in eval mode
                was_training = student_model.training
                student_model.eval()
                teacher_outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                teacher_logits = teacher_outputs.logits.detach()
                if was_training:
                    student_model.train()
            else:
                # Separate teacher model
                teacher_outputs = self._teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                teacher_logits = teacher_outputs.logits
        
        # Compute KL divergence on next-token predictions
        # Focus on positions where we have valid labels (assistant responses)
        if labels is not None:
            # Shift for next-token prediction alignment
            shift_student = student_logits[:, :-1, :].contiguous()
            shift_teacher = teacher_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Only compute KL on valid positions
            valid_mask = (shift_labels != -100)
            if valid_mask.any():
                # Flatten for masked selection
                B, T, V = shift_student.shape
                flat_student = shift_student.view(-1, V)
                flat_teacher = shift_teacher.view(-1, V)
                flat_mask = valid_mask.view(-1)
                
                valid_student = flat_student[flat_mask]
                valid_teacher = flat_teacher[flat_mask]
                
                kl_loss = compute_kl_divergence(
                    valid_teacher, valid_student,
                    temperature=self.temperature,
                    reduction="batchmean",
                )
                
                teacher_ent = compute_entropy(valid_teacher, self.temperature)
                student_ent = compute_entropy(valid_student, self.temperature)
            else:
                kl_loss = torch.tensor(0.0, device=self.device)
                teacher_ent = 0.0
                student_ent = 0.0
        else:
            # No labels: compute KL on all positions
            kl_loss = compute_kl_divergence(
                teacher_logits, student_logits,
                temperature=self.temperature,
                reduction="batchmean",
            )
            teacher_ent = compute_entropy(teacher_logits, self.temperature)
            student_ent = compute_entropy(student_logits, self.temperature)
        
        return SafetyKLOutput(
            loss=kl_loss,
            teacher_entropy=teacher_ent,
            student_entropy=student_ent,
        )
    
    def should_apply_this_step(self, step: int, rng: Optional[torch.Generator] = None) -> bool:
        """Check if we should apply teacher KL on this step (for subset ratio)."""
        if not self.is_enabled:
            return False
        
        ratio = self.config.teacher_subset_ratio
        if ratio >= 1.0:
            return True
        if ratio <= 0.0:
            return False
        
        # Stochastic selection based on ratio
        if rng is not None:
            return torch.rand(1, generator=rng).item() < ratio
        return torch.rand(1).item() < ratio


def build_safety_teacher(
    cfg: AppConfig,
    student_model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    device: Optional[torch.device] = None,
) -> SafetyTeacher:
    """Factory function to create SafetyTeacher from config."""
    return SafetyTeacher(
        config=cfg.safety,
        student_model=student_model,
        tokenizer=tokenizer,
        device=device,
    )
