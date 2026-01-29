"""
Decoding Policy: Style Tokens and Two-Step Controller

Implements:
1. Short style tokens (e.g., <tone:warm><persona:best friend>)
2. Two-step controller with internal reflection
3. Safety-triggered re-decode with directive/advice token penalties

The two-step controller:
1. First generates an internal reflection (hidden from user)
2. Validates reply includes: (i) acknowledgment, (ii) feeling-naming, (iii) follow-up question
3. If safety rule triggered, re-decodes with stronger penalties on directive/advice tokens
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Style Tokens
# =============================================================================

class Tone(Enum):
    """Supported tone style tokens."""
    WARM = "warm"
    GENTLE = "gentle"
    SUPPORTIVE = "supportive"
    CALM = "calm"
    ENCOURAGING = "encouraging"
    EMPATHETIC = "empathetic"


class Persona(Enum):
    """Supported persona style tokens."""
    BEST_FRIEND = "best friend"
    COUNSELOR = "counselor"
    MENTOR = "mentor"
    PEER = "peer"
    COMPANION = "companion"


@dataclass
class StyleConfig:
    """Configuration for style tokens."""
    tone: Tone = Tone.WARM
    persona: Persona = Persona.BEST_FRIEND
    
    def to_prefix(self) -> str:
        """Convert to style token prefix string."""
        return f"<tone:{self.tone.value}><persona:{self.persona.value}>"
    
    @classmethod
    def from_string(cls, style_str: str) -> "StyleConfig":
        """Parse style config from string like '<tone:warm><persona:best friend>'."""
        tone_match = re.search(r"<tone:(\w+)>", style_str)
        persona_match = re.search(r"<persona:([^>]+)>", style_str)
        
        tone = Tone(tone_match.group(1)) if tone_match else Tone.WARM
        persona = Persona(persona_match.group(1)) if persona_match else Persona.BEST_FRIEND
        
        return cls(tone=tone, persona=persona)


# =============================================================================
# Reflection Validation
# =============================================================================

@dataclass
class ReflectionCheck:
    """Result of checking if a response meets quality criteria."""
    has_acknowledgment: bool = False
    has_feeling_naming: bool = False
    has_follow_up_question: bool = False
    
    @property
    def is_valid(self) -> bool:
        """Check if all criteria are met."""
        return self.has_acknowledgment and self.has_feeling_naming and self.has_follow_up_question
    
    @property
    def missing(self) -> list[str]:
        """Get list of missing criteria."""
        missing = []
        if not self.has_acknowledgment:
            missing.append("acknowledgment")
        if not self.has_feeling_naming:
            missing.append("feeling-naming")
        if not self.has_follow_up_question:
            missing.append("follow-up question")
        return missing


# Acknowledgment phrases that show understanding
ACKNOWLEDGMENT_PATTERNS = [
    r"\bi hear you\b",
    r"\bi understand\b",
    r"\bthat (sounds|must be|seems)\b",
    r"\bi can (see|imagine|understand)\b",
    r"\bthank you for sharing\b",
    r"\bi appreciate you\b",
    r"\bit makes sense\b",
    r"\bthat's (hard|difficult|tough|challenging)\b",
    r"\bi'm (sorry|here)\b",
    r"\bwow\b",
    r"\boh\b",
]

# Feeling-naming patterns
FEELING_PATTERNS = [
    r"\bfeel(ing|s)?\s+(like|as if|\w+ed|\w+ing)\b",
    r"\b(sad|happy|angry|frustrated|anxious|worried|scared|hurt|lonely|overwhelmed)\b",
    r"\b(disappointed|excited|nervous|stressed|confused|upset|hopeful)\b",
    r"\b(emotion|feeling)s?\b",
    r"\byour (pain|frustration|sadness|worry|anxiety)\b",
    r"\bthat (hurt|stings|wounds)\b",
]

# Follow-up question patterns
FOLLOW_UP_PATTERNS = [
    r"\?$",  # Ends with question mark
    r"\bwhat\s+(do|did|would|could|can|about|happened)\b.*\?",
    r"\bhow\s+(do|did|are|is|would|can|about)\b.*\?",
    r"\bwould you (like|want|mind)\b.*\?",
    r"\bcan you (tell|share|help)\b.*\?",
    r"\bdo you (want|feel|think|need)\b.*\?",
    r"\bis there\b.*\?",
    r"\bwhat's\b.*\?",
]


def check_reflection(response: str) -> ReflectionCheck:
    """
    Check if a response meets the three-part criteria:
    (i) acknowledgment, (ii) feeling-naming, (iii) follow-up question
    """
    response_lower = response.lower()
    
    # Check acknowledgment
    has_acknowledgment = any(
        re.search(pattern, response_lower)
        for pattern in ACKNOWLEDGMENT_PATTERNS
    )
    
    # Check feeling-naming
    has_feeling_naming = any(
        re.search(pattern, response_lower)
        for pattern in FEELING_PATTERNS
    )
    
    # Check follow-up question
    has_follow_up = any(
        re.search(pattern, response, re.IGNORECASE)
        for pattern in FOLLOW_UP_PATTERNS
    )
    
    return ReflectionCheck(
        has_acknowledgment=has_acknowledgment,
        has_feeling_naming=has_feeling_naming,
        has_follow_up_question=has_follow_up,
    )


# =============================================================================
# Safety Detection
# =============================================================================

# Directive/advice phrases to penalize when safety triggered
DIRECTIVE_PATTERNS = [
    r"\byou should\b",
    r"\byou need to\b",
    r"\byou must\b",
    r"\byou have to\b",
    r"\bjust (do|try|think|be|stop)\b",
    r"\bwhy don't you\b",
    r"\bwhy not\b",
    r"\bhave you tried\b",
    r"\bmy advice\b",
    r"\bi (suggest|recommend|think you should)\b",
    r"\bthe (best|right) thing to do\b",
    r"\bstop (being|feeling|doing)\b",
    r"\bdon't (be|feel|worry)\b",
    r"\bget over\b",
    r"\bmove on\b",
    r"\blook on the bright side\b",
    r"\bat least\b",
    r"\bthink positive\b",
]

# Safety triggers (from safety_teacher.py / label_policy.py)
SAFETY_TRIGGERS = [
    r"\bkill (myself|yourself)\b",
    r"\bsuicid(e|al)\b",
    r"\bharm (myself|yourself)\b",
    r"\bself[- ]harm\b",
    r"\bend (my|it all)\b",
    r"\bwant to die\b",
    r"\bnot worth living\b",
    r"\bbetter off dead\b",
    r"\babuse\b",
    r"\bviolence\b",
]


def check_safety_triggered(user_input: str) -> bool:
    """Check if user input contains safety-sensitive content."""
    input_lower = user_input.lower()
    return any(
        re.search(pattern, input_lower)
        for pattern in SAFETY_TRIGGERS
    )


def count_directive_phrases(response: str) -> int:
    """Count number of directive/advice phrases in response."""
    response_lower = response.lower()
    return sum(
        1 for pattern in DIRECTIVE_PATTERNS
        if re.search(pattern, response_lower)
    )


# =============================================================================
# Logits Processors for Controlled Decoding
# =============================================================================

class DirectivePenaltyProcessor(LogitsProcessor):
    """
    Penalize tokens that commonly start directive/advice phrases.
    Used when safety rule is triggered to encourage supportive vs prescriptive responses.
    """
    
    # Tokens to penalize (common starts of directive phrases)
    PENALTY_TOKENS = [
        "you", "should", "must", "need", "have", "try", "just",
        "why", "don't", "stop", "get", "move", "think", "advice",
        "suggest", "recommend", "better", "best", "right",
    ]
    
    def __init__(self, tokenizer, penalty: float = 2.0):
        """
        Args:
            tokenizer: Tokenizer to get token IDs
            penalty: Penalty to apply (subtracted from logits)
        """
        self.penalty = penalty
        self.penalty_token_ids = set()
        
        for word in self.PENALTY_TOKENS:
            # Get IDs for various casings and with/without space prefix
            for variant in [word, word.capitalize(), f" {word}", f" {word.capitalize()}"]:
                tokens = tokenizer.encode(variant, add_special_tokens=False)
                self.penalty_token_ids.update(tokens)
    
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for token_id in self.penalty_token_ids:
            if token_id < scores.shape[-1]:
                scores[:, token_id] -= self.penalty
        return scores


class EmpathyBoostProcessor(LogitsProcessor):
    """
    Boost tokens associated with empathetic language.
    """
    
    BOOST_TOKENS = [
        "understand", "hear", "feel", "sounds", "hard", "difficult",
        "appreciate", "share", "tell", "how", "what", "sorry",
        "here", "support", "care", "?",
    ]
    
    def __init__(self, tokenizer, boost: float = 1.0):
        self.boost = boost
        self.boost_token_ids = set()
        
        for word in self.BOOST_TOKENS:
            for variant in [word, f" {word}"]:
                tokens = tokenizer.encode(variant, add_special_tokens=False)
                self.boost_token_ids.update(tokens)
    
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for token_id in self.boost_token_ids:
            if token_id < scores.shape[-1]:
                scores[:, token_id] += self.boost
        return scores


# =============================================================================
# Two-Step Controller
# =============================================================================

REFLECTION_PROMPT_TEMPLATE = """<|im_start|>system
You are a supportive companion. Before responding to the user, generate a one-line internal reflection to ensure your reply includes:
1. Acknowledgment of their situation
2. Naming or validating their feelings
3. A gentle follow-up question

Format: [REFLECTION: your one-line plan]
Then provide your actual response.
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
{style_prefix}[REFLECTION: """


DIRECT_RESPONSE_TEMPLATE = """<|im_start|>system
You are a warm, supportive companion who responds like a best friend. Your response should:
1. Acknowledge what they shared
2. Name or validate their feelings
3. Ask a gentle follow-up question

Keep responses concise and conversational.
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
{style_prefix}"""


@dataclass
class DecodingResult:
    """Result from the two-step decoding controller."""
    response: str
    reflection: Optional[str] = None
    reflection_check: Optional[ReflectionCheck] = None
    safety_triggered: bool = False
    redecoded: bool = False
    generation_config: dict = field(default_factory=dict)


class TwoStepController:
    """
    Two-step controller for empathetic response generation.
    
    Step 1: Generate internal reflection (hidden from user)
    Step 2: Generate response validated against reflection criteria
    
    If safety rule triggered, re-decode with stronger directive penalties.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        style_config: StyleConfig = None,
        device: str = "cuda",
        max_reflection_tokens: int = 64,
        max_response_tokens: int = 256,
        directive_penalty: float = 2.0,
        empathy_boost: float = 1.0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_reflection: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.style_config = style_config or StyleConfig()
        self.device = device
        self.max_reflection_tokens = max_reflection_tokens
        self.max_response_tokens = max_response_tokens
        self.directive_penalty = directive_penalty
        self.empathy_boost = empathy_boost
        self.temperature = temperature
        self.top_p = top_p
        self.enable_reflection = enable_reflection
        
        # Build logits processors
        self.empathy_processor = EmpathyBoostProcessor(tokenizer, boost=empathy_boost)
        self.directive_processor = DirectivePenaltyProcessor(tokenizer, penalty=directive_penalty)
    
    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
        logits_processors: LogitsProcessorList = None,
        stop_strings: list[str] = None,
    ) -> str:
        """Generate text with optional logits processing."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        
        if logits_processors:
            gen_kwargs["logits_processor"] = logits_processors
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        # Handle stop strings
        if stop_strings:
            for stop_str in stop_strings:
                if stop_str in response:
                    response = response.split(stop_str)[0]
        
        return response.strip()
    
    def _generate_with_reflection(
        self,
        user_message: str,
        safety_mode: bool = False,
    ) -> tuple[str, str]:
        """
        Generate response with internal reflection.
        
        Returns:
            (reflection, response) tuple
        """
        style_prefix = self.style_config.to_prefix()
        
        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            user_message=user_message,
            style_prefix=style_prefix,
        )
        
        # Build processors
        processors = LogitsProcessorList([self.empathy_processor])
        if safety_mode:
            processors.append(self.directive_processor)
        
        # Generate with reflection
        full_output = self._generate(
            prompt,
            max_new_tokens=self.max_reflection_tokens + self.max_response_tokens,
            logits_processors=processors,
            stop_strings=["<|im_end|>", "<|im_start|>"],
        )
        
        # Parse reflection and response
        if "]" in full_output:
            parts = full_output.split("]", 1)
            reflection = parts[0].strip()
            response = parts[1].strip() if len(parts) > 1 else ""
        else:
            reflection = ""
            response = full_output
        
        return reflection, response
    
    def _generate_direct(
        self,
        user_message: str,
        safety_mode: bool = False,
    ) -> str:
        """Generate response without reflection step."""
        style_prefix = self.style_config.to_prefix()
        
        prompt = DIRECT_RESPONSE_TEMPLATE.format(
            user_message=user_message,
            style_prefix=style_prefix,
        )
        
        processors = LogitsProcessorList([self.empathy_processor])
        if safety_mode:
            processors.append(self.directive_processor)
        
        response = self._generate(
            prompt,
            max_new_tokens=self.max_response_tokens,
            logits_processors=processors,
            stop_strings=["<|im_end|>", "<|im_start|>"],
        )
        
        return response
    
    def generate(self, user_message: str) -> DecodingResult:
        """
        Generate empathetic response with two-step control.
        
        Args:
            user_message: User's input message
            
        Returns:
            DecodingResult with response and metadata
        """
        # Check for safety trigger
        safety_triggered = check_safety_triggered(user_message)
        
        if self.enable_reflection:
            # Step 1: Generate with reflection
            reflection, response = self._generate_with_reflection(
                user_message, safety_mode=safety_triggered
            )
            
            # Step 2: Validate reflection criteria
            check = check_reflection(response)
            
            # If criteria not met or has too many directives, re-decode
            directive_count = count_directive_phrases(response)
            needs_redecode = (
                (safety_triggered and directive_count > 0)
                or (not check.is_valid and directive_count > 1)
            )
            
            if needs_redecode:
                logger.info(f"Re-decoding: safety={safety_triggered}, directives={directive_count}")
                # Re-decode with stronger penalties
                _, response = self._generate_with_reflection(
                    user_message, safety_mode=True
                )
                check = check_reflection(response)
                redecoded = True
            else:
                redecoded = False
            
            return DecodingResult(
                response=response,
                reflection=reflection,
                reflection_check=check,
                safety_triggered=safety_triggered,
                redecoded=redecoded,
                generation_config={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "style": self.style_config.to_prefix(),
                    "directive_penalty": self.directive_penalty if safety_triggered else 0,
                },
            )
        else:
            # Direct generation without reflection
            response = self._generate_direct(user_message, safety_mode=safety_triggered)
            
            return DecodingResult(
                response=response,
                reflection=None,
                reflection_check=check_reflection(response),
                safety_triggered=safety_triggered,
                redecoded=False,
                generation_config={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "style": self.style_config.to_prefix(),
                },
            )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_controller(
    model,
    tokenizer,
    tone: str = "warm",
    persona: str = "best friend",
    **kwargs,
) -> TwoStepController:
    """
    Create a TwoStepController with specified style.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        tone: Tone style (warm, gentle, supportive, etc.)
        persona: Persona style (best friend, counselor, etc.)
        **kwargs: Additional arguments for TwoStepController
        
    Returns:
        Configured TwoStepController
    """
    style_config = StyleConfig(
        tone=Tone(tone),
        persona=Persona(persona),
    )
    return TwoStepController(model, tokenizer, style_config=style_config, **kwargs)


# =============================================================================
# CLI Demo
# =============================================================================

if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Two-Step Decoding Controller Demo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--tone", type=str, default="warm")
    parser.add_argument("--persona", type=str, default="best friend")
    parser.add_argument("--no-reflection", action="store_true")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    
    model.eval()
    
    # Create controller
    controller = create_controller(
        model, tokenizer,
        tone=args.tone,
        persona=args.persona,
        enable_reflection=not args.no_reflection,
    )
    
    print(f"\nStyle: {controller.style_config.to_prefix()}")
    print("Enter messages (Ctrl+C to exit):\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            
            result = controller.generate(user_input)
            
            print(f"\n[Safety: {result.safety_triggered}] [Redecoded: {result.redecoded}]")
            if result.reflection:
                print(f"[Reflection: {result.reflection}]")
            if result.reflection_check:
                print(f"[Criteria: ack={result.reflection_check.has_acknowledgment}, "
                      f"feel={result.reflection_check.has_feeling_naming}, "
                      f"q={result.reflection_check.has_follow_up_question}]")
            print(f"Bot: {result.response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
