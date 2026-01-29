"""
EQ-Bench: Emotional Intelligence Benchmark for LLMs

Evaluates model ability to predict emotional intensities in dialogue scenarios.
Based on the EQ-Bench methodology: https://eqbench.com/

EQ-Bench score correlates strongly with human ratings of emotional intelligence
and empathetic response quality.
"""

import json
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)


# =============================================================================
# EQ-Bench Test Items (subset of full benchmark for demonstration)
# Full benchmark: https://github.com/EQ-bench/EQ-Bench
# =============================================================================

EQ_BENCH_ITEMS = [
    {
        "id": "eq_001",
        "scenario": "Alex just found out they didn't get the job they really wanted after three rounds of interviews.",
        "question": "On a scale of 0-10, rate the likely intensity of the following emotions Alex might be feeling:",
        "emotions": ["Disappointment", "Frustration", "Sadness", "Determination"],
        "reference_scores": [8, 6, 7, 4],  # Human-rated ground truth
    },
    {
        "id": "eq_002",
        "scenario": "Sam's best friend just told them they're moving to another country next month.",
        "question": "On a scale of 0-10, rate the likely intensity of the following emotions Sam might be feeling:",
        "emotions": ["Sadness", "Happiness for friend", "Anxiety", "Nostalgia"],
        "reference_scores": [7, 5, 6, 8],
    },
    {
        "id": "eq_003",
        "scenario": "Jordan finished a marathon for the first time, though their time was slower than expected.",
        "question": "On a scale of 0-10, rate the likely intensity of the following emotions Jordan might be feeling:",
        "emotions": ["Pride", "Exhaustion", "Disappointment", "Accomplishment"],
        "reference_scores": [8, 9, 3, 9],
    },
    {
        "id": "eq_004",
        "scenario": "Riley's partner forgot their anniversary but made up for it with a surprise party the next day.",
        "question": "On a scale of 0-10, rate the likely intensity of the following emotions Riley might be feeling:",
        "emotions": ["Initial hurt", "Surprise", "Appreciation", "Lingering disappointment"],
        "reference_scores": [5, 8, 7, 2],
    },
    {
        "id": "eq_005",
        "scenario": "Morgan found out their coworker took credit for their idea in a meeting with executives.",
        "question": "On a scale of 0-10, rate the likely intensity of the following emotions Morgan might be feeling:",
        "emotions": ["Anger", "Betrayal", "Frustration", "Determination to address it"],
        "reference_scores": [7, 8, 8, 6],
    },
    {
        "id": "eq_006",
        "scenario": "Casey's elderly parent was just diagnosed with a manageable but chronic illness.",
        "question": "On a scale of 0-10, rate the likely intensity of the following emotions Casey might be feeling:",
        "emotions": ["Worry", "Relief it's manageable", "Sadness", "Protectiveness"],
        "reference_scores": [7, 6, 5, 8],
    },
    {
        "id": "eq_007",
        "scenario": "Drew got accepted to their dream school but it means leaving their partner behind.",
        "question": "On a scale of 0-10, rate the likely intensity of the following emotions Drew might be feeling:",
        "emotions": ["Excitement", "Guilt", "Anxiety", "Hope"],
        "reference_scores": [8, 6, 7, 7],
    },
    {
        "id": "eq_008",
        "scenario": "Taylor's startup idea was rejected by investors for the fifth time this month.",
        "question": "On a scale of 0-10, rate the likely intensity of the following emotions Taylor might be feeling:",
        "emotions": ["Discouragement", "Frustration", "Self-doubt", "Resolve"],
        "reference_scores": [7, 8, 6, 5],
    },
]


@dataclass
class EQBenchResult:
    """Result from a single EQ-Bench item evaluation."""
    item_id: str
    scenario: str
    emotions: list[str]
    predicted_scores: list[float]
    reference_scores: list[float]
    absolute_errors: list[float]
    mean_absolute_error: float
    correlation: float  # Pearson correlation between predicted and reference


@dataclass
class EQBenchReport:
    """Full EQ-Bench evaluation report."""
    model_name: str
    num_items: int
    mean_absolute_error: float  # Lower is better
    mean_correlation: float  # Higher is better (max 1.0)
    eq_bench_score: float  # Normalized 0-100 score
    results: list[EQBenchResult] = field(default_factory=list)
    raw_outputs: dict = field(default_factory=dict)


def parse_emotion_scores(response: str, emotions: list[str]) -> list[float]:
    """
    Parse model response to extract emotion intensity scores.
    
    Handles various formats:
    - "Disappointment: 8"
    - "Disappointment - 8"
    - "1. Disappointment: 8/10"
    - Just numbers in order
    """
    scores = []
    
    for emotion in emotions:
        # Try to find "Emotion: X" or "Emotion - X" pattern
        patterns = [
            rf"{re.escape(emotion)}\s*[:\-]\s*(\d+(?:\.\d+)?)",
            rf"{re.escape(emotion.lower())}\s*[:\-]\s*(\d+(?:\.\d+)?)",
            rf"\b{re.escape(emotion)}\b.*?(\d+(?:\.\d+)?)",
        ]
        
        score = None
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                # Clamp to 0-10 range
                score = max(0.0, min(10.0, score))
                break
        
        if score is None:
            # Default to middle value if parsing fails
            score = 5.0
            logger.warning(f"Could not parse score for emotion '{emotion}', defaulting to 5.0")
        
        scores.append(score)
    
    return scores


def compute_pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    
    ss_x = sum((xi - mean_x) ** 2 for xi in x)
    ss_y = sum((yi - mean_y) ** 2 for yi in y)
    
    denominator = (ss_x * ss_y) ** 0.5
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def build_eq_bench_prompt(item: dict) -> str:
    """Build the prompt for an EQ-Bench item."""
    emotions_list = "\n".join(f"- {e}" for e in item["emotions"])
    
    prompt = f"""You are an expert in emotional intelligence. Read the following scenario and rate the likely emotional intensities.

Scenario: {item["scenario"]}

{item["question"]}

Emotions to rate:
{emotions_list}

Provide your ratings in the format "Emotion: X" where X is a number from 0 to 10.
0 = not feeling this emotion at all
10 = feeling this emotion extremely intensely

Your ratings:"""
    
    return prompt


def run_eq_bench_item(
    model,
    tokenizer,
    item: dict,
    device: str = "cuda",
    max_new_tokens: int = 256,
) -> EQBenchResult:
    """Run a single EQ-Bench item and compute metrics."""
    
    prompt = build_eq_bench_prompt(item)
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,  # Low temp for more consistent ratings
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    # Parse scores
    predicted_scores = parse_emotion_scores(response, item["emotions"])
    reference_scores = item["reference_scores"]
    
    # Compute metrics
    absolute_errors = [
        abs(pred - ref) for pred, ref in zip(predicted_scores, reference_scores)
    ]
    mean_absolute_error = sum(absolute_errors) / len(absolute_errors)
    correlation = compute_pearson_correlation(predicted_scores, reference_scores)
    
    return EQBenchResult(
        item_id=item["id"],
        scenario=item["scenario"],
        emotions=item["emotions"],
        predicted_scores=predicted_scores,
        reference_scores=reference_scores,
        absolute_errors=absolute_errors,
        mean_absolute_error=mean_absolute_error,
        correlation=correlation,
    )


def run_eq_bench(
    model,
    tokenizer,
    items: list[dict] = None,
    device: str = "cuda",
    max_new_tokens: int = 256,
    model_name: str = "unknown",
) -> EQBenchReport:
    """
    Run full EQ-Bench evaluation.
    
    Args:
        model: HuggingFace model (can be PeftModel)
        tokenizer: Tokenizer
        items: List of EQ-Bench items (defaults to built-in set)
        device: Device to run on
        max_new_tokens: Max tokens to generate per item
        model_name: Name for the report
        
    Returns:
        EQBenchReport with all results and aggregate scores
    """
    if items is None:
        items = EQ_BENCH_ITEMS
    
    results = []
    raw_outputs = {}
    
    logger.info(f"Running EQ-Bench on {len(items)} items...")
    
    for i, item in enumerate(items):
        logger.info(f"  [{i+1}/{len(items)}] {item['id']}")
        
        try:
            result = run_eq_bench_item(
                model, tokenizer, item, device, max_new_tokens
            )
            results.append(result)
            raw_outputs[item["id"]] = {
                "predicted": result.predicted_scores,
                "reference": result.reference_scores,
                "mae": result.mean_absolute_error,
                "correlation": result.correlation,
            }
        except Exception as e:
            logger.error(f"Error on item {item['id']}: {e}")
            continue
    
    if not results:
        raise RuntimeError("No EQ-Bench items completed successfully")
    
    # Aggregate metrics
    mean_mae = sum(r.mean_absolute_error for r in results) / len(results)
    mean_corr = sum(r.correlation for r in results) / len(results)
    
    # Convert to 0-100 score
    # MAE of 0 = score 100, MAE of 10 = score 0
    # Correlation of 1 = bonus, correlation of -1 = penalty
    mae_component = max(0, 100 - (mean_mae * 10))  # 0-100
    corr_component = (mean_corr + 1) / 2 * 20  # 0-20 bonus
    eq_bench_score = min(100, mae_component + corr_component)
    
    return EQBenchReport(
        model_name=model_name,
        num_items=len(results),
        mean_absolute_error=mean_mae,
        mean_correlation=mean_corr,
        eq_bench_score=eq_bench_score,
        results=results,
        raw_outputs=raw_outputs,
    )


def load_model_for_eq_bench(
    base_model_id: str,
    adapter_path: Optional[str] = None,
    device: str = "cuda",
    load_in_4bit: bool = True,
) -> tuple:
    """
    Load model and tokenizer for EQ-Bench evaluation.
    
    Args:
        base_model_id: HuggingFace model ID
        adapter_path: Path to LoRA adapter (optional)
        device: Device to load on
        load_in_4bit: Use 4-bit quantization
        
    Returns:
        (model, tokenizer) tuple
    """
    from transformers import BitsAndBytesConfig
    
    logger.info(f"Loading base model: {base_model_id}")
    
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    if adapter_path:
        logger.info(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    
    model.eval()
    
    return model, tokenizer


def save_eq_bench_report(report: EQBenchReport, output_path: Path) -> None:
    """Save EQ-Bench report to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_dict = {
        "model_name": report.model_name,
        "num_items": report.num_items,
        "mean_absolute_error": report.mean_absolute_error,
        "mean_correlation": report.mean_correlation,
        "eq_bench_score": report.eq_bench_score,
        "results": [
            {
                "item_id": r.item_id,
                "scenario": r.scenario,
                "emotions": r.emotions,
                "predicted_scores": r.predicted_scores,
                "reference_scores": r.reference_scores,
                "absolute_errors": r.absolute_errors,
                "mean_absolute_error": r.mean_absolute_error,
                "correlation": r.correlation,
            }
            for r in report.results
        ],
        "raw_outputs": report.raw_outputs,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    
    logger.info(f"Saved EQ-Bench report to {output_path}")


def print_eq_bench_report(report: EQBenchReport) -> None:
    """Print formatted EQ-Bench report to console."""
    print("\n" + "=" * 60)
    print("EQ-BENCH EVALUATION REPORT")
    print("=" * 60)
    print(f"Model: {report.model_name}")
    print(f"Items evaluated: {report.num_items}")
    print("-" * 60)
    print(f"Mean Absolute Error: {report.mean_absolute_error:.3f} (lower is better)")
    print(f"Mean Correlation:    {report.mean_correlation:.3f} (higher is better)")
    print(f"EQ-Bench Score:      {report.eq_bench_score:.1f}/100")
    print("-" * 60)
    
    print("\nPer-Item Results:")
    for r in report.results:
        print(f"\n  {r.item_id}: MAE={r.mean_absolute_error:.2f}, r={r.correlation:.2f}")
        for i, emotion in enumerate(r.emotions):
            print(f"    {emotion}: pred={r.predicted_scores[i]:.1f}, ref={r.reference_scores[i]:.1f}")
    
    print("\n" + "=" * 60)


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Run EQ-Bench evaluation")
    parser.add_argument("--base-model", type=str, required=True, help="Base model ID")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--output", type=str, default="artifacts/eval/eq_bench.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model_for_eq_bench(
        base_model_id=args.base_model,
        adapter_path=args.adapter,
        device=args.device,
        load_in_4bit=not args.no_4bit,
    )
    
    # Run evaluation
    model_name = args.adapter if args.adapter else args.base_model
    report = run_eq_bench(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        device=args.device,
    )
    
    # Output
    print_eq_bench_report(report)
    save_eq_bench_report(report, Path(args.output))
