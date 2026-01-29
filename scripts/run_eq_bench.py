"""
Run EQ-Bench evaluation on base and SFT models.

Usage:
    python scripts/run_eq_bench.py --config config/example_config.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eq_bench import (
    run_eq_bench,
    load_model_for_eq_bench,
    save_eq_bench_report,
    print_eq_bench_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run EQ-Bench evaluation")
    parser.add_argument("--config", type=str, default="config/example_config.json")
    parser.add_argument("--base-only", action="store_true", help="Only evaluate base model")
    parser.add_argument("--sft-only", action="store_true", help="Only evaluate SFT model")
    parser.add_argument("--output-dir", type=str, default="artifacts/eval")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)
    
    base_model_id = config.get("base_model_id", "Qwen/Qwen2.5-3B-Instruct")
    adapter_path = Path(config.get("output_dir", "artifacts")) / "checkpoints" / "final" / "adapter"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reports = {}
    
    # Evaluate base model
    if not args.sft_only:
        logger.info("=" * 60)
        logger.info("Evaluating BASE model")
        logger.info("=" * 60)
        
        model, tokenizer = load_model_for_eq_bench(
            base_model_id=base_model_id,
            adapter_path=None,
            device=args.device,
            load_in_4bit=True,
        )
        
        report_base = run_eq_bench(
            model=model,
            tokenizer=tokenizer,
            model_name=f"Base: {base_model_id}",
            device=args.device,
        )
        
        print_eq_bench_report(report_base)
        save_eq_bench_report(report_base, output_dir / "eq_bench_base.json")
        reports["base"] = report_base
        
        # Free memory
        del model
        import torch
        torch.cuda.empty_cache()
    
    # Evaluate SFT model
    if not args.base_only:
        if adapter_path.exists():
            logger.info("=" * 60)
            logger.info("Evaluating SFT model")
            logger.info("=" * 60)
            
            model, tokenizer = load_model_for_eq_bench(
                base_model_id=base_model_id,
                adapter_path=str(adapter_path),
                device=args.device,
                load_in_4bit=True,
            )
            
            report_sft = run_eq_bench(
                model=model,
                tokenizer=tokenizer,
                model_name=f"SFT: {adapter_path}",
                device=args.device,
            )
            
            print_eq_bench_report(report_sft)
            save_eq_bench_report(report_sft, output_dir / "eq_bench_sft.json")
            reports["sft"] = report_sft
            
            del model
            torch.cuda.empty_cache()
        else:
            logger.warning(f"Adapter not found at {adapter_path}, skipping SFT evaluation")
    
    # Print comparison
    if "base" in reports and "sft" in reports:
        print("\n" + "=" * 60)
        print("EQ-BENCH COMPARISON")
        print("=" * 60)
        print(f"{'Model':<30} {'EQ-Score':>10} {'MAE':>10} {'Corr':>10}")
        print("-" * 60)
        print(f"{'Base':<30} {reports['base'].eq_bench_score:>10.1f} {reports['base'].mean_absolute_error:>10.3f} {reports['base'].mean_correlation:>10.3f}")
        print(f"{'SFT':<30} {reports['sft'].eq_bench_score:>10.1f} {reports['sft'].mean_absolute_error:>10.3f} {reports['sft'].mean_correlation:>10.3f}")
        
        delta_score = reports['sft'].eq_bench_score - reports['base'].eq_bench_score
        print("-" * 60)
        print(f"{'Δ (SFT - Base)':<30} {delta_score:>+10.1f}")
        print("=" * 60)
    
    logger.info("EQ-Bench evaluation complete!")


if __name__ == "__main__":
    main()
