#!/usr/bin/env python
"""
Single Entry Point for Empathetic Chatbot SFT Pipeline
=======================================================

Usage:
    python main.py train --config config/example_config.json
    python main.py eval --config config/example_config.json --run_id my_run
    python main.py eq-bench --config config/example_config.json
    python main.py pipeline --config config/example_config.json
    python main.py all --config config/example_config.json

Commands:
    train      Run QLoRA SFT training with multi-objective loss
    eval       Export evaluation artifacts (EQ-Bench, red-team, empathy)
    eq-bench   Run EQ-Bench emotional intelligence evaluation
    pipeline   Run full pipeline (eval + plots + report)
    all        Run everything (train + eval + eq-bench + pipeline)

Safety KL Configuration:
    Safety KL regularization is controlled via config:
    - safety.enabled: false  -> Skip Safety KL computation (default)
    - safety.enabled: true   -> Enable Safety KL computation
    - losses.lambda_safe     -> Weight for safety loss (0.0 to disable effect)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def bootstrap_environment() -> None:
    """Initialize environment variables for stable training."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))


def run_train(args: argparse.Namespace) -> int:
    """Run SFT training."""
    bootstrap_environment()
    
    from config import load_config
    from training import train
    
    print("=" * 60)
    print("TRAINING: QLoRA SFT with Multi-Objective Loss")
    print("=" * 60)
    print(f"Config: {args.config}")
    
    cfg = load_config(config_path=args.config)
    
    # Log safety KL status
    if cfg.safety.enabled:
        print(f"Safety KL: ENABLED (lambda_safe={cfg.losses.lambda_safe})")
    else:
        print("Safety KL: DISABLED (skipping computation)")
    
    print("=" * 60)
    
    train(cfg)
    return 0


def run_eval(args: argparse.Namespace) -> int:
    """Run evaluation exports."""
    bootstrap_environment()
    
    # Build command for run_eval.py
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_eval.py"),
        "--config", args.config,
    ]
    
    if args.run_id:
        cmd.extend(["--run_id", args.run_id])
    if args.out_dir:
        cmd.extend(["--out_dir", args.out_dir])
    if args.adapter_dir:
        cmd.extend(["--adapter_dir", args.adapter_dir])
    if args.eqbench_limit:
        cmd.extend(["--eqbench_limit", str(args.eqbench_limit)])
    if args.base_only:
        cmd.append("--base_only")
    if args.sft_only:
        cmd.append("--sft_only")
    
    print("=" * 60)
    print("EVALUATION: Exporting Artifacts")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd)
    return result.returncode


def run_eq_bench(args: argparse.Namespace) -> int:
    """Run EQ-Bench evaluation."""
    bootstrap_environment()
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_eq_bench.py"),
        "--config", args.config,
    ]
    
    if args.base_only:
        cmd.append("--base-only")
    if args.sft_only:
        cmd.append("--sft-only")
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    
    print("=" * 60)
    print("EQ-BENCH: Emotional Intelligence Evaluation")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd)
    return result.returncode


def run_pipeline(args: argparse.Namespace) -> int:
    """Run full evaluation pipeline (eval + plots + report)."""
    bootstrap_environment()
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--config", args.config,
    ]
    
    if args.run_id:
        cmd.extend(["--run_id", args.run_id])
    if args.adapter_dir:
        cmd.extend(["--adapter_dir", args.adapter_dir])
    if args.skip_eval:
        cmd.append("--skip_eval")
    if args.skip_plots:
        cmd.append("--skip_plots")
    if args.skip_report:
        cmd.append("--skip_report")
    if args.plot_loss:
        cmd.append("--plot_loss")
    
    print("=" * 60)
    print("PIPELINE: Full Evaluation Pipeline")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd)
    return result.returncode


def run_all(args: argparse.Namespace) -> int:
    """Run everything: train + eval + eq-bench."""
    print("=" * 60)
    print("RUNNING ALL: Train -> Eval -> EQ-Bench -> Pipeline")
    print("=" * 60)
    
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Train
    print("\n[1/4] TRAINING...")
    train_args = argparse.Namespace(config=args.config)
    ret = run_train(train_args)
    if ret != 0:
        print("Training failed!")
        return ret
    
    # 2. Eval
    print("\n[2/4] EVALUATION...")
    eval_args = argparse.Namespace(
        config=args.config,
        run_id=run_id,
        out_dir=None,
        adapter_dir=args.adapter_dir,
        eqbench_limit=args.eqbench_limit,
        base_only=False,
        sft_only=False,
    )
    ret = run_eval(eval_args)
    if ret != 0:
        print("Evaluation failed!")
        return ret
    
    # 3. EQ-Bench
    print("\n[3/4] EQ-BENCH...")
    eq_args = argparse.Namespace(
        config=args.config,
        base_only=False,
        sft_only=False,
        output_dir=None,
    )
    ret = run_eq_bench(eq_args)
    if ret != 0:
        print("EQ-Bench failed!")
        return ret
    
    # 4. Pipeline (plots + report)
    print("\n[4/4] PIPELINE (plots + report)...")
    pipe_args = argparse.Namespace(
        config=args.config,
        run_id=run_id,
        adapter_dir=args.adapter_dir,
        skip_eval=True,  # Already done
        skip_plots=False,
        skip_report=False,
        plot_loss=True,
    )
    ret = run_pipeline(pipe_args)
    
    print("\n" + "=" * 60)
    print("ALL COMPLETE!" if ret == 0 else "PIPELINE FAILED!")
    print("=" * 60)
    
    return ret


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Single entry point for Empathetic Chatbot SFT Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py train --config config/example_config.json
    python main.py eval --config config/example_config.json
    python main.py eq-bench --config config/example_config.json
    python main.py pipeline --config config/example_config.json
    python main.py all --config config/example_config.json

Safety KL is controlled via config file:
    "safety": { "enabled": false }  -> Skip Safety KL (default)
    "safety": { "enabled": true }   -> Enable Safety KL
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # -------------------------------------------------------------------------
    # train subcommand
    # -------------------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Run QLoRA SFT training with multi-objective loss",
    )
    train_parser.add_argument(
        "--config",
        type=str,
        default="config/example_config.json",
        help="Path to JSON config file",
    )
    
    # -------------------------------------------------------------------------
    # eval subcommand
    # -------------------------------------------------------------------------
    eval_parser = subparsers.add_parser(
        "eval",
        help="Export evaluation artifacts (EQ-Bench, red-team, empathy)",
    )
    eval_parser.add_argument("--config", type=str, default="config/example_config.json")
    eval_parser.add_argument("--run_id", type=str, default=None)
    eval_parser.add_argument("--out_dir", type=str, default=None)
    eval_parser.add_argument("--adapter_dir", type=str, default=None)
    eval_parser.add_argument("--eqbench_limit", type=int, default=10)
    eval_parser.add_argument("--base_only", action="store_true")
    eval_parser.add_argument("--sft_only", action="store_true")
    
    # -------------------------------------------------------------------------
    # eq-bench subcommand
    # -------------------------------------------------------------------------
    eq_parser = subparsers.add_parser(
        "eq-bench",
        help="Run EQ-Bench emotional intelligence evaluation",
    )
    eq_parser.add_argument("--config", type=str, default="config/example_config.json")
    eq_parser.add_argument("--base-only", action="store_true", dest="base_only")
    eq_parser.add_argument("--sft-only", action="store_true", dest="sft_only")
    eq_parser.add_argument("--output-dir", type=str, default=None, dest="output_dir")
    
    # -------------------------------------------------------------------------
    # pipeline subcommand
    # -------------------------------------------------------------------------
    pipe_parser = subparsers.add_parser(
        "pipeline",
        help="Run full pipeline (eval + plots + report)",
    )
    pipe_parser.add_argument("--config", type=str, default="config/example_config.json")
    pipe_parser.add_argument("--run_id", type=str, default=None)
    pipe_parser.add_argument("--adapter_dir", type=str, default=None)
    pipe_parser.add_argument("--skip_eval", action="store_true")
    pipe_parser.add_argument("--skip_plots", action="store_true")
    pipe_parser.add_argument("--skip_report", action="store_true")
    pipe_parser.add_argument("--plot_loss", action="store_true")
    
    # -------------------------------------------------------------------------
    # all subcommand
    # -------------------------------------------------------------------------
    all_parser = subparsers.add_parser(
        "all",
        help="Run everything: train + eval + eq-bench + pipeline",
    )
    all_parser.add_argument("--config", type=str, default="config/example_config.json")
    all_parser.add_argument("--run_id", type=str, default=None)
    all_parser.add_argument("--adapter_dir", type=str, default="artifacts/checkpoints/final/adapter")
    all_parser.add_argument("--eqbench_limit", type=int, default=10)
    
    # Parse and dispatch
    args = parser.parse_args()
    
    if args.command == "train":
        return run_train(args)
    elif args.command == "eval":
        return run_eval(args)
    elif args.command == "eq-bench":
        return run_eq_bench(args)
    elif args.command == "pipeline":
        return run_pipeline(args)
    elif args.command == "all":
        return run_all(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
