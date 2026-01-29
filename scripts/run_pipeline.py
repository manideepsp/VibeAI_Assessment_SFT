from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from bootstrap import bootstrap


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    bootstrap()

    ap = argparse.ArgumentParser(description="Run end-to-end evaluation exports + plots + report generation.")
    ap.add_argument("--config", type=str, default="config/example_config.json")
    ap.add_argument(
        "--adapter_dir",
        type=str,
        default="artifacts/checkpoints/final/adapter",
        help="Adapter dir for SFT exports (set empty to export base only)",
    )
    ap.add_argument("--run_id", type=str, default=None)
    ap.add_argument("--eqbench_limit", type=int, default=10)

    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--skip_validate", action="store_true")
    ap.add_argument("--skip_plots", action="store_true")
    ap.add_argument("--skip_report", action="store_true")
    ap.add_argument("--plot_loss", action="store_true", help="Also plot training loss curves from artifacts/logs")

    args = ap.parse_args(argv)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable

    eval_dir = repo_root / "artifacts" / "eval" / f"run_{run_id}"
    plots_dir = repo_root / "artifacts" / "plots"
    reports_dir = repo_root / "artifacts" / "reports"

    eval_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1) Export eval artifacts
    if not args.skip_eval:
        eval_cmd = [
            py,
            str(repo_root / "scripts" / "run_eval.py"),
            "--config",
            args.config,
            "--out_dir",
            str(eval_dir),
            "--eqbench",
            "--qual",
            "--redteam",
            "--eqbench_limit",
            str(int(args.eqbench_limit)),
        ]

        if args.adapter_dir:
            eval_cmd += ["--adapter_dir", args.adapter_dir]
        else:
            eval_cmd += ["--skip_sft"]

        _run(eval_cmd)

    # 2) Validate exports + pair parity
    if not args.skip_validate:
        _run([py, str(repo_root / "scripts" / "validate_exports.py"), "--dir", str(eval_dir)])
        if args.adapter_dir:
            _run([py, str(repo_root / "scripts" / "validate_exports.py"), "--dir", str(eval_dir), "--check_pairs"])

    # 3) Summarize stats
    stats_path = reports_dir / f"run_{run_id}_stats.json"
    _run([py, str(repo_root / "scripts" / "summarize_eval_stats.py"), "--eval_dir", str(eval_dir), "--out", str(stats_path)])

    # 4) Plots
    if not args.skip_plots:
        _run(
            [
                py,
                str(repo_root / "scripts" / "plot_eval_comparisons.py"),
                "--eval_dir",
                str(eval_dir),
                "--out_dir",
                str(plots_dir),
                "--prefix",
                f"run_{run_id}",
            ]
        )

        if args.plot_loss:
            _run(
                [
                    py,
                    str(repo_root / "scripts" / "plot_loss_curves.py"),
                    "--train_log",
                    str(repo_root / "artifacts" / "logs" / "train_metrics.jsonl"),
                    "--out",
                    str(plots_dir / "loss_curves.png"),
                ]
            )

    # 5) Build report
    if not args.skip_report:
        report_path = reports_dir / f"run_{run_id}_report.md"
        _run(
            [
                py,
                str(repo_root / "scripts" / "build_run_report.py"),
                "--run_id",
                f"run_{run_id}",
                "--eval_dir",
                str(eval_dir),
                "--stats_json",
                str(stats_path),
                "--plots_dir",
                str(plots_dir),
                "--plots_prefix",
                f"run_{run_id}",
                "--loss_plot",
                str(plots_dir / "loss_curves.png"),
                "--out",
                str(report_path),
                "--config",
                args.config,
                "--adapter_dir",
                args.adapter_dir,
            ]
        )

    print("done")
    print(f"eval_dir={eval_dir}")
    print(f"stats={stats_path}")
    if not args.skip_report:
        print(f"report={reports_dir / f'run_{run_id}_report.md'}")


if __name__ == "__main__":
    main()
