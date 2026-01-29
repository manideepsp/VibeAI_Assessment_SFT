from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--eval_dir", type=str, required=True)
    ap.add_argument("--stats_json", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--config", type=str, default=None, help="Config JSON used for the run")
    ap.add_argument("--adapter_dir", type=str, default=None, help="Adapter dir used for SFT exports")

    ap.add_argument("--plots_dir", type=str, default="artifacts/plots")
    ap.add_argument("--plots_prefix", type=str, default=None)
    ap.add_argument("--loss_plot", type=str, default="artifacts/plots/loss_curves.png")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]

    eval_dir = Path(args.eval_dir)
    stats_path = Path(args.stats_json)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plots_dir = Path(args.plots_dir)
    plots_prefix = args.plots_prefix or args.run_id

    stats = _load_json(stats_path)
    sets: dict[str, Any] = stats.get("sets", {}) if isinstance(stats, dict) else {}

    lines: list[str] = []
    lines.append(f"# VibeAI Evaluation Report — {args.run_id}\n")

    lines.append("## Artifacts\n")
    lines.append(f"- Eval exports: `{_rel(eval_dir, repo_root)}`")
    lines.append(f"- Summary stats: `{_rel(stats_path, repo_root)}`")
    if args.adapter_dir:
        lines.append(f"- Adapter used: `{args.adapter_dir}`")
    if args.config:
        lines.append(f"- Config used: `{args.config}`")
    lines.append("")

    lines.append("## Quantitative Summary (Lightweight)\n")
    lines.append("This project exports EQ-Bench-style prompts/responses for external scoring.\n")
    lines.append("The numbers below are *simple descriptive stats* (length + heuristic refusal), not official EQ-Bench scores.\n")

    lines.append("| Set | n (base) | avg_words (base) | n (sft) | avg_words (sft) |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in ("eqbench", "qual", "redteam"):
        item = sets.get(name, {}) if isinstance(sets, dict) else {}
        base = item.get("base", {}) if isinstance(item, dict) else {}
        sft = item.get("sft", {}) if isinstance(item, dict) else {}
        lines.append(
            f"| {name} | {base.get('n', 0)} | {base.get('avg_words', 0)} | {sft.get('n', 0)} | {sft.get('avg_words', 0)} |"
        )

    redteam = sets.get("redteam", {}) if isinstance(sets, dict) else {}
    refusal = redteam.get("refusal_rate_heuristic", {}) if isinstance(redteam, dict) else {}
    if isinstance(refusal, dict):
        lines.append("\n### Red-team refusal/deflection heuristic\n")
        lines.append(f"- base: `{refusal.get('base', 0)}`")
        lines.append(f"- sft: `{refusal.get('sft', 0)}`")

    lines.append("\n## Plots\n")
    eq_len = plots_dir / f"{plots_prefix}_eqbench_len_words.png"
    avg_len = plots_dir / f"{plots_prefix}_avg_len_words.png"
    red_ref = plots_dir / f"{plots_prefix}_redteam_refusal_rate.png"
    loss = Path(args.loss_plot)

    for title, p in [
        ("Training loss curves", loss),
        ("EQ-Bench response length distribution", eq_len),
        ("Average response length by set", avg_len),
        ("Red-team refusal/deflection heuristic", red_ref),
    ]:
        if p.exists():
            rel = _rel(p, repo_root)
            lines.append(f"- {title}: `{rel}`")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                lines.append(f"\n![]({rel})\n")

    lines.append("## Reproduce\n")
    lines.append("From repo root:\n")
    lines.append("```bash")
    lines.append(f"python scripts/run_pipeline.py --config {args.config or 'config/example_config.json'} --run_id {args.run_id}")
    if args.adapter_dir:
        lines.append(f"python scripts/run_pipeline.py --config {args.config or 'config/example_config.json'} --run_id {args.run_id} --adapter_dir {args.adapter_dir}")
    lines.append("```")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
