from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import index_by_key, md_escape, read_jsonl

def _render_pair_table(title: str, base_rows: list[dict[str, Any]], sft_rows: list[dict[str, Any]], limit: int = 5) -> str:
    base = index_by_key(base_rows, key="prompt_id")
    sft = index_by_key(sft_rows, key="prompt_id")
    ids = list(dict.fromkeys(list(base.keys()) + list(sft.keys())))
    ids = ids[:limit]

    lines: list[str] = []
    lines.append(f"## {title}\n")
    if not ids:
        lines.append("No artifacts found for this section.\n")
        return "\n".join(lines)

    lines.append("| Prompt ID | Prompt | Base | SFT |")
    lines.append("|---|---|---|---|")

    for pid in ids:
        p = base.get(pid, sft.get(pid, {}))
        prompt = md_escape(str(p.get("prompt", "")))
        base_resp = md_escape(str(base.get(pid, {}).get("response", "")))
        sft_resp = md_escape(str(sft.get(pid, {}).get("response", "")))
        lines.append(f"| {pid} | {prompt} | {base_resp} | {sft_resp} |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=str, default="artifacts/eval", help="Directory containing exported JSONL eval artifacts")
    ap.add_argument("--out", type=str, default="artifacts/report.md", help="Output report markdown path")
    ap.add_argument("--limit", type=int, default=5, help="Max items per section")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eq_base = read_jsonl(eval_dir / "eqbench_base.jsonl")
    eq_sft = read_jsonl(eval_dir / "eqbench_sft.jsonl")
    q_base = read_jsonl(eval_dir / "qual_base.jsonl")
    q_sft = read_jsonl(eval_dir / "qual_sft.jsonl")
    r_base = read_jsonl(eval_dir / "redteam_base.jsonl")
    r_sft = read_jsonl(eval_dir / "redteam_sft.jsonl")

    lines: list[str] = []
    lines.append("# VibeAI SFT Report (Template)\n")
    lines.append("## Quantitative (EQ-Bench 3)\n")
    lines.append("Fill in EQ-Bench 3 scores after running the official scorer on `eqbench_*.jsonl`.\n")
    lines.append("- Base: (raw / normalized / Elo)\n")
    lines.append("- SFT: (raw / normalized / Elo)\n")
    lines.append("\n")

    lines.append(_render_pair_table("Qualitative Examples (Base vs SFT)", q_base, q_sft, limit=args.limit))
    lines.append(_render_pair_table("Safety Sheet (Red-team prompts)", r_base, r_sft, limit=args.limit))

    lines.append("## Ablations (at least two)\n")
    lines.append("Suggested runs:\n")
    lines.append("- Remove emotion head: set `losses.lambda_emo=0.0` and retrain\n")
    lines.append("- Remove strategy head: set `losses.lambda_strat=0.0` and retrain\n")
    lines.append("\n")

    lines.append("## Notes\n")
    lines.append("- Ensure base vs SFT use identical decoding params (temperature/top-p/max tokens).\n")
    lines.append("- Keep prompts identical and use the same tokenizer/chat template.\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
