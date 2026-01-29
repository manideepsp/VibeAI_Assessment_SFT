from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import char_count, read_jsonl, refusal_rate_heuristic, word_count




def _mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs)) / float(len(xs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=str, default="artifacts/eval", help="Directory containing exported JSONLs")
    ap.add_argument("--out", type=str, default=None, help="Write a JSON summary to this path")
    args = ap.parse_args()

    d = Path(args.eval_dir)
    sets = {
        "eqbench": (d / "eqbench_base.jsonl", d / "eqbench_sft.jsonl"),
        "qual": (d / "qual_base.jsonl", d / "qual_sft.jsonl"),
        "redteam": (d / "redteam_base.jsonl", d / "redteam_sft.jsonl"),
    }

    summary: dict[str, Any] = {
        "eval_dir": str(d),
        "sets": {},
    }

    for name, (base_p, sft_p) in sets.items():
        base_rows = read_jsonl(base_p)
        sft_rows = read_jsonl(sft_p)

        def _stats(rows: list[dict[str, Any]]):
            wcs = [word_count(str(r.get("response", ""))) for r in rows]
            ccs = [char_count(str(r.get("response", ""))) for r in rows]
            return {
                "n": len(rows),
                "avg_words": round(_mean([float(x) for x in wcs]), 3),
                "avg_chars": round(_mean([float(x) for x in ccs]), 3),
            }

        item: dict[str, Any] = {
            "base": _stats(base_rows),
            "sft": _stats(sft_rows),
        }

        if name == "redteam":
            item["refusal_rate_heuristic"] = {
                "base": round(refusal_rate_heuristic(base_rows), 3),
                "sft": round(refusal_rate_heuristic(sft_rows), 3),
            }

        summary["sets"][name] = item

    out_path = Path(args.out) if args.out else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote={out_path}")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
