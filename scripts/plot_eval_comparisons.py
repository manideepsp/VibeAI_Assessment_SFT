from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import read_jsonl, refusal_rate_heuristic, word_count

 


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=str, default="artifacts/eval", help="Directory containing *_base.jsonl/*_sft.jsonl")
    ap.add_argument("--out_dir", type=str, default="artifacts/plots", help="Where to write PNGs")
    ap.add_argument("--prefix", type=str, default="eval_compare", help="Filename prefix")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eq_base = read_jsonl(eval_dir / "eqbench_base.jsonl")
    eq_sft = read_jsonl(eval_dir / "eqbench_sft.jsonl")
    q_base = read_jsonl(eval_dir / "qual_base.jsonl")
    q_sft = read_jsonl(eval_dir / "qual_sft.jsonl")
    r_base = read_jsonl(eval_dir / "redteam_base.jsonl")
    r_sft = read_jsonl(eval_dir / "redteam_sft.jsonl")

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with: pip install matplotlib\n"
            f"Import error: {e}"
        )

    # 1) EQ-Bench response length distribution (words) — grouped bar chart
    eq_base_wc = [word_count(str(r.get("response", ""))) for r in eq_base]
    eq_sft_wc = [word_count(str(r.get("response", ""))) for r in eq_sft]

    # Build frequency counts for grouped bars
    import numpy as np
    all_lens = sorted(set(eq_base_wc + eq_sft_wc))
    base_counts = [eq_base_wc.count(l) for l in all_lens]
    sft_counts = [eq_sft_wc.count(l) for l in all_lens]
    x = np.arange(len(all_lens))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, base_counts, width=width, label=f"base (n={len(eq_base_wc)})", alpha=0.8)
    plt.bar(x + width / 2, sft_counts, width=width, label=f"sft (n={len(eq_sft_wc)})", alpha=0.8)
    plt.xticks(x, [str(l) for l in all_lens])
    plt.xlabel("response length (words)")
    plt.ylabel("count")
    plt.title("EQ-Bench: response length distribution")
    plt.legend()
    plt.tight_layout()

    out1 = out_dir / f"{args.prefix}_eqbench_len_words.png"
    plt.savefig(out1)
    plt.close()
    print(f"wrote={out1}")

    # 2) Average response length by set and model
    sets = [
        ("eqbench", eq_base, eq_sft),
        ("qual", q_base, q_sft),
        ("redteam", r_base, r_sft),
    ]

    labels = [name for name, _, _ in sets]
    base_means = []
    sft_means = []

    for _, b, s in sets:
        base_means.append(sum(word_count(str(r.get("response", ""))) for r in b) / max(1, len(b)))
        sft_means.append(sum(word_count(str(r.get("response", ""))) for r in s) / max(1, len(s)))

    x = list(range(len(labels)))
    width = 0.38

    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], base_means, width=width, label="base")
    plt.bar([i + width / 2 for i in x], sft_means, width=width, label="sft")
    plt.xticks(x, labels)
    plt.ylabel("avg response length (words)")
    plt.title("Average response length by prompt set")
    plt.legend()
    plt.tight_layout()

    out2 = out_dir / f"{args.prefix}_avg_len_words.png"
    plt.savefig(out2)
    plt.close()
    print(f"wrote={out2}")

    # 3) Red-team refusal/deflection heuristic rate
    base_ref = refusal_rate_heuristic(r_base)
    sft_ref = refusal_rate_heuristic(r_sft)

    plt.figure(figsize=(6, 4))
    plt.bar(["base", "sft"], [base_ref, sft_ref])
    plt.ylim(0, 1.0)
    plt.ylabel("rate")
    plt.title("Red-team: refusal/deflection heuristic")
    plt.tight_layout()

    out3 = out_dir / f"{args.prefix}_redteam_refusal_rate.png"
    plt.savefig(out3)
    plt.close()
    print(f"wrote={out3}")


if __name__ == "__main__":
    main()
