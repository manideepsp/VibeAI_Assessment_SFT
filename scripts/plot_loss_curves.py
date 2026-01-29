from __future__ import annotations

import argparse
from pathlib import Path

from common import read_jsonl

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_log", type=str, default="artifacts/logs/train_metrics.jsonl")
    ap.add_argument("--out", type=str, default="artifacts/loss_curves.png")
    args = ap.parse_args()

    train_log = Path(args.train_log)
    rows = read_jsonl(train_log)

    # Keep only scalar metric rows (support both legacy and current key names)
    def _is_metric_row(r: dict) -> bool:
        return any(k in r for k in ("loss", "total_loss", "lm_loss", "emotion_loss", "strategy_loss")) and "step" in r

    rows = [r for r in rows if isinstance(r, dict) and _is_metric_row(r)]
    if not rows:
        raise SystemExit(f"No metric rows found in {train_log}")

    def _get_float(r: dict, *keys: str) -> float:
        for k in keys:
            if k in r:
                try:
                    return float(r.get(k))
                except Exception:
                    return float("nan")
        return float("nan")

    steps = [int(r["step"]) for r in rows]
    total = [_get_float(r, "loss", "total_loss") for r in rows]
    lm = [_get_float(r, "lm_loss") for r in rows]
    emo = [_get_float(r, "emotion_loss", "emo_loss") for r in rows]
    strat = [_get_float(r, "strategy_loss", "strat_loss") for r in rows]

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with: pip install matplotlib\n"
            f"Import error: {e}"
        )

    plt.figure(figsize=(10, 5))
    plt.plot(steps, total, label="total")
    plt.plot(steps, lm, label="lm")
    plt.plot(steps, emo, label="emo")
    plt.plot(steps, strat, label="strat")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training losses")
    plt.legend()
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    print(f"wrote={out}")


if __name__ == "__main__":
    main()
