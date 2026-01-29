from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

# Allow running this script directly from the scripts/ folder.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config  # noqa: E402
from dataset import load_canonical_datasets  # noqa: E402
from mixture import TemperatureMixtureSampler  # noqa: E402


def main() -> None:
    cfg = load_config(overrides={"datasets": {"max_train_examples": 200, "max_eval_examples": 50}})
    tok = AutoTokenizer.from_pretrained(cfg.model.base_model_id, use_fast=True)

    canonical = load_canonical_datasets(cfg, tok)

    # Split the canonical train dataset by provenance.
    sources = sorted(set(canonical.train["source"]))
    by_source = {}
    for s in sources:
        idxs = [i for i, v in enumerate(canonical.train["source"]) if v == s]
        by_source[s] = canonical.train.select(idxs)

    sampler = TemperatureMixtureSampler(by_source, alpha=cfg.datasets.mixture_alpha, seed=cfg.train.seed, max_examples=5000)
    print("sizes:", sampler.sizes)
    print("probs:", {k: round(v, 4) for k, v in sampler.probs.items()})

    counts = Counter()
    for ex in sampler:
        counts[ex["mixture_source"]] += 1

    print("observed_counts:", dict(counts))
    total = sum(counts.values())
    print("observed_probs:", {k: round(v / total, 4) for k, v in counts.items()})


if __name__ == "__main__":
    main()
