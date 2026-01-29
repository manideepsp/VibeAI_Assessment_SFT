from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import read_jsonl


REQUIRED_KEYS = {"prompt_id", "prompt", "model", "response", "meta", "decoding"}


def _validate_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))

    rows = read_jsonl(path, require_exists=True)
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")

    for i, r in enumerate(rows[:10]):
        missing = REQUIRED_KEYS - set(r.keys())
        if missing:
            raise ValueError(f"Missing keys in {path} row {i}: {sorted(missing)}")

        if not str(r["prompt"]).strip():
            raise ValueError(f"Empty prompt in {path} row {i}")
        if not str(r["response"]).strip():
            raise ValueError(f"Empty response in {path} row {i}")

        model = r["model"]
        if not isinstance(model, dict) or "type" not in model:
            raise ValueError(f"Bad model field in {path} row {i} (expected object with type)")

        decoding = r["decoding"]
        if not isinstance(decoding, dict) or "max_new_tokens" not in decoding:
            raise ValueError(f"Bad decoding field in {path} row {i} (expected object with max_new_tokens)")

    print(f"ok: {path} rows={len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="artifacts/eval", help="Directory containing exported JSONL artifacts")
    ap.add_argument(
        "--check_pairs",
        action="store_true",
        help="If *_base.jsonl and *_sft.jsonl pairs exist, assert equal prompt counts and IDs.",
    )
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.exists():
        raise FileNotFoundError(str(d))

    files = sorted(d.glob("*.jsonl"))
    for name in files:
        _validate_file(name)

    if args.check_pairs:
        by_name = {p.name: p for p in files}
        prefixes = set()
        for p in files:
            if p.name.endswith("_base.jsonl"):
                prefixes.add(p.name[: -len("_base.jsonl")])

        for pref in sorted(prefixes):
            base = by_name.get(f"{pref}_base.jsonl")
            sft = by_name.get(f"{pref}_sft.jsonl")
            if base is None or sft is None:
                continue
            base_rows = read_jsonl(base, require_exists=True)
            sft_rows = read_jsonl(sft, require_exists=True)
            base_ids = [str(r.get("prompt_id")) for r in base_rows]
            sft_ids = [str(r.get("prompt_id")) for r in sft_rows]
            if len(base_ids) != len(sft_ids):
                raise ValueError(f"Pair size mismatch for {pref}: base={len(base_ids)} sft={len(sft_ids)}")
            if base_ids != sft_ids:
                raise ValueError(f"Pair prompt_id mismatch for {pref}: base and sft must align")
            print(f"ok_pair: {pref} count={len(base_ids)}")


if __name__ == "__main__":
    main()
