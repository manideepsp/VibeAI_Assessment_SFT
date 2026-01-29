from __future__ import annotations

import argparse

from bootstrap import bootstrap


bootstrap()

from config import load_config
from training import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to JSON config override")
    args = ap.parse_args()

    cfg = load_config(config_path=args.config)
    train(cfg)


if __name__ == "__main__":
    main()
