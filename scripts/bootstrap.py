from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def bootstrap(*, set_thread_env: bool = True) -> Path:
    """Normalize runtime environment for repo scripts.

    - Sets conservative thread/env defaults for Windows stability.
    - Loads secrets from .env (e.g., HF_TOKEN).
    - Ensures REPO_ROOT is on sys.path so root modules can be imported.

    Returns:
        repo_root
    """

    repo_root = Path(__file__).resolve().parents[1]

    if set_thread_env:
        # Keep threading extremely conservative to avoid "OS can't spawn worker thread"
        # errors on constrained Windows setups.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("RAYON_NUM_THREADS", "1")
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Load environment variables from a local .env file (if present), so HF_TOKEN
    # and similar secrets don't need to be hard-coded.
    load_dotenv(repo_root / ".env")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    return repo_root
