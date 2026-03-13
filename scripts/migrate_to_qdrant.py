#!/usr/bin/env python3
"""Deprecated in v3: vectors are written to Qdrant directly by the embedding pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    print("This script is deprecated in v3.")
    print("Use `scripts/11_reprocess_pipeline.py` or `scripts/03_embed.py` to write vectors directly to Qdrant.")
    print("Legacy Supabase vector migration is no longer part of the v3 pipeline.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
