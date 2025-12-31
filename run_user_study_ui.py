"""Launcher for the redesigned Audio Preference Learning UI (user study mode)."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from preference_learning.interface.ui_study import user_main

    return user_main()


if __name__ == "__main__":
    raise SystemExit(main())
