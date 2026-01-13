"""Launcher that runs the user study first, then captures the user's best signal."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional


def _latest_session_dir(base_dir: Path, since: float) -> Optional[Path]:
    if not base_dir.exists():
        return None
    candidates = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        session_file = child / "session.json"
        if not session_file.exists():
            continue
        try:
            mtime = session_file.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, child))
    if not candidates:
        return None
    recent = [item for item in candidates if item[0] >= since]
    pool = recent if recent else candidates
    pool.sort(key=lambda item: item[0], reverse=True)
    return pool[0][1]


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from preference_learning.interface.ui_study import user_main

    start_time = time.time()
    user_main(auto_close_on_complete=True)

    data_dir = _latest_session_dir(repo_root / "data", start_time)
    from xbox_control import main as xbox_main

    if data_dir is None:
        print("No session data folder found; recording will go to data/bestparam/.")
        return xbox_main(["--complete-dialog"])
    return xbox_main(
        [
            "--output-dir",
            str(data_dir),
            "--filename",
            "favorite_signal.json",
            "--complete-dialog",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
