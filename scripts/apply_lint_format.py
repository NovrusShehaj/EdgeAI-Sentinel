#!/usr/bin/env python3
"""Apply the repo Ruff and Black configs to the production-readiness sources."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUFF_TARGETS = ["edge", "training", "tests", "scripts", "benchmarks"]
BLACK_TARGETS = ["edge", "training", "tests", "scripts"]


def _tool(name: str) -> Path:
    candidate = ROOT / ".venv" / "bin" / name
    return candidate if candidate.is_file() else Path(name)


def _apply_black_inprocess(check: bool) -> int:
    from black import FileMode, WriteBack, format_file_in_place
    from black.mode import TargetVersion

    mode = FileMode(line_length=100, target_versions={TargetVersion.PY311})
    write_back = WriteBack.DIFF if check else WriteBack.YES
    changed = False
    for target in BLACK_TARGETS:
        root = ROOT / target
        paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in paths:
            if format_file_in_place(path, fast=False, mode=mode, write_back=write_back):
                changed = True
                print(f"black would reformat {path.relative_to(ROOT)}")
    return 1 if check and changed else 0


def main() -> int:
    ruff = _tool("ruff")
    black = _tool("black")
    check = "--check" in sys.argv
    ruff_cmd = [str(ruff), "check", *RUFF_TARGETS]
    if not check:
        ruff_cmd.insert(2, "--fix")
    ruff_proc = subprocess.run(ruff_cmd, cwd=ROOT, check=False)
    try:
        black_code = _apply_black_inprocess(check)
    except Exception as exc:
        print(f"in-process black failed ({exc}); falling back to CLI", file=sys.stderr)
        black_cmd = [str(black), *BLACK_TARGETS]
        if check:
            black_cmd.insert(1, "--check")
        black_code = subprocess.run(black_cmd, cwd=ROOT, check=False).returncode
    return ruff_proc.returncode or black_code


if __name__ == "__main__":
    raise SystemExit(main())
