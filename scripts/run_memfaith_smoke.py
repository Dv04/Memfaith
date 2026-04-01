"""Run both MemFaith smoke datasets end to end."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    commands = [
        [sys.executable, "scripts/run_fever_ccs.py"],
        [sys.executable, "scripts/run_hotpotqa_ccs.py"],
        [
            sys.executable,
            "scripts/export_chunk_labels.py",
            "--log-path",
            "outputs/memfaith/fever_smoke_ccs.jsonl",
            "--output-path",
            "outputs/memfaith/fever_chunk_labels.csv",
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
