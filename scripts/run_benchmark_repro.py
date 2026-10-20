"""Run the dev benchmark twice and compare the summaries.

This is the local equivalent of the CI reproducibility gate. It uses temporary
directories so repeated runs cannot accumulate reports or compare the wrong file.

    python scripts/run_benchmark_repro.py [--split dev]
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main(argv: list[str]) -> int:
    split = "dev"
    if argv:
        if len(argv) != 2 or argv[0] != "--split":
            raise SystemExit("usage: run_benchmark_repro.py [--split <name>]")
        split = argv[1]

    recap = shutil.which("recap")
    if recap is None:
        for name in ("recap.exe", "recap"):
            candidate = Path(sys.executable).with_name(name)
            if candidate.exists():
                recap = str(candidate)
                break
    if recap is None:
        raise SystemExit("recap executable not found; activate the virtual environment first")

    script = Path(__file__).with_name("check_benchmark_repro.py")
    with tempfile.TemporaryDirectory(prefix="recap-benchmark-repro-") as temp:
        root = Path(temp)
        directories = [root / "first", root / "second"]
        for directory in directories:
            directory.mkdir()
            print(f"[repro] recap eval run --split {split} --output {directory}", flush=True)
            completed = subprocess.run(
                [recap, "eval", "run", "--split", split, "--output", str(directory)],
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                sys.stdout.write(completed.stdout)
                sys.stderr.write(completed.stderr)
                return completed.returncode

        return subprocess.call(
            [sys.executable, str(script), *(str(directory) for directory in directories)]
        )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
