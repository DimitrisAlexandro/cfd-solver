"""Run an AeroLab tool from the command line without installing the package:

    python src/run.py <tool> [arguments]
    python src/run.py              (lists the tools)

The same as the `aerolab` command after `pip install -e .` (see aerolab/cli.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aerolab.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
