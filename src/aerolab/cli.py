"""Run an AeroLab tool from the command line.

    aerolab <tool> [arguments]                 (after pip install -e .)
    python src/run.py <tool> [arguments]       (without installing)
    aerolab airfoil polar naca0012 --re 3e6 --alpha=0:10:2 --out cases/n0012
    aerolab jobqueue worker
    aerolab                                    (lists the tools)

<tool> is a module name of the aerolab package (airfoil, polar, adjoint, fiml_multi, ...);
`python -m aerolab.<package>.<tool>` does the same.
"""
import runpy
import sys
from pathlib import Path

import aerolab


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        groups = {}
        for name, g in aerolab.MODULES.items():
            groups.setdefault(g, []).append(name)
        for g, names in groups.items():
            print(f"  {g:10s} " + ", ".join(sorted(names)))
        return 0
    name = Path(sys.argv[1]).stem
    if name not in aerolab.MODULES:
        print(f"unknown tool {sys.argv[1]!r}; aerolab (or python src/run.py) lists them", file=sys.stderr)
        return 2
    sys.argv = [str(aerolab.script_path(name))] + sys.argv[2:]
    runpy.run_module(aerolab.module(name), run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
